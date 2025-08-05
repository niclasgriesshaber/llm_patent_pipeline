#!/usr/bin/env python3

"""
Parallel script to rerun failed pages for all PDFs that have error files.
This script processes multiple PDFs simultaneously using multiprocessing.
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal

# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CSVS_DIR = DATA_DIR / "01_dataset_construction" / "csvs"
PDF_SRC_DIR = DATA_DIR / "pdfs" / "patent_pdfs"
SCRIPT_PATH = Path(__file__).parent / "gemini-2.5-parallel.py"

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def find_pdfs_with_errors() -> List[Tuple[str, str]]:
    """
    Find all PDFs that have corresponding error files.
    
    Returns:
        List of tuples (pdf_filename, error_file_path)
    """
    pdfs_with_errors = []
    
    # Get all PDF files
    pdf_files = list(PDF_SRC_DIR.glob("*.pdf"))
    
    for pdf_path in pdf_files:
        pdf_stem = pdf_path.stem
        error_file_path = CSVS_DIR / pdf_stem / f"errors_{pdf_stem}.txt"
        
        if error_file_path.exists():
            pdfs_with_errors.append((pdf_path.name, str(error_file_path)))
    
    return pdfs_with_errors

def process_single_pdf(args_tuple: Tuple[str, str, int, int]) -> Tuple[str, bool, str]:
    """
    Process a single PDF with failed pages.
    
    Args:
        args_tuple: (pdf_filename, error_file_path, worker_id, total_workers)
        
    Returns:
        Tuple of (pdf_filename, success, error_message)
    """
    pdf_filename, error_file_path, worker_id, total_workers = args_tuple
    
    try:
        # Set up logging for this worker
        worker_logger = logging.getLogger(f"Worker-{worker_id}")
        worker_logger.info(f"[Worker-{worker_id}] Processing: {pdf_filename}")
        
        # Build the command with worker prefix for log identification
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--pdf", pdf_filename,
            "--from_error_file", "yes"
        ]
        
        # Print worker start message
        print(f"\n{'='*60}")
        print(f"WORKER-{worker_id} STARTING: {pdf_filename}")
        print(f"{'='*60}")
        
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show all output from the core script
            text=True,
            cwd=PROJECT_ROOT,
            timeout=3600  # 1 hour timeout per PDF
        )
        
        if result.returncode == 0:
            print(f"\n{'='*60}")
            print(f"WORKER-{worker_id} COMPLETED SUCCESSFULLY: {pdf_filename}")
            print(f"{'='*60}")
            worker_logger.info(f"[Worker-{worker_id}] Successfully processed: {pdf_filename}")
            return (pdf_filename, True, "")
        else:
            print(f"\n{'='*60}")
            print(f"WORKER-{worker_id} FAILED: {pdf_filename}")
            print(f"{'='*60}")
            error_msg = f"Failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f" - {result.stderr.strip()}"
            worker_logger.error(f"[Worker-{worker_id}] Failed to process {pdf_filename}: {error_msg}")
            return (pdf_filename, False, error_msg)
            
    except subprocess.TimeoutExpired:
        error_msg = f"Timeout after 1 hour"
        worker_logger.error(f"[Worker-{worker_id}] Timeout processing {pdf_filename}: {error_msg}")
        return (pdf_filename, False, error_msg)
    except Exception as e:
        error_msg = str(e)
        worker_logger.error(f"[Worker-{worker_id}] Error processing {pdf_filename}: {error_msg}")
        return (pdf_filename, False, error_msg)

def main():
    """Main function to process all PDFs with error files in parallel."""
    setup_logging()
    
    logging.info("Starting parallel rerun of failed pages for all PDFs with error files")
    logging.info("=" * 80)
    
    # Check if the main script exists
    if not SCRIPT_PATH.exists():
        logging.error(f"Main script not found: {SCRIPT_PATH}")
        sys.exit(1)
    
    # Find PDFs with error files
    pdfs_with_errors = find_pdfs_with_errors()
    
    if not pdfs_with_errors:
        logging.info("No PDFs with error files found. Nothing to process.")
        return
    
    logging.info(f"Found {len(pdfs_with_errors)} PDFs with error files to process")
    
    # Determine number of workers (use fewer workers to avoid overwhelming the API)
    max_workers = min(4, len(pdfs_with_errors), cpu_count())
    logging.info(f"Using {max_workers} parallel workers")
    logging.info("")
    
    # Prepare arguments for each worker
    worker_args = []
    for i, (pdf_filename, error_file_path) in enumerate(pdfs_with_errors):
        worker_args.append((pdf_filename, error_file_path, i + 1, len(pdfs_with_errors)))
    
    # Process PDFs in parallel
    successful = 0
    failed = 0
    start_time = time.time()
    
    logging.info(f"Starting parallel processing of {len(pdfs_with_errors)} PDFs...")
    logging.info("=" * 80)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pdf = {
            executor.submit(process_single_pdf, args): args[0] 
            for args in worker_args
        }
        
        # Process completed tasks
        for future in as_completed(future_to_pdf):
            pdf_filename = future_to_pdf[future]
            try:
                pdf_name, success, error_msg = future.result()
                if success:
                    successful += 1
                    logging.info(f"✓ Completed ({successful + failed}/{len(pdfs_with_errors)}): {pdf_name}")
                else:
                    failed += 1
                    logging.error(f"✗ Failed ({successful + failed}/{len(pdfs_with_errors)}): {pdf_name} - {error_msg}")
            except Exception as e:
                failed += 1
                logging.error(f"✗ Exception ({successful + failed}/{len(pdfs_with_errors)}): {pdf_filename} - {e}")
    
    # Summary
    total_time = time.time() - start_time
    logging.info("=" * 80)
    logging.info("PARALLEL PROCESSING SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Total PDFs with error files: {len(pdfs_with_errors)}")
    logging.info(f"Successfully processed: {successful}")
    logging.info(f"Failed to process: {failed}")
    logging.info(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logging.info(f"Average time per PDF: {total_time/len(pdfs_with_errors):.2f} seconds")
    
    if failed > 0:
        logging.warning(f"{failed} PDFs failed to process. Check the logs above for details.")
        sys.exit(1)
    else:
        logging.info("All PDFs processed successfully!")

if __name__ == "__main__":
    main() 