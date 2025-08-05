#!/usr/bin/env python3

"""
Script to automatically rerun failed pages for all PDFs that have error files.
This script finds all PDFs in the patent_pdfs directory that have corresponding error files
and reruns the failed pages for each one.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple

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
            logging.info(f"Found error file for: {pdf_path.name}")
    
    return pdfs_with_errors

def run_failed_pages_processing(pdf_filename: str) -> bool:
    """
    Run the failed pages processing for a specific PDF.
    
    Args:
        pdf_filename: Name of the PDF file to process
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logging.info(f"Processing failed pages for: {pdf_filename}")
        
        # Build the command
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--pdf", pdf_filename,
            "--from_error_file", "yes"
        ]
        
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=False,  # Let output go to console
            text=True,
            cwd=PROJECT_ROOT
        )
        
        if result.returncode == 0:
            logging.info(f"Successfully processed failed pages for: {pdf_filename}")
            return True
        else:
            logging.error(f"Failed to process failed pages for: {pdf_filename}")
            return False
            
    except Exception as e:
        logging.error(f"Error processing {pdf_filename}: {e}")
        return False

def main():
    """Main function to process all PDFs with error files."""
    setup_logging()
    
    logging.info("Starting automatic rerun of failed pages for all PDFs with error files")
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
    logging.info("")
    
    # Process each PDF
    successful = 0
    failed = 0
    
    for i, (pdf_filename, error_file_path) in enumerate(pdfs_with_errors, 1):
        logging.info(f"Processing {i}/{len(pdfs_with_errors)}: {pdf_filename}")
        logging.info(f"Error file: {error_file_path}")
        logging.info("-" * 60)
        
        if run_failed_pages_processing(pdf_filename):
            successful += 1
        else:
            failed += 1
        
        logging.info("")
    
    # Summary
    logging.info("=" * 80)
    logging.info("PROCESSING SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Total PDFs with error files: {len(pdfs_with_errors)}")
    logging.info(f"Successfully processed: {successful}")
    logging.info(f"Failed to process: {failed}")
    
    if failed > 0:
        logging.warning(f"{failed} PDFs failed to process. Check the logs above for details.")
        sys.exit(1)
    else:
        logging.info("All PDFs processed successfully!")

if __name__ == "__main__":
    main() 