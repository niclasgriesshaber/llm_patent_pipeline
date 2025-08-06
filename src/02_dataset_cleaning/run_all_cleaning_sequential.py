#!/usr/bin/env python3
"""
Sequential Patent Cleaning Script

This script runs complete_patent.py sequentially for all CSV files in the complete_csvs folder
that haven't been processed yet. It checks the cleaned_csvs folder to avoid reprocessing.

Usage:
    python run_all_cleaning_sequential.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "01_dataset_construction" / "complete_csvs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "02_dataset_cleaning" / "cleaned_csvs"
COMPLETE_PATENT_SCRIPT = SCRIPT_DIR / "complete_patent.py"

# CLI arguments for complete_patent.py
MODEL = "gemini-2.5-flash-lite"
MAX_WORKERS = 50  # Start with 80 workers for testing

def get_input_files() -> List[Path]:
    """Get all CSV files from the input directory."""
    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        sys.exit(1)
    
    csv_files = list(INPUT_DIR.glob("*.csv"))
    csv_files.sort()  # Sort for consistent processing order
    
    print(f"[INFO] Found {len(csv_files)} CSV files in {INPUT_DIR}")
    return csv_files

def get_processed_files() -> List[Path]:
    """Get all already processed CSV files from the output directory."""
    if not OUTPUT_DIR.exists():
        print(f"[INFO] Output directory does not exist, will be created: {OUTPUT_DIR}")
        return []
    
    processed_files = list(OUTPUT_DIR.glob("*.csv"))
    print(f"[INFO] Found {len(processed_files)} already processed files in {OUTPUT_DIR}")
    return processed_files

def is_already_processed(input_file: Path, processed_files: List[Path]) -> bool:
    """Check if a file has already been processed."""
    # Extract year from input filename (e.g., "Patentamt_1880.csv" -> "1880")
    year = input_file.stem.split('_')[-1]
    
    # Check for cleaned CSV file
    expected_output = f"Patentamt_{year}_cleaned.csv"
    expected_output_path = OUTPUT_DIR / expected_output
    
    # Also check for check_merge xlsx file (alternative output location)
    check_merge_dir = PROJECT_ROOT / "data" / "02_dataset_cleaning" / "check_merge_xlsx"
    check_merge_xlsx = check_merge_dir / f"Patentamt_{year}_check_merge.xlsx"
    
    return expected_output_path.exists() or check_merge_xlsx.exists()

def run_complete_patent(csv_file: Path) -> Tuple[bool, str]:
    """Run complete_patent.py for a single CSV file."""
    csv_name = csv_file.name
    
    cmd = [
        sys.executable,
        str(COMPLETE_PATENT_SCRIPT),
        "--csv", csv_name,
        "--model", MODEL,
        "--max_workers", str(MAX_WORKERS)
    ]
    
    print(f"[INFO] Running: {' '.join(cmd)}")
    
    try:
        # Run the subprocess and stream output in real-time
        process = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Collect all output
        output_lines = []
        
        # Read output line by line and print in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())  # Print in real-time
                output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Combine all output
        full_output = ''.join(output_lines)
        
        if return_code == 0:
            return True, full_output
        else:
            return False, full_output
            
    except subprocess.TimeoutExpired:
        process.kill()
        return False, "Process timed out after 1 hour"
    except Exception as e:
        return False, str(e)

def main():
    """Main function to process all unprocessed CSV files."""
    print("=" * 80)
    print("🚀 SEQUENTIAL PATENT CLEANING SCRIPT")
    print("=" * 80)
    print(f"📁 Input directory: {INPUT_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🤖 Model: {MODEL}")
    print(f"👥 Max workers: {MAX_WORKERS}")
    print()
    
    # Get all input files
    input_files = get_input_files()
    if not input_files:
        print("[ERROR] No CSV files found to process.")
        sys.exit(1)
    
    # Get already processed files
    processed_files = get_processed_files()
    
    # Filter out already processed files
    unprocessed_files = []
    for input_file in input_files:
        if is_already_processed(input_file, processed_files):
            print(f"[SKIP] {input_file.name} already processed")
        else:
            unprocessed_files.append(input_file)
    
    if not unprocessed_files:
        print("[INFO] All files have already been processed!")
        sys.exit(0)
    
    print(f"\n[INFO] Found {len(unprocessed_files)} files to process:")
    for file in unprocessed_files:
        print(f"  - {file.name}")
    print()
    
    # Process files sequentially
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    for i, csv_file in enumerate(unprocessed_files, 1):
        print(f"\n{'='*60}")
        print(f"[PROCESSING {i}/{len(unprocessed_files)}] {csv_file.name}")
        print(f"{'='*60}")
        
        file_start_time = time.time()
        success, message = run_complete_patent(csv_file)
        file_time = time.time() - file_start_time
        
        if success:
            print(f"[✅ SUCCESS] {csv_file.name} processed in {file_time:.1f}s")
            success_count += 1
        else:
            print(f"[❌ FAILED] {csv_file.name} after {file_time:.1f}s")
            print(f"[ERROR] {message}")
            fail_count += 1
        
        # Add a small delay between files to avoid overwhelming the API
        if i < len(unprocessed_files):
            print("[INFO] Waiting 5 seconds before next file...")
            time.sleep(5)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("📊 PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total files processed: {success_count + fail_count}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    if fail_count > 0:
        print(f"\n⚠️  {fail_count} files failed. Check the logs above for details.")
        sys.exit(1)
    else:
        print(f"\n🎉 All files processed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main() 