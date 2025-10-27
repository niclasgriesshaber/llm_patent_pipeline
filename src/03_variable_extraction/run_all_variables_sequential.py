#!/usr/bin/env python3
"""
Sequential Variable Extraction Script

This script runs variable_extraction.py sequentially for all CSV files in the cleaned_csvs folder
that haven't been processed yet. It checks the cleaned_with_variables_csvs folder to avoid reprocessing.

Usage:
    python run_all_variables_sequential.py
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Tuple

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "02_dataset_cleaning" / "cleaned_csvs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "03_variable_extraction" / "cleaned_with_variables_csvs"
VARIABLE_EXTRACTION_SCRIPT = SCRIPT_DIR / "variable_extraction.py"

# CLI arguments for variable_extraction.py
MODEL = "gemini-2.5-flash-lite"
MAX_WORKERS = 20  # Start with 50 workers for variable extraction

def get_input_files() -> List[Path]:
    """Get all CSV files from the input directory."""
    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        sys.exit(1)
    
    # Only get files matching the pattern "Patentamt_YEAR_cleaned.csv"
    csv_files = list(INPUT_DIR.glob("Patentamt_*_cleaned.csv"))
    csv_files.sort()  # Sort for consistent processing order
    
    print(f"[INFO] Found {len(csv_files)} valid CSV files in {INPUT_DIR}")
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
    # Extract year from input filename (e.g., "Patentamt_1880_cleaned.csv" -> "1880")
    year = input_file.stem.split('_')[-2]  # Always extract from "_cleaned" pattern
    
    # Check for cleaned_with_variables CSV file
    expected_output = f"Patentamt_{year}_cleaned_with_variables.csv"
    expected_output_path = OUTPUT_DIR / expected_output
    
    # Also check for XLSX file in cleaned_with_variables_xlsx directory
    xlsx_output_dir = PROJECT_ROOT / "data" / "03_variable_extraction" / "cleaned_with_variables_xlsx"
    xlsx_output_path = xlsx_output_dir / f"Patentamt_{year}_cleaned_with_variables.xlsx"
    
    return expected_output_path.exists() or xlsx_output_path.exists()

def run_variable_extraction(csv_file: Path, prompt_file: str) -> Tuple[bool, str]:
    """Run variable_extraction.py for a single CSV file."""
    csv_name = csv_file.name
    
    cmd = [
        sys.executable,
        str(VARIABLE_EXTRACTION_SCRIPT),
        "--csv", csv_name,
        "--model", MODEL,
        "--max_workers", str(MAX_WORKERS),
        "--prompt", prompt_file
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
    parser = argparse.ArgumentParser(description="Sequential Variable Extraction Script")
    parser.add_argument("--prompt", type=str, default="prompt.txt", help="Prompt filename (default=prompt.txt)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 SEQUENTIAL VARIABLE EXTRACTION SCRIPT")
    print("=" * 80)
    print(f"📁 Input directory: {INPUT_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🤖 Model: {MODEL}")
    print(f"👥 Max workers: {MAX_WORKERS}")
    print(f"📝 Prompt file: {args.prompt}")
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
        success, message = run_variable_extraction(csv_file, args.prompt)
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
