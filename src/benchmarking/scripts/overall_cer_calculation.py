#!/usr/bin/env python3
"""
Overall CER Calculation Script

This script concatenates all files from three sources (LLM CSV, perfect transcriptions, student transcriptions)
into separate text files and calculates Character Error Rate (CER) between them.

Requirements:
- Only processes files that exist in all three sources
- Concatenates only the 'entry' column values
- Separates entries with newlines
- Uses raw text (no normalization) for CER calculation
- Saves results to results/overall/ folder
"""

import os
import pandas as pd
from pathlib import Path
import logging
from rapidfuzz.distance import Levenshtein
from collections import defaultdict

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hardcoded paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Go up to project root
BENCHMARKING_ROOT = PROJECT_ROOT / 'data' / 'benchmarking'

# Input directories (hardcoded)
LLM_CSV_DIR = BENCHMARKING_ROOT / 'results' / '02_dataset_cleaning' / 'gemini-2.5-flash-lite' / 'cleaning_v0.1_prompt' / 'llm_csv'
PERFECT_XLSX_DIR = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'perfect_transcriptions_xlsx'
STUDENT_XLSX_DIR = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'student_transcriptions_xlsx'

# Output directory
OUTPUT_DIR = BENCHMARKING_ROOT / 'results' / 'overall'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_file_stems(directory: Path, extension: str) -> set:
    """Get file stems from a directory with given extension, normalizing suffixes."""
    stems = set()
    for f in directory.glob(f'*.{extension}'):
        stem = f.stem
        # Normalize stems by removing common suffixes
        if stem.endswith('_cleaned'):
            stem = stem[:-8]  # Remove '_cleaned'
        elif stem.endswith('_perfected'):
            stem = stem[:-10]  # Remove '_perfected'
        stems.add(stem)
    return stems

def load_entries_from_csv(csv_path: Path) -> list:
    """Load entry column from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if 'entry' not in df.columns:
            logging.warning(f"No 'entry' column found in {csv_path.name}")
            return []
        # Filter out empty entries
        entries = [str(entry).strip() for entry in df['entry'].tolist() if pd.notna(entry) and str(entry).strip()]
        return entries
    except Exception as e:
        logging.error(f"Error loading CSV {csv_path.name}: {e}")
        return []

def load_entries_from_xlsx(xlsx_path: Path) -> list:
    """Load entry column from Excel file."""
    try:
        df = pd.read_excel(xlsx_path)
        if 'entry' not in df.columns:
            logging.warning(f"No 'entry' column found in {xlsx_path.name}")
            return []
        # Filter out empty entries
        entries = [str(entry).strip() for entry in df['entry'].tolist() if pd.notna(entry) and str(entry).strip()]
        return entries
    except Exception as e:
        logging.error(f"Error loading Excel {xlsx_path.name}: {e}")
        return []

def concatenate_entries(entries: list) -> str:
    """Concatenate entries with newline separation."""
    return '\n'.join(entries)

def calculate_cer(text1: str, text2: str) -> float:
    """Calculate Character Error Rate using Levenshtein distance."""
    if not text1 or not text2:
        return 1.0  # 100% error if one text is empty
    
    # Use raw text without normalization
    distance = Levenshtein.distance(text1, text2)
    max_length = max(len(text1), len(text2))
    
    if max_length == 0:
        return 0.0
    
    return distance / max_length

def main():
    """Main function to execute the CER calculation process."""
    logging.info("Starting overall CER calculation process")
    
    # Check if input directories exist
    if not LLM_CSV_DIR.exists():
        logging.error(f"LLM CSV directory not found: {LLM_CSV_DIR}")
        return
    
    if not PERFECT_XLSX_DIR.exists():
        logging.error(f"Perfect transcriptions directory not found: {PERFECT_XLSX_DIR}")
        return
    
    if not STUDENT_XLSX_DIR.exists():
        logging.error(f"Student transcriptions directory not found: {STUDENT_XLSX_DIR}")
        return
    
    # Get file stems from all three sources
    llm_stems = get_file_stems(LLM_CSV_DIR, 'csv')
    perfect_stems = get_file_stems(PERFECT_XLSX_DIR, 'xlsx')
    student_stems = get_file_stems(STUDENT_XLSX_DIR, 'xlsx')
    
    logging.info(f"Found {len(llm_stems)} LLM CSV files")
    logging.info(f"Found {len(perfect_stems)} perfect transcription files")
    logging.info(f"Found {len(student_stems)} student transcription files")
    
    # Find common files (files that exist in all three sources)
    common_stems = llm_stems.intersection(perfect_stems).intersection(student_stems)
    common_stems = sorted(list(common_stems))
    
    if not common_stems:
        logging.error("No common files found across all three sources")
        return
    
    logging.info(f"Found {len(common_stems)} common files: {common_stems}")
    
    # Initialize concatenated text storage
    all_llm_entries = []
    all_perfect_entries = []
    all_student_entries = []
    
    processed_files = []
    skipped_files = []
    
    # Process each common file
    for stem in common_stems:
        logging.info(f"Processing {stem}")
        
        # Load entries from each source with correct file naming
        llm_entries = load_entries_from_csv(LLM_CSV_DIR / f"{stem}_cleaned.csv")
        perfect_entries = load_entries_from_xlsx(PERFECT_XLSX_DIR / f"{stem}_perfected.xlsx")
        student_entries = load_entries_from_xlsx(STUDENT_XLSX_DIR / f"{stem}.xlsx")
        
        # Check if all sources have entries
        if llm_entries and perfect_entries and student_entries:
            all_llm_entries.extend(llm_entries)
            all_perfect_entries.extend(perfect_entries)
            all_student_entries.extend(student_entries)
            processed_files.append(stem)
            logging.info(f"  Added {len(llm_entries)} LLM entries, {len(perfect_entries)} perfect entries, {len(student_entries)} student entries")
        else:
            skipped_files.append(stem)
            logging.warning(f"  Skipped {stem} - missing entries in one or more sources")
    
    if not processed_files:
        logging.error("No files were successfully processed")
        return
    
    # Concatenate all entries
    logging.info("Concatenating entries...")
    llm_text = concatenate_entries(all_llm_entries)
    perfect_text = concatenate_entries(all_perfect_entries)
    student_text = concatenate_entries(all_student_entries)
    
    logging.info(f"Concatenated text lengths:")
    logging.info(f"  LLM: {len(llm_text)} characters")
    logging.info(f"  Perfect: {len(perfect_text)} characters")
    logging.info(f"  Student: {len(student_text)} characters")
    
    # Calculate CER
    logging.info("Calculating CER...")
    cer_llm_vs_perfect = calculate_cer(perfect_text, llm_text)
    cer_student_vs_perfect = calculate_cer(perfect_text, student_text)
    
    # Calculate performance gap
    performance_gap = cer_student_vs_perfect - cer_llm_vs_perfect
    
    # Save concatenated text files
    logging.info("Saving concatenated text files...")
    
    llm_text_path = OUTPUT_DIR / "llm_concatenated.txt"
    perfect_text_path = OUTPUT_DIR / "perfect_concatenated.txt"
    student_text_path = OUTPUT_DIR / "student_concatenated.txt"
    
    llm_text_path.write_text(llm_text, encoding='utf-8')
    perfect_text_path.write_text(perfect_text, encoding='utf-8')
    student_text_path.write_text(student_text, encoding='utf-8')
    
    logging.info(f"Saved concatenated text files to {OUTPUT_DIR}")
    
    # Save CER results and summary
    results_path = OUTPUT_DIR / "cer_results.txt"
    
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("Overall CER Calculation Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Input Directories:\n")
        f.write(f"  LLM CSV: {LLM_CSV_DIR}\n")
        f.write(f"  Perfect: {PERFECT_XLSX_DIR}\n")
        f.write(f"  Student: {STUDENT_XLSX_DIR}\n\n")
        
        f.write("File Processing Summary:\n")
        f.write(f"  Total LLM CSV files found: {len(llm_stems)}\n")
        f.write(f"  Total perfect files found: {len(perfect_stems)}\n")
        f.write(f"  Total student files found: {len(student_stems)}\n")
        f.write(f"  Common files processed: {len(processed_files)}\n")
        f.write(f"  Files skipped: {len(skipped_files)}\n\n")
        
        if skipped_files:
            f.write("Skipped files:\n")
            for file in skipped_files:
                f.write(f"  - {file}\n")
            f.write("\n")
        
        f.write("Processed files:\n")
        for file in processed_files:
            f.write(f"  - {file}\n")
        f.write("\n")
        
        f.write("Concatenated Text Statistics:\n")
        f.write(f"  LLM text length: {len(llm_text):,} characters\n")
        f.write(f"  Perfect text length: {len(perfect_text):,} characters\n")
        f.write(f"  Student text length: {len(student_text):,} characters\n")
        f.write(f"  Total LLM entries: {len(all_llm_entries):,}\n")
        f.write(f"  Total perfect entries: {len(all_perfect_entries):,}\n")
        f.write(f"  Total student entries: {len(all_student_entries):,}\n\n")
        
        f.write("Character Error Rate (CER) Results:\n")
        f.write(f"  CER (Perfect vs LLM): {cer_llm_vs_perfect:.4f} ({cer_llm_vs_perfect*100:.2f}%)\n")
        f.write(f"  CER (Perfect vs Student): {cer_student_vs_perfect:.4f} ({cer_student_vs_perfect*100:.2f}%)\n")
        f.write(f"  Performance Gap: {performance_gap:+.4f} ({performance_gap*100:+.2f}%)\n\n")
        
        f.write("Interpretation:\n")
        f.write("  - Lower CER indicates better performance\n")
        f.write("  - Positive performance gap means LLM performs better than Student\n")
        f.write("  - Negative performance gap means Student performs better than LLM\n")
        f.write("  - CER calculated using raw text (no normalization)\n")
    
    logging.info(f"Saved CER results to {results_path}")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("OVERALL CER CALCULATION COMPLETE")
    print("=" * 60)
    print(f"Files processed: {len(processed_files)}")
    print(f"Files skipped: {len(skipped_files)}")
    print(f"\nCER Results:")
    print(f"  Perfect vs LLM: {cer_llm_vs_perfect:.4f} ({cer_llm_vs_perfect*100:.2f}%)")
    print(f"  Perfect vs Student: {cer_student_vs_perfect:.4f} ({cer_student_vs_perfect*100:.2f}%)")
    print(f"  Performance Gap: {performance_gap:+.4f} ({performance_gap*100:+.2f}%)")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
