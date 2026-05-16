#!/usr/bin/env python3
"""
=============================================================================
REVISIONS FOR VSWG — Zero-Shot Benchmarking & Evaluation Script
=============================================================================

Purpose:
    Evaluates the zero-shot inference results against ground truth, producing
    a single consolidated table suitable for inclusion in the VSWG paper.

    This script uses IDENTICAL methodology to the existing pipeline benchmarks
    to ensure a fair comparison:
    - Same fuzzy matching algorithm (greedy mutual-best, threshold=0.85)
    - Same CER calculation (Levenshtein normalized distance on cleaned text)
    - Same performance gap formula (CER_student - CER_llm)
    - Same variable comparison (compare_variables with 0.85 threshold)

Metrics computed:
    1. Entry Match Rate: fuzzy match of extracted entries vs perfect transcriptions
    2. CER (LLM vs Perfect): character error rate of zero-shot extraction
    3. CER (Student vs Perfect): student baseline for performance gap
    4. Performance Gap: student_cer - llm_cer (positive = LLM better)
    5. Variable-level match rates: per-field accuracy (patent_id, name, location,
       description, date) using same fuzzy threshold

Outputs:
    - revisions_for_VSWG_consolidated_table.csv  (for the paper)
    - revisions_for_VSWG_report.html             (visual reference)

Usage:
    python revisions_for_VSWG_zero_shot_benchmarking.py
=============================================================================
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from rapidfuzz.distance import Levenshtein

# Add the scripts directory to path so we can import core modules
sys.path.insert(0, str(Path(__file__).parent))

# Import EXISTING benchmarking functions to ensure identical methodology
from core.benchmarking import (
    normalize_text_for_cer,
    create_clean_text_for_cer,
    load_gt_file,
    load_llm_file,
    match_entries_fuzzy,
    extract_year_from_filename,
    html_escape
)

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKING_ROOT = PROJECT_ROOT / "data" / "benchmarking"

# Input directories (shared with existing benchmarks — read-only)
PERFECT_XLSX_DIR = BENCHMARKING_ROOT / "input_data" / "transcriptions_xlsx" / "perfect_transcriptions_xlsx"
STUDENT_XLSX_DIR = BENCHMARKING_ROOT / "input_data" / "transcriptions_xlsx" / "student_transcriptions_xlsx"

# Zero-shot results directory
ZERO_SHOT_DIR = BENCHMARKING_ROOT / "results" / "revisions_for_VSWG_zero_shot"
LLM_CSV_DIR = ZERO_SHOT_DIR / "llm_csv"

# Fuzzy matching threshold — same as used in core/benchmarking.py defaults
FUZZY_THRESHOLD = 0.85

# Variable fields (same as in 03_variable_extraction_benchmarking.py)
VARIABLE_FIELDS = ['patent_id', 'name', 'location', 'description', 'date']


# =============================================================================
# DATA LOADING (mirrors existing pipeline exactly)
# =============================================================================

def load_llm_csv_with_variables(filepath: Path) -> pd.DataFrame:
    """
    Load a zero-shot LLM CSV file which contains both entry text AND variables.
    Applies same preprocessing as existing scripts: NFC normalization, whitespace trimming.
    """
    try:
        df = pd.read_csv(filepath, dtype=str)

        # Ensure required columns exist
        if 'id' not in df.columns or 'entry' not in df.columns:
            logging.warning(f"File {filepath.name} missing 'id' or 'entry' columns")
            return pd.DataFrame()

        # Drop rows with empty entries
        df = df[df['entry'].notna() & (df['entry'].str.strip() != '')].copy()

        # NFC normalize entry text (same as load_gt_file and load_llm_file)
        df['entry'] = df['entry'].str.normalize('NFC')

        # Trim whitespace from variable columns
        for field in VARIABLE_FIELDS:
            if field in df.columns:
                # Special handling for patent_id: remove .0 suffix
                if field == 'patent_id':
                    df[field] = df[field].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                else:
                    df[field] = df[field].astype(str).str.strip()
            else:
                df[field] = "NaN"

        return df.reset_index(drop=True)
    except Exception as e:
        logging.error(f"Error loading LLM CSV {filepath}: {e}")
        return pd.DataFrame()


def load_perfect_with_variables(filepath: Path) -> pd.DataFrame:
    """
    Load a perfect transcription XLSX file with both entries and variables.
    Same logic as load_gt_variables() in 03_variable_extraction_benchmarking.py.
    """
    try:
        df = pd.read_excel(filepath, dtype=str)

        # Trim whitespace from column names (ground truth has trailing space on 'description ')
        df.columns = df.columns.str.strip()

        if 'id' not in df.columns or 'entry' not in df.columns:
            logging.warning(f"File {filepath.name} missing 'id' or 'entry' columns")
            return pd.DataFrame()

        # Drop rows with empty entries
        df = df[df['entry'].notna() & (df['entry'].str.strip() != '')].copy()

        # NFC normalize entry text
        df['entry'] = df['entry'].str.normalize('NFC')

        # Process variable columns
        for field in VARIABLE_FIELDS:
            if field in df.columns:
                if field == 'patent_id':
                    df[field] = df[field].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                else:
                    df[field] = df[field].astype(str).str.strip()
            else:
                df[field] = "NaN"

        return df.reset_index(drop=True)
    except Exception as e:
        logging.error(f"Error loading perfect XLSX {filepath}: {e}")
        return pd.DataFrame()


# =============================================================================
# VARIABLE COMPARISON (same logic as 03_variable_extraction_benchmarking.py)
# =============================================================================

def compare_variables(gt_value: str, llm_value: str, threshold: float = 0.85) -> bool:
    """
    Compare two variable values using fuzzy matching.
    IDENTICAL to compare_variables() in 03_variable_extraction_benchmarking.py.
    """
    if pd.isna(gt_value) or pd.isna(llm_value):
        return False

    gt_str = str(gt_value).strip()
    llm_str = str(llm_value).strip()

    if gt_str == "" or llm_str == "" or gt_str == "NaN" or llm_str == "NaN":
        return False

    # Special handling for patent_id: remove .0 from LLM value
    if gt_str.isdigit() and llm_str.endswith('.0'):
        llm_str = llm_str[:-2]

    similarity = Levenshtein.normalized_similarity(gt_str, llm_str)
    return similarity >= threshold


# =============================================================================
# FILE MATCHING
# =============================================================================

def find_common_files() -> list:
    """
    Find file stems that exist in all three sources: LLM CSV, perfect XLSX, student XLSX.
    Handles naming conventions (e.g., _perfected suffix in perfect transcriptions).
    Returns list of tuples: (stem, llm_path, perfect_path, student_path)
    """
    # Get LLM CSV stems
    llm_files = {f.stem: f for f in LLM_CSV_DIR.glob("*.csv")}

    # Get perfect transcription stems (normalize _perfected suffix)
    perfect_files = {}
    for f in PERFECT_XLSX_DIR.glob("*.xlsx"):
        stem = f.stem
        if stem.endswith('_perfected'):
            base_stem = stem[:-10]  # Remove '_perfected'
        else:
            base_stem = stem
        perfect_files[base_stem] = f

    # Get student transcription stems
    student_files = {f.stem: f for f in STUDENT_XLSX_DIR.glob("*.xlsx")}

    # Find common stems across all three sources
    common_stems = sorted(
        set(llm_files.keys()) & set(perfect_files.keys()) & set(student_files.keys())
    )

    results = []
    for stem in common_stems:
        results.append((stem, llm_files[stem], perfect_files[stem], student_files[stem]))

    return results


# =============================================================================
# EVALUATION (uses identical functions to existing benchmarks)
# =============================================================================

def evaluate_all():
    """
    Main evaluation function. Computes all metrics and produces consolidated output.
    """
    common_files = find_common_files()
    if not common_files:
        logging.error("No common files found across LLM, perfect, and student directories.")
        logging.error(f"  LLM CSV dir: {LLM_CSV_DIR}")
        logging.error(f"  Perfect dir: {PERFECT_XLSX_DIR}")
        logging.error(f"  Student dir: {STUDENT_XLSX_DIR}")
        return

    logging.info(f"Found {len(common_files)} files for three-way comparison")

    # =========================================================================
    # PER-FILE METRICS
    # =========================================================================
    per_file_results = []

    # Accumulators for aggregate variable matching
    total_variable_cells = 0
    matched_variable_cells = 0
    variable_stats = {field: {'total': 0, 'matched': 0} for field in VARIABLE_FIELDS}

    for stem, llm_path, perfect_path, student_path in common_files:
        logging.info(f"Evaluating: {stem}")
        year = extract_year_from_filename(stem)

        # Load data from all three sources
        llm_df = load_llm_csv_with_variables(llm_path)
        perfect_df = load_perfect_with_variables(perfect_path)
        student_df = load_gt_file(student_path)  # Students only have id+entry

        if llm_df.empty or perfect_df.empty or student_df.empty:
            logging.warning(f"  Skipping {stem}: one or more sources is empty")
            continue

        # ----- ENTRY MATCHING (identical to core/benchmarking.py) -----
        # Fuzzy match LLM entries against perfect transcriptions
        gt_matches, llm_matches, _, _ = match_entries_fuzzy(
            perfect_df[['id', 'entry']], llm_df[['id', 'entry']], FUZZY_THRESHOLD
        )
        entry_match_rate = sum(gt_matches) / len(gt_matches) * 100 if gt_matches else 0

        # ----- CER CALCULATION (identical to core/benchmarking.py) -----
        # Concatenate all entries and normalize for CER
        perfect_clean = normalize_text_for_cer(
            ' '.join(perfect_df['entry'].astype(str).tolist())
        )
        llm_clean = normalize_text_for_cer(
            ' '.join(llm_df['entry'].astype(str).tolist())
        )
        student_clean = normalize_text_for_cer(
            ' '.join(student_df['entry'].astype(str).tolist())
        )

        # CER = normalized Levenshtein distance (same as existing scripts)
        llm_cer = Levenshtein.normalized_distance(perfect_clean, llm_clean)
        student_cer = Levenshtein.normalized_distance(perfect_clean, student_clean)
        performance_gap = student_cer - llm_cer  # Positive = LLM better

        # ----- VARIABLE MATCHING (identical to 03_variable_extraction_benchmarking.py) -----
        # Match entries first, then compare variables for matched pairs
        perfect_var_df = load_perfect_with_variables(perfect_path)
        gt_matches_var, llm_matches_var, gt_match_ids, _ = match_entries_fuzzy(
            perfect_var_df[['id', 'entry']], llm_df[['id', 'entry']], FUZZY_THRESHOLD
        )

        # For each matched pair, compare variables field by field
        for i, is_matched in enumerate(gt_matches_var):
            if not is_matched:
                continue
            # Find the LLM index that was matched to this GT index
            matched_llm_id = gt_match_ids[i]
            if matched_llm_id == '—':
                continue

            # Find the LLM row by its id
            llm_row_candidates = llm_df[llm_df['id'].astype(str) == str(matched_llm_id)]
            if llm_row_candidates.empty:
                continue
            llm_row = llm_row_candidates.iloc[0]
            gt_row = perfect_var_df.iloc[i]

            for field in VARIABLE_FIELDS:
                gt_val = str(gt_row.get(field, "NaN"))
                llm_val = str(llm_row.get(field, "NaN"))

                variable_stats[field]['total'] += 1
                total_variable_cells += 1

                if compare_variables(gt_val, llm_val, FUZZY_THRESHOLD):
                    variable_stats[field]['matched'] += 1
                    matched_variable_cells += 1

        per_file_results.append({
            'file': stem,
            'year': year,
            'entry_match_rate': entry_match_rate,
            'llm_cer': llm_cer,
            'student_cer': student_cer,
            'performance_gap': performance_gap,
            'perfect_entries': len(perfect_df),
            'llm_entries': len(llm_df),
            'student_entries': len(student_df),
            'matched_entries': sum(gt_matches),
        })

    if not per_file_results:
        logging.error("No files could be evaluated.")
        return

    # =========================================================================
    # AGGREGATE METRICS
    # =========================================================================

    # Aggregate CER: concatenate ALL entries across all files then compute CER
    # (this is the approach used in overall_cer_calculation.py)
    all_perfect_entries = []
    all_llm_entries = []
    all_student_entries = []

    for stem, llm_path, perfect_path, student_path in common_files:
        perfect_df = load_perfect_with_variables(perfect_path)
        llm_df = load_llm_csv_with_variables(llm_path)
        student_df = load_gt_file(student_path)

        if not perfect_df.empty:
            all_perfect_entries.extend(perfect_df['entry'].astype(str).tolist())
        if not llm_df.empty:
            all_llm_entries.extend(llm_df['entry'].astype(str).tolist())
        if not student_df.empty:
            all_student_entries.extend(student_df['entry'].astype(str).tolist())

    # Compute aggregate CER on concatenated text
    agg_perfect_clean = normalize_text_for_cer(' '.join(all_perfect_entries))
    agg_llm_clean = normalize_text_for_cer(' '.join(all_llm_entries))
    agg_student_clean = normalize_text_for_cer(' '.join(all_student_entries))

    agg_llm_cer = Levenshtein.normalized_distance(agg_perfect_clean, agg_llm_clean)
    agg_student_cer = Levenshtein.normalized_distance(agg_perfect_clean, agg_student_clean)
    agg_performance_gap = agg_student_cer - agg_llm_cer

    # Aggregate entry match rate (average across files)
    avg_entry_match_rate = sum(r['entry_match_rate'] for r in per_file_results) / len(per_file_results)

    # Aggregate variable match rates
    overall_variable_rate = (matched_variable_cells / total_variable_cells * 100) if total_variable_cells > 0 else 0
    variable_rates = {}
    for field in VARIABLE_FIELDS:
        stats = variable_stats[field]
        rate = (stats['matched'] / stats['total'] * 100) if stats['total'] > 0 else 0
        variable_rates[field] = rate

    # =========================================================================
    # OUTPUT: CONSOLIDATED TABLE (CSV)
    # =========================================================================

    table_rows = [
        {"Metric": "Model", "Value": "Gemini-3.1-Pro-Preview"},
        {"Metric": "Approach", "Value": "Zero-shot (1 call/page)"},
        {"Metric": "Files Evaluated", "Value": f"{len(per_file_results)}"},
        {"Metric": "Entry Match Rate (avg)", "Value": f"{avg_entry_match_rate:.2f}%"},
        {"Metric": "CER (Zero-Shot vs Perfect)", "Value": f"{agg_llm_cer:.4f}"},
        {"Metric": "CER (Student vs Perfect)", "Value": f"{agg_student_cer:.4f}"},
        {"Metric": "Performance Gap", "Value": f"{agg_performance_gap:+.4f}"},
        {"Metric": "Variable: patent_id", "Value": f"{variable_rates.get('patent_id', 0):.2f}%"},
        {"Metric": "Variable: name", "Value": f"{variable_rates.get('name', 0):.2f}%"},
        {"Metric": "Variable: location", "Value": f"{variable_rates.get('location', 0):.2f}%"},
        {"Metric": "Variable: description", "Value": f"{variable_rates.get('description', 0):.2f}%"},
        {"Metric": "Variable: date", "Value": f"{variable_rates.get('date', 0):.2f}%"},
        {"Metric": "Overall Variable Match", "Value": f"{overall_variable_rate:.2f}%"},
    ]

    table_df = pd.DataFrame(table_rows)
    csv_output_path = ZERO_SHOT_DIR / "revisions_for_VSWG_consolidated_table.csv"
    table_df.to_csv(csv_output_path, index=False)
    logging.info(f"Consolidated table saved to: {csv_output_path}")

    # =========================================================================
    # OUTPUT: HTML REPORT
    # =========================================================================
    html_output_path = ZERO_SHOT_DIR / "revisions_for_VSWG_report.html"
    html_content = generate_html_report(
        table_rows, per_file_results, variable_rates, overall_variable_rate,
        agg_llm_cer, agg_student_cer, agg_performance_gap, avg_entry_match_rate
    )
    html_output_path.write_text(html_content, encoding='utf-8')
    logging.info(f"HTML report saved to: {html_output_path}")

    # Print summary to console
    logging.info("=" * 70)
    logging.info("CONSOLIDATED RESULTS (Zero-Shot Digitization)")
    logging.info("=" * 70)
    for row in table_rows:
        logging.info(f"  {row['Metric']:30s} {row['Value']}")

    return per_file_results


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_html_report(table_rows, per_file_results, variable_rates, overall_variable_rate,
                         agg_llm_cer, agg_student_cer, agg_performance_gap, avg_entry_match_rate):
    """Generate a clean HTML report with the consolidated table and per-file details."""

    # Consolidated table HTML
    table_html = '<table class="consolidated-table">\n'
    table_html += '<tr><th>Metric</th><th>Value</th></tr>\n'
    for row in table_rows:
        table_html += f'<tr><td>{html_escape(row["Metric"])}</td><td>{html_escape(row["Value"])}</td></tr>\n'
    table_html += '</table>'

    # Per-file results table
    per_file_html = '<table class="per-file-table">\n'
    per_file_html += '<tr><th>Year</th><th>File</th><th>Entry Match Rate</th><th>CER (LLM)</th><th>CER (Student)</th><th>Gap</th><th>GT Entries</th><th>LLM Entries</th></tr>\n'
    for r in sorted(per_file_results, key=lambda x: x['year']):
        gap_color = '#d4edda' if r['performance_gap'] > 0 else '#f8d7da'
        per_file_html += (
            f'<tr>'
            f'<td>{html_escape(r["year"])}</td>'
            f'<td>{html_escape(r["file"])}</td>'
            f'<td>{r["entry_match_rate"]:.2f}%</td>'
            f'<td>{r["llm_cer"]:.4f}</td>'
            f'<td>{r["student_cer"]:.4f}</td>'
            f'<td style="background-color:{gap_color}">{r["performance_gap"]:+.4f}</td>'
            f'<td>{r["perfect_entries"]}</td>'
            f'<td>{r["llm_entries"]}</td>'
            f'</tr>\n'
        )
    per_file_html += '</table>'

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>VSWG Revisions — Zero-Shot Digitization Benchmark</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #f8f9fa; color: #333; }}
        .container {{ max-width: 1000px; margin: auto; padding: 30px; }}
        h1 {{ color: #2c3e50; text-align: center; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .consolidated-table, .per-file-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .consolidated-table th, .consolidated-table td,
        .per-file-table th, .per-file-table td {{ border: 1px solid #ddd; padding: 10px 15px; text-align: left; }}
        .consolidated-table th, .per-file-table th {{ background: #2c3e50; color: #fff; }}
        .consolidated-table tr:nth-child(even), .per-file-table tr:nth-child(even) {{ background: #f8f9fa; }}
        .note {{ background: #e8f4fd; border-left: 4px solid #2196f3; padding: 15px; margin: 20px 0; }}
        .methodology {{ background: #f0f7e6; border-left: 4px solid #4caf50; padding: 15px; margin: 20px 0; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Zero-Shot Digitization Benchmark</h1>
    <p style="text-align:center; color:#666;">VSWG Paper Revisions — Gemini-3.1-Pro-Preview</p>

    <div class="methodology">
        <strong>Methodology:</strong> This evaluation uses identical comparison logic to the
        existing multi-stage pipeline benchmarks: same fuzzy matching threshold (0.85),
        same CER normalization (lowercase, ASCII a-z + 0-9), same Levenshtein distance
        metric, same variable comparison function. Results are directly comparable.
    </div>

    <h2>Consolidated Results</h2>
    {table_html}

    <div class="note">
        <strong>Performance Gap:</strong> Computed as CER(Student) - CER(Zero-Shot).
        Positive values indicate the zero-shot approach outperforms research assistants.
    </div>

    <h2>Per-File Results</h2>
    {per_file_html}
</div>
</body>
</html>'''


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    evaluate_all()
