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
    - Same fuzzy matching algorithm (greedy mutual-best, threshold=0.9 for entries)
    - Same CER calculation (Levenshtein normalized distance on raw concatenated text)
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
import argparse
import pandas as pd
from pathlib import Path
from rapidfuzz.distance import Levenshtein

# Add the scripts directory to path so we can import core modules
sys.path.insert(0, str(Path(__file__).parent))

# Import EXISTING benchmarking functions to ensure identical methodology
from core.benchmarking import (
    create_clean_text_for_cer,
    load_gt_file,
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

# Label used in the "Model" row / HTML title, and suffix appended to output
# filenames. Defaults reproduce the original 3.1-Pro run; overridable via CLI so a
# second model (gemini-2.5-pro) can be evaluated without overwriting outputs.
MODEL_LABEL = "Gemini-3.1-Pro-Preview"
OUT_SUFFIX = ""

# Fuzzy matching thresholds — aligned with existing pipeline benchmarks
ENTRY_FUZZY_THRESHOLD = 0.9      # For entry-level matching (same as 01/02 scripts)
VARIABLE_FUZZY_THRESHOLD = 0.85  # For variable-level comparison (same as 03 script)

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
    """Compare two variable values using fuzzy matching."""
    if pd.isna(gt_value) or pd.isna(llm_value):
        return False

    try:
        gt_str = str(gt_value).strip()
        llm_str = str(llm_value).strip()
    except (TypeError, ValueError):
        return False

    if gt_str == "" or llm_str == "":
        return False

    # Special handling for patent_id: remove .0 from LLM value if present
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
            perfect_df[['id', 'entry']], llm_df[['id', 'entry']], ENTRY_FUZZY_THRESHOLD
        )
        entry_match_rate = sum(gt_matches) / len(gt_matches) * 100 if gt_matches else 0

        # ----- CER CALCULATION (identical to core/benchmarking.py) -----
        # Concatenate all entry text (raw, no normalization) — same as main branch
        perfect_clean = create_clean_text_for_cer(perfect_df[['id', 'entry']])
        llm_clean = create_clean_text_for_cer(llm_df[['id', 'entry']])
        student_clean = create_clean_text_for_cer(student_df)  # already [id, entry] only

        # CER = normalized Levenshtein distance (same as existing scripts)
        llm_cer = Levenshtein.normalized_distance(perfect_clean, llm_clean)
        student_cer = Levenshtein.normalized_distance(perfect_clean, student_clean)
        performance_gap = student_cer - llm_cer  # Positive = LLM better

        # ----- VARIABLE MATCHING (identical to 03_variable_extraction_benchmarking.py) -----
        # Match entries first, then compare variables for matched pairs
        gt_matches_var, llm_matches_var, gt_match_ids, _ = match_entries_fuzzy(
            perfect_df[['id', 'entry']], llm_df[['id', 'entry']], ENTRY_FUZZY_THRESHOLD
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
            gt_row = perfect_df.iloc[i]

            for field in VARIABLE_FIELDS:
                gt_val = str(gt_row.get(field, "NaN"))
                llm_val = str(llm_row.get(field, "NaN"))

                variable_stats[field]['total'] += 1
                total_variable_cells += 1

                if compare_variables(gt_val, llm_val, VARIABLE_FUZZY_THRESHOLD):
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
    # (same approach as create_performance_gap_analysis in core/benchmarking.py)
    all_perfect_text = ""
    all_llm_text = ""
    all_student_text = ""

    for stem, llm_path, perfect_path, student_path in common_files:
        perfect_df = load_perfect_with_variables(perfect_path)
        llm_df = load_llm_csv_with_variables(llm_path)
        student_df = load_gt_file(student_path)

        if not perfect_df.empty and not llm_df.empty and not student_df.empty:
            perfect_text = create_clean_text_for_cer(perfect_df[['id', 'entry']])
            llm_text = create_clean_text_for_cer(llm_df[['id', 'entry']])
            student_text = create_clean_text_for_cer(student_df)

            if perfect_text and llm_text and student_text:
                all_perfect_text += perfect_text + " "
                all_llm_text += llm_text + " "
                all_student_text += student_text + " "

    # Compute aggregate CER on concatenated text (same as main branch)
    agg_perfect_clean = all_perfect_text
    agg_llm_clean = all_llm_text
    agg_student_clean = all_student_text

    agg_llm_cer = Levenshtein.normalized_distance(agg_perfect_clean, agg_llm_clean)
    agg_student_cer = Levenshtein.normalized_distance(agg_perfect_clean, agg_student_clean)
    agg_performance_gap = agg_student_cer - agg_llm_cer

    # Aggregate entry match rate (average across files)
    avg_entry_match_rate = sum(r['entry_match_rate'] for r in per_file_results) / len(per_file_results)

    # Aggregate entry counts (for the "X / N (NN.NN%)" cell in the LaTeX table)
    total_matched_entries = sum(r['matched_entries'] for r in per_file_results)
    total_gt_entries = sum(r['perfect_entries'] for r in per_file_results)
    aggregate_entry_match_rate = (total_matched_entries / total_gt_entries * 100) if total_gt_entries else 0

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
        {"Metric": "Model", "Value": MODEL_LABEL},
        {"Metric": "Approach", "Value": "Single-step (1 call/page)"},
        {"Metric": "Files Evaluated", "Value": f"{len(per_file_results)}"},
        {"Metric": "Entry Match Rate (avg)", "Value": f"{avg_entry_match_rate:.2f}%"},
        {"Metric": "Matched Entries (total)", "Value": f"{total_matched_entries}"},
        {"Metric": "GT Entries (total)", "Value": f"{total_gt_entries}"},
        {"Metric": "Entry Match Rate (aggregate)", "Value": f"{aggregate_entry_match_rate:.2f}%"},
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
    csv_output_path = ZERO_SHOT_DIR / f"revisions_for_VSWG_consolidated_table{OUT_SUFFIX}.csv"
    table_df.to_csv(csv_output_path, index=False)
    logging.info(f"Consolidated table saved to: {csv_output_path}")

    # =========================================================================
    # OUTPUT: HTML REPORT
    # =========================================================================
    html_output_path = ZERO_SHOT_DIR / f"revisions_for_VSWG_report{OUT_SUFFIX}.html"
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
    <p style="text-align:center; color:#666;">VSWG Paper Revisions — {html_escape(MODEL_LABEL)}</p>

    <div class="methodology">
        <strong>Methodology:</strong> This evaluation uses identical comparison logic to the
        existing multi-stage pipeline benchmarks: same fuzzy entry matching threshold (0.9),
        same CER calculation (Levenshtein normalized distance on raw concatenated text),
        same variable comparison (0.85 threshold). Results are directly comparable.
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

def parse_args():
    parser = argparse.ArgumentParser(description="VSWG single-step benchmarking/evaluation")
    parser.add_argument(
        "--llm-dir", default=str(LLM_CSV_DIR),
        help="Directory of single-step LLM CSVs to evaluate (default: 3.1-Pro llm_csv)"
    )
    parser.add_argument(
        "--model-label", default=MODEL_LABEL,
        help=f"Model label for the report (default: {MODEL_LABEL})"
    )
    parser.add_argument(
        "--out-suffix", default=OUT_SUFFIX,
        help="Suffix appended to output filenames (default: none — overwrites 3.1-Pro outputs)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    LLM_CSV_DIR = Path(args.llm_dir)
    MODEL_LABEL = args.model_label
    OUT_SUFFIX = args.out_suffix
    evaluate_all()
