#!/usr/bin/env python3
"""
Student-Constructed Benchmarking Script

This script compares student transcriptions to perfect transcriptions (ground truth)
to evaluate human transcription accuracy. It mirrors the LLM benchmarking pipeline
but uses student data instead of LLM-generated data.

Stages:
- Stage 1 (Dataset Construction): Compares student entries to perfect entries
- Stage 3 (Variable Extraction): Compares student-extracted variables to perfect variables

Output:
- patent_entry_matching.html
- variable_extraction_report.html
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Any
from rapidfuzz.distance import Levenshtein

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define project root
project_root = Path(__file__).parent.parent.parent.parent
BENCHMARKING_ROOT = project_root / 'data' / 'benchmarking'
STUDENT_XLSX_DIR = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'student_transcriptions_xlsx'
PERFECT_XLSX_DIR = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'perfect_transcriptions_xlsx'
OUTPUT_BASE_DIR = BENCHMARKING_ROOT / 'results' / 'student-constructed'

# Variable fields to extract and compare
VARIABLE_FIELDS = ['patent_id', 'name', 'location', 'description', 'date']

# Display names for HTML output (maps internal field names to user-friendly names)
VARIABLE_DISPLAY_NAMES = {
    'patent_id': 'patent_id',
    'name': 'assignee',
    'location': 'location',
    'description': 'description',
    'date': 'date'
}

def get_display_name(field: str) -> str:
    """Get the display name for a field (for HTML output only)."""
    return VARIABLE_DISPLAY_NAMES.get(field, field)


# --- Data Loading Functions ---

def load_xlsx_file(filepath: Path) -> pd.DataFrame:
    """Loads and preprocesses an Excel file."""
    try:
        df = pd.read_excel(filepath, dtype=str)
        
        # Trim whitespace from column names
        df.columns = df.columns.str.strip()
        
        if 'id' not in df.columns or 'entry' not in df.columns:
            raise ValueError(f"File {filepath} missing 'id' or 'entry' column.")
        
        # Keep all columns but ensure entry is not empty
        df = df.dropna(subset=['entry'])
        df = df[df['entry'].astype(str).str.strip() != ''].copy()
        df['entry'] = df['entry'].str.normalize('NFC')
        
        return df.reset_index(drop=True)
    except Exception as e:
        logging.error(f"Error loading file {filepath}: {e}")
        return pd.DataFrame()


def get_common_file_stems() -> List[str]:
    """Find files that exist in both student and perfect directories."""
    student_files = list(STUDENT_XLSX_DIR.glob('*.xlsx'))
    perfect_files = list(PERFECT_XLSX_DIR.glob('*.xlsx'))
    
    # Extract base stems
    student_stems = set()
    for f in student_files:
        stem = f.stem
        student_stems.add(stem)
    
    perfect_stems = set()
    for f in perfect_files:
        stem = f.stem
        if stem.endswith('_perfected'):
            base_stem = stem[:-10]  # Remove '_perfected'
            perfect_stems.add(base_stem)
        else:
            perfect_stems.add(stem)
    
    # Find common stems (student files use format: Patentamt_YYYY_sampled.xlsx)
    common = []
    for student_stem in student_stems:
        # Check if there's a matching perfect file
        if student_stem in perfect_stems:
            common.append(student_stem)
    
    return sorted(common)


# --- Fuzzy Matching Logic ---

def match_entries_fuzzy(gt_df: pd.DataFrame, student_df: pd.DataFrame, threshold: float = 0.85) -> Tuple[List[bool], List[bool], List[str], List[str]]:
    """Performs mutual best fuzzy matching between two dataframes."""
    gt_entries = gt_df['entry'].astype(str).tolist()
    student_entries = student_df['entry'].astype(str).tolist()
    gt_ids = gt_df['id'].astype(str).tolist()
    student_ids = student_df['id'].astype(str).tolist()

    gt_matches = [False] * len(gt_entries)
    student_matches = [False] * len(student_entries)
    gt_match_ids = ['—'] * len(gt_entries)
    student_match_ids = ['—'] * len(student_entries)

    # Calculate all similarity scores
    similarity_matrix = []
    for i, gt_entry in enumerate(gt_entries):
        row = []
        for j, student_entry in enumerate(student_entries):
            score = Levenshtein.normalized_similarity(gt_entry, student_entry)
            row.append(score)
        similarity_matrix.append(row)

    # Find mutual best matches
    used_gt_indices = set()
    used_student_indices = set()
    
    while True:
        best_match = None
        best_score = -1
        
        for i in range(len(gt_entries)):
            if i in used_gt_indices:
                continue
            for j in range(len(student_entries)):
                if j in used_student_indices:
                    continue
                score = similarity_matrix[i][j]
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = (i, j)
        
        if best_match is None:
            break
            
        gt_idx, student_idx = best_match
        gt_matches[gt_idx] = True
        student_matches[student_idx] = True
        gt_match_ids[gt_idx] = student_ids[student_idx]
        student_match_ids[student_idx] = gt_ids[gt_idx]
        used_gt_indices.add(gt_idx)
        used_student_indices.add(student_idx)
            
    return gt_matches, student_matches, gt_match_ids, student_match_ids


def compare_variables(gt_value: str, student_value: str, threshold: float = 0.85) -> bool:
    """Compare two variable values using fuzzy matching."""
    if pd.isna(gt_value) or pd.isna(student_value):
        return False
    
    try:
        gt_str = str(gt_value).strip()
        student_str = str(student_value).strip()
    except (TypeError, ValueError):
        return False
    
    # Handle empty strings and "nan" values
    gt_is_empty_or_nan = gt_str == "" or gt_str.lower() == "nan"
    student_is_empty_or_nan = student_str == "" or student_str.lower() == "nan"
    
    # If both are empty/nan, they match
    if gt_is_empty_or_nan and student_is_empty_or_nan:
        return True
    
    # If only one is empty/nan, they don't match
    if gt_is_empty_or_nan or student_is_empty_or_nan:
        return False
    
    # Special handling for patent_id: remove .0 suffix if present
    if gt_str.isdigit() and student_str.endswith('.0'):
        student_str = student_str[:-2]
    
    similarity = Levenshtein.normalized_similarity(gt_str, student_str)
    return similarity >= threshold


# --- HTML Generation ---

def html_escape(s: str) -> str:
    """Escapes a string for safe inclusion in HTML."""
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')


def make_table_html(df: pd.DataFrame, matches: List[bool], match_ids: List[str], title: str) -> str:
    """Creates an HTML table from a dataframe."""
    rows_html = []
    for i, row in df.iterrows():
        color = '#d4edda' if matches[i] else '#f8d7da'  # green/red
        rows_html.append(
            f'<tr style="background-color:{color}">'
            f'<td>{html_escape(row["id"])}</td>'
            f'<td>{html_escape(row["entry"])}</td>'
            f'<td>{html_escape(match_ids[i])}</td>'
            f'</tr>'
        )
    return (
        f'<table class="benchmark-table">\n'
        f'<caption>{title}</caption>\n'
        f'<tr><th>ID</th><th>Entry</th><th>Match ID</th></tr>\n'
        f'{"".join(rows_html)}\n'
        f'</table>'
    )


# --- Stage 1: Patent Entry Matching ---

def run_stage1_entry_matching(threshold: float = 0.85):
    """
    Stage 1: Compare student entries to perfect entries.
    Generates patent_entry_matching.html report.
    """
    logging.info("=== Stage 1: Patent Entry Matching ===")
    
    output_dir = OUTPUT_BASE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    common_stems = get_common_file_stems()
    if not common_stems:
        logging.error("No common files found between student and perfect directories.")
        return None
    
    logging.info(f"Found {len(common_stems)} common files to process")
    
    # Collect results
    comparison_results = []
    total_perfect_entries = 0
    total_student_entries = 0
    total_matches = 0
    
    for stem in common_stems:
        logging.info(f"Processing: {stem}")
        
        # Load files
        student_path = STUDENT_XLSX_DIR / f"{stem}.xlsx"
        perfect_path = PERFECT_XLSX_DIR / f"{stem}_perfected.xlsx"
        
        if not student_path.exists() or not perfect_path.exists():
            logging.warning(f"Skipping {stem}: files not found")
            continue
        
        student_df = load_xlsx_file(student_path)
        perfect_df = load_xlsx_file(perfect_path)
        
        if student_df.empty or perfect_df.empty:
            logging.warning(f"Skipping {stem}: empty dataframe")
            continue
        
        # Perform fuzzy matching
        perfect_matches, student_matches, perfect_match_ids, student_match_ids = match_entries_fuzzy(
            perfect_df, student_df, threshold
        )
        
        # Create HTML tables
        perfect_table_html = make_table_html(perfect_df, perfect_matches, perfect_match_ids, 'Perfect Transcription')
        student_table_html = make_table_html(student_df, student_matches, student_match_ids, 'Student Transcription')
        
        # Store results
        matches_count = sum(perfect_matches)
        comparison_results.append({
            'filename': stem,
            'perfect_table': perfect_table_html,
            'student_table': student_table_html,
            'perfect_entries': len(perfect_df),
            'student_entries': len(student_df),
            'matches': matches_count
        })
        
        total_perfect_entries += len(perfect_df)
        total_student_entries += len(student_df)
        total_matches += matches_count
    
    # Generate HTML report
    generate_entry_matching_report(comparison_results, total_perfect_entries, total_student_entries, 
                                   total_matches, threshold, output_dir)
    
    return {
        'files_processed': len(comparison_results),
        'total_perfect_entries': total_perfect_entries,
        'total_student_entries': total_student_entries,
        'total_matches': total_matches
    }


def generate_entry_matching_report(comparison_results: List[Dict], total_perfect: int, total_student: int,
                                   total_matches: int, threshold: float, output_dir: Path):
    """Generate the patent entry matching HTML report."""
    
    # Calculate overall statistics
    match_rate_perfect = (total_matches / total_perfect * 100) if total_perfect > 0 else 0
    match_rate_student = (total_matches / total_student * 100) if total_student > 0 else 0
    
    # Build sections HTML
    sections_html = []
    for result in comparison_results:
        section = f'''
        <section class="comparison-section">
            <h2>{html_escape(result['filename'])}.pdf</h2>
            <div class="metrics">
                <b>Perfect Entries:</b> {result['perfect_entries']} &nbsp;
                <b>Student Entries:</b> {result['student_entries']} &nbsp;
                <b>Matches:</b> {result['matches']} &nbsp;
                <b>Match Rate (Perfect):</b> {(result['matches'] / result['perfect_entries'] * 100) if result['perfect_entries'] > 0 else 0:.2f}%
            </div>
            <div class="table-container">
                {result['perfect_table']}
                {result['student_table']}
            </div>
        </section>
        '''
        sections_html.append(section)
    
    # Build summary HTML
    summary_html = f'''
    <div class="summary-section">
        <h2>Patent Entry Matching Summary - <em>Student-constructed</em> vs <em>Perfect</em></h2>
        <p><strong>Fuzzy Matching Threshold:</strong> {threshold}</p>
        <p><strong>Total <em>Perfect</em> Entries:</strong> {total_perfect}</p>
        <p><strong>Total <em>Student-constructed</em> Entries:</strong> {total_student}</p>
        <p><strong>Total Matches:</strong> {total_matches}</p>
        <p><strong>Match Rate (<em>Perfect</em> perspective):</strong> {match_rate_perfect:.2f}%</p>
        <p><strong>Match Rate (<em>Student-constructed</em> perspective):</strong> {match_rate_student:.2f}%</p>
    </div>
    '''
    
    # Build full HTML
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Patent Entry Matching - Student-constructed vs Perfect Entries</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f4f4f9; color: #222; margin: 0; }}
        .container {{ max-width: 1400px; margin: auto; padding: 20px; }}
        h1 {{ text-align: center; color: #444; }}
        .summary-section {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 30px; }}
        .comparison-section {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 30px; padding: 20px; }}
        .metrics {{ margin-bottom: 15px; font-size: 1.1em; }}
        .table-container {{ display: flex; gap: 20px; overflow-x: auto; }}
        .benchmark-table {{ flex: 1; min-width: 400px; border-collapse: collapse; margin: 10px 0; }}
        .benchmark-table th, .benchmark-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .benchmark-table th {{ background-color: #f2f2f2; font-weight: bold; }}
        .benchmark-table caption {{ font-weight: bold; margin-bottom: 10px; font-size: 1.1em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Patent Entry Matching - <em>Student-constructed</em> vs <em>Perfect</em> Entries</h1>
        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; margin-bottom: 30px;">
            <p style="margin: 0; color: #856404;"><strong>Note:</strong> This report compares <em>student-constructed</em> (research assistant) transcriptions against <em>perfect</em> (error-free) transcriptions. Any unmatched entries indicate human transcription errors or omissions.</p>
        </div>
        {summary_html}
        {''.join(sections_html)}
    </div>
</body>
</html>'''
    
    report_path = output_dir / 'patent_entry_matching.html'
    report_path.write_text(html_content, encoding='utf-8')
    logging.info(f"Patent entry matching report saved to: {report_path}")


# --- Stage 3: Variable Extraction Comparison ---

def run_stage3_variable_extraction(threshold: float = 0.85):
    """
    Stage 3: Compare student-extracted variables to perfect variables.
    Generates variable_extraction_report.html report.
    """
    logging.info("=== Stage 3: Variable Extraction Comparison ===")
    
    output_dir = OUTPUT_BASE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    common_stems = get_common_file_stems()
    if not common_stems:
        logging.error("No common files found between student and perfect directories.")
        return None
    
    logging.info(f"Found {len(common_stems)} common files to process")
    
    # Collect all data for global threshold sensitivity analysis
    all_perfect_dfs = []
    all_student_dfs = []
    all_matched_pairs = []  # (file_idx, perfect_idx, student_idx)
    
    # First pass: collect all data and perform fuzzy matching
    for stem in common_stems:
        logging.info(f"Processing: {stem}")
        
        # Load files
        student_path = STUDENT_XLSX_DIR / f"{stem}.xlsx"
        perfect_path = PERFECT_XLSX_DIR / f"{stem}_perfected.xlsx"
        
        if not student_path.exists() or not perfect_path.exists():
            logging.warning(f"Skipping {stem}: files not found")
            continue
        
        student_df = load_xlsx_file(student_path)
        perfect_df = load_xlsx_file(perfect_path)
        
        if student_df.empty or perfect_df.empty:
            logging.warning(f"Skipping {stem}: empty dataframe")
            continue
        
        # Ensure variable columns exist (trim whitespace from column names was done in load_xlsx_file)
        for field in VARIABLE_FIELDS:
            if field not in perfect_df.columns:
                perfect_df[field] = "NaN"
            else:
                if field == "patent_id":
                    perfect_df[field] = perfect_df[field].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                else:
                    perfect_df[field] = perfect_df[field].astype(str).str.strip()
            
            if field not in student_df.columns:
                student_df[field] = "NaN"
            else:
                if field == "patent_id":
                    student_df[field] = student_df[field].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                else:
                    student_df[field] = student_df[field].astype(str).str.strip()
        
        # Perform fuzzy matching on entry field
        perfect_matches, student_matches, perfect_match_ids, student_match_ids = match_entries_fuzzy(
            perfect_df, student_df, threshold
        )
        
        # Collect matched pairs
        file_idx = len(all_perfect_dfs)
        for i, matched in enumerate(perfect_matches):
            if matched:
                student_idx = int(perfect_match_ids[i]) - 1  # Convert to 0-based
                if student_idx < len(student_df):
                    all_matched_pairs.append((file_idx, i, student_idx))
        
        all_perfect_dfs.append(perfect_df)
        all_student_dfs.append(student_df)
    
    if not all_matched_pairs:
        logging.error("No matched pairs found for variable comparison")
        return None
    
    # Calculate global threshold sensitivity
    threshold_sensitivity = calculate_threshold_sensitivity(all_perfect_dfs, all_student_dfs, all_matched_pairs)
    
    # Calculate statistics at default threshold
    total_cells, matched_cells, variable_stats = calculate_variable_stats(
        all_perfect_dfs, all_student_dfs, all_matched_pairs, threshold
    )
    
    # Generate individual file sections
    file_sections = generate_file_sections(
        common_stems, all_perfect_dfs, all_student_dfs, threshold
    )
    
    # Generate HTML report
    generate_variable_extraction_report(
        threshold_sensitivity, total_cells, matched_cells, variable_stats,
        file_sections, threshold, output_dir
    )
    
    return {
        'files_processed': len(all_perfect_dfs),
        'total_cells': total_cells,
        'matched_cells': matched_cells,
        'overall_match_rate': (matched_cells / total_cells * 100) if total_cells > 0 else 0
    }


def calculate_threshold_sensitivity(all_perfect_dfs: List, all_student_dfs: List, all_matched_pairs: List) -> Dict:
    """Calculate match rates for different thresholds across all files."""
    thresholds = [round(t * 0.1, 1) for t in range(11)]  # 0.0, 0.1, ..., 1.0
    results = {}
    
    for thresh in thresholds:
        total_cells = len(all_matched_pairs) * len(VARIABLE_FIELDS)
        matched_cells = 0
        variable_stats = {field: {'total': 0, 'matched': 0} for field in VARIABLE_FIELDS}
        
        for file_idx, perfect_idx, student_idx in all_matched_pairs:
            perfect_df = all_perfect_dfs[file_idx]
            student_df = all_student_dfs[file_idx]
            
            for field in VARIABLE_FIELDS:
                perfect_value = str(perfect_df.iloc[perfect_idx].get(field, "NaN"))
                student_value = str(student_df.iloc[student_idx].get(field, "NaN"))
                
                is_match = compare_variables(perfect_value, student_value, thresh)
                variable_stats[field]['total'] += 1
                if is_match:
                    variable_stats[field]['matched'] += 1
                    matched_cells += 1
        
        overall_rate = (matched_cells / total_cells * 100) if total_cells > 0 else 0
        variable_rates = {}
        for field in VARIABLE_FIELDS:
            stats = variable_stats[field]
            rate = (stats['matched'] / stats['total'] * 100) if stats['total'] > 0 else 0
            variable_rates[field] = rate
        
        results[thresh] = {
            'overall_rate': overall_rate,
            'variable_rates': variable_rates
        }
    
    return results


def calculate_variable_stats(all_perfect_dfs: List, all_student_dfs: List, all_matched_pairs: List, 
                             threshold: float) -> Tuple[int, int, Dict]:
    """Calculate variable-level statistics at a given threshold."""
    total_cells = len(all_matched_pairs) * len(VARIABLE_FIELDS)
    matched_cells = 0
    variable_stats = {field: {'total': 0, 'matched': 0} for field in VARIABLE_FIELDS}
    
    for file_idx, perfect_idx, student_idx in all_matched_pairs:
        perfect_df = all_perfect_dfs[file_idx]
        student_df = all_student_dfs[file_idx]
        
        for field in VARIABLE_FIELDS:
            perfect_value = str(perfect_df.iloc[perfect_idx].get(field, "NaN"))
            student_value = str(student_df.iloc[student_idx].get(field, "NaN"))
            
            is_match = compare_variables(perfect_value, student_value, threshold)
            variable_stats[field]['total'] += 1
            if is_match:
                variable_stats[field]['matched'] += 1
                matched_cells += 1
    
    return total_cells, matched_cells, variable_stats


def generate_file_sections(common_stems: List[str], all_perfect_dfs: List, all_student_dfs: List, 
                           threshold: float) -> List[str]:
    """Generate HTML sections for each file."""
    file_sections = []
    
    for i, stem in enumerate(common_stems):
        if i >= len(all_perfect_dfs):
            continue
            
        perfect_df = all_perfect_dfs[i]
        student_df = all_student_dfs[i]
        
        # Perform fuzzy matching
        perfect_matches, student_matches, perfect_match_ids, student_match_ids = match_entries_fuzzy(
            perfect_df, student_df, threshold
        )
        
        # Get matched pairs for this file
        matched_pairs = []
        for j, matched in enumerate(perfect_matches):
            if matched:
                student_idx = int(perfect_match_ids[j]) - 1
                if student_idx < len(student_df):
                    matched_pairs.append((j, student_idx))
        
        if not matched_pairs:
            continue
        
        # Build table rows
        table_rows = []
        file_matched_cells = 0
        file_total_cells = len(matched_pairs) * len(VARIABLE_FIELDS)
        
        for perfect_idx, student_idx in matched_pairs:
            row_cells = []
            for field in VARIABLE_FIELDS:
                perfect_value = str(perfect_df.iloc[perfect_idx].get(field, "NaN"))
                student_value = str(student_df.iloc[student_idx].get(field, "NaN"))
                
                is_match = compare_variables(perfect_value, student_value, threshold)
                if is_match:
                    file_matched_cells += 1
                    bg_color = "#d4edda"  # Green
                else:
                    bg_color = "#f8d7da"  # Red
                
                safe_perfect = html_escape(perfect_value)
                safe_student = html_escape(student_value)
                cell_content = f"{safe_perfect} / {safe_student}"
                row_cells.append(f'<td style="background-color:{bg_color}">{cell_content}</td>')
            
            table_rows.append(f"<tr>{''.join(row_cells)}</tr>")
        
        file_match_rate = (file_matched_cells / file_total_cells * 100) if file_total_cells > 0 else 0
        
        header_cells = ''.join([f'<th>{get_display_name(field)}</th>' for field in VARIABLE_FIELDS])
        section_html = f'''
        <section class="pair-section">
            <h2>File: <span class="filename">{stem}</span></h2>
            <div class="metrics">
                <b>Total Cells:</b> {file_total_cells} &nbsp;
                <b>Matched Cells:</b> {file_matched_cells} &nbsp;
                <b>Match Rate:</b> {file_match_rate:.2f}%
            </div>
            <div class="table-legend">
                <p><strong>Legend:</strong> Values in each cell show "Perfect / Student"</p>
            </div>
            <table class="benchmark-table">
                <tr>{header_cells}</tr>
                {''.join(table_rows)}
            </table>
        </section>
        '''
        file_sections.append(section_html)
    
    return file_sections


def generate_variable_extraction_report(threshold_sensitivity: Dict, total_cells: int, matched_cells: int,
                                        variable_stats: Dict, file_sections: List[str], threshold: float,
                                        output_dir: Path):
    """Generate the variable extraction HTML report."""
    
    # Calculate overall match rate
    overall_match_rate = (matched_cells / total_cells * 100) if total_cells > 0 else 0
    
    # Calculate variable-level match rates
    variable_rates = {}
    for field in VARIABLE_FIELDS:
        stats = variable_stats[field]
        rate = (stats['matched'] / stats['total'] * 100) if stats['total'] > 0 else 0
        variable_rates[field] = rate
    
    # Build threshold sensitivity table
    threshold_rows = []
    for thresh in sorted(threshold_sensitivity.keys()):
        data = threshold_sensitivity[thresh]
        row_cells = [f'<td>{thresh}</td>', f'<td>{data["overall_rate"]:.2f}%</td>']
        for field in VARIABLE_FIELDS:
            row_cells.append(f'<td>{data["variable_rates"][field]:.2f}%</td>')
        threshold_rows.append(f"<tr>{''.join(row_cells)}</tr>")
    
    threshold_header = ''.join([f'<th>{get_display_name(field)}</th>' for field in VARIABLE_FIELDS])
    threshold_table_html = f'''
    <div class="global-threshold-analysis">
        <h2>Global Threshold Sensitivity Analysis</h2>
        <p>This table shows how match rates change across all files for different similarity thresholds.</p>
        <table class="benchmark-table">
            <tr><th>Threshold</th><th>Overall</th>{threshold_header}</tr>
            {''.join(threshold_rows)}
        </table>
    </div>
    '''
    
    # Build summary HTML
    summary_html = f'''
    <div class="summary-section">
        <h2>Overall Variable Extraction Summary - <em>Student-constructed</em> vs <em>Perfect</em></h2>
        <p><b>Fuzzy Matching Threshold:</b> {threshold}</p>
        <p><b>Total Cells:</b> {total_cells}</p>
        <p><b>Matched Cells:</b> {matched_cells}</p>
        <p><b>Overall Match Rate:</b> {overall_match_rate:.2f}%</p>
        <p><b>Variable Match Rates:</b></p>
        <ul>
            {''.join([f'<li><b>{get_display_name(field)}:</b> {rate:.2f}%</li>' for field, rate in variable_rates.items()])}
        </ul>
    </div>
    '''
    
    # Build full HTML
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Variable Extraction: Student-constructed vs Perfect</title>
    <style>
        body {{ font-family: sans-serif; margin: 0; background-color: #f4f4f9; color: #333; }}
        .container {{ max-width: 1400px; margin: auto; padding: 20px; }}
        h1, h2 {{ color: #444; }}
        h1 {{ text-align: center; }}
        .global-threshold-analysis {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 30px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .global-threshold-analysis h2 {{ margin-top: 0; color: #2c3e50; }}
        .global-threshold-analysis p {{ color: #666; margin-bottom: 20px; }}
        .pair-section {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 20px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .metrics {{ margin-bottom: 15px; font-size: 1.1em; }}
        .table-legend {{ margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff; }}
        .table-legend p {{ margin: 0; font-weight: 500; color: #495057; }}
        .benchmark-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        .benchmark-table th, .benchmark-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .benchmark-table th {{ background-color: #f2f2f2; font-weight: bold; }}
        .filename {{ font-family: monospace; background: #eee; padding: 2px 5px; border-radius: 4px; }}
        .summary-section {{ margin-bottom: 30px; padding: 20px; background: #fff; border-radius: 8px; border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Variable Extraction: <em>Student-constructed</em> vs <em>Perfect</em></h1>
        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; margin-bottom: 30px;">
            <p style="margin: 0; color: #856404;"><strong>Note:</strong> This report compares <em>student-constructed</em> extracted variables against <em>perfect</em> (error-free) transcriptions. Any mismatches indicate human variable extraction errors.</p>
        </div>
        {summary_html}
        {threshold_table_html}
        <div style="margin-top: 40px;">
            {''.join(file_sections)}
        </div>
    </div>
</body>
</html>'''
    
    report_path = output_dir / 'variable_extraction_report.html'
    report_path.write_text(html_content, encoding='utf-8')
    logging.info(f"Variable extraction report saved to: {report_path}")


# --- Main Function ---

def main():
    """Main function to run the student-constructed benchmarking."""
    parser = argparse.ArgumentParser(description="Run student vs perfect transcription benchmarking.")
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.9,
        help='Fuzzy matching threshold (0.0-1.0). Default: 0.9'
    )
    parser.add_argument(
        '--stage',
        type=str,
        choices=['all', '1', '3'],
        default='all',
        help='Which stage to run: "all", "1" (entry matching), or "3" (variable extraction). Default: all'
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not (0.0 <= args.threshold <= 1.0):
        logging.error(f"Threshold must be between 0.0 and 1.0, got: {args.threshold}")
        sys.exit(1)
    
    logging.info("=" * 60)
    logging.info("Student-Constructed Benchmarking")
    logging.info("Comparing Student Transcriptions to Perfect Transcriptions")
    logging.info("=" * 60)
    
    results = {}
    
    # Run Stage 1: Entry Matching
    if args.stage in ['all', '1']:
        stage1_results = run_stage1_entry_matching(args.threshold)
        if stage1_results:
            results['stage1'] = stage1_results
            logging.info(f"Stage 1 complete: {stage1_results['files_processed']} files processed")
    
    # Run Stage 3: Variable Extraction
    if args.stage in ['all', '3']:
        stage3_results = run_stage3_variable_extraction(args.threshold)
        if stage3_results:
            results['stage3'] = stage3_results
            logging.info(f"Stage 3 complete: {stage3_results['files_processed']} files processed")
    
    logging.info("=" * 60)
    logging.info("Benchmarking Complete!")
    logging.info(f"Results saved to: {OUTPUT_BASE_DIR}")
    logging.info("=" * 60)
    
    return results


if __name__ == "__main__":
    main()

