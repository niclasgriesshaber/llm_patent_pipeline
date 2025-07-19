import os
import pandas as pd
from pathlib import Path
from rapidfuzz.distance import Levenshtein
from collections import defaultdict
import re
import difflib
import json
import logging
from typing import List, Tuple, Dict, Any
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Text Normalization Utilities ---

def normalize_text_for_cer(text: str) -> str:
    """
    Normalize text for CER calculation: ASCII a-z, 0-9, lowercase, no linebreaks, trimmed.
    This matches academic CER reporting standards.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove linebreaks and normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Keep only ASCII letters a-z and digits 0-9
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Trim whitespace
    text = text.strip()
    
    return text

def create_normalized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create a normalized version of a dataframe for CER comparison."""
    if df.empty:
        return df.copy()
    
    normalized_df = df.copy()
    normalized_df['entry'] = normalized_df['entry'].apply(normalize_text_for_cer)
    return normalized_df

# --- Data Loading ---

def get_file_stems(directory: Path, ext: str) -> set:
    """Gets the set of file stems from a directory."""
    return {f.stem for f in directory.glob(f'*.{ext}')}

def load_gt_file(filepath: Path) -> pd.DataFrame:
    """Loads and preprocesses a ground truth Excel file."""
    try:
        df = pd.read_excel(filepath, dtype=str)
        if 'id' not in df.columns or 'entry' not in df.columns:
            raise ValueError(f"File {filepath} missing 'id' or 'entry' column.")
        df = df[['id', 'entry']].dropna(subset=['entry'])
        df = df[df['entry'].astype(str).str.strip() != ''].copy()
        df['entry'] = df['entry'].str.normalize('NFC')
        return df.reset_index(drop=True)
    except Exception as e:
        logging.error(f"Error loading GT file {filepath}: {e}")
        return pd.DataFrame({'id': [], 'entry': []})

def load_llm_file(filepath: Path) -> pd.DataFrame:
    """Loads and preprocesses an LLM-generated CSV file."""
    try:
        df = pd.read_csv(filepath, dtype=str)
        # Handle cases where LLM might output different column names
        if 'id' not in df.columns and 'Id' in df.columns:
            df = df.rename(columns={'Id': 'id'})
        if 'entry' not in df.columns and 'Entry' in df.columns:
            df = df.rename(columns={'Entry': 'entry'})
            
        if 'id' not in df.columns or 'entry' not in df.columns:
            # If still not found, try to infer from a common structure
            if len(df.columns) >= 2:
                df = df.iloc[:, :2]
                df.columns = ['id', 'entry']
                logging.warning(f"File {filepath} missing 'id' or 'entry' columns. Inferred from first two columns.")
            else:
                 raise ValueError(f"File {filepath} missing 'id' or 'entry' column.")
        
        df = df[['id', 'entry']].dropna(subset=['entry'])
        df = df[df['entry'].astype(str).str.strip() != ''].copy()
        df['entry'] = df['entry'].str.normalize('NFC')
        return df.reset_index(drop=True)
    except Exception as e:
        logging.error(f"Error loading LLM file {filepath}: {e}")
        return pd.DataFrame({'id': [], 'entry': []})

# --- Fuzzy Matching Logic ---

def match_entries_fuzzy(gt_df: pd.DataFrame, llm_df: pd.DataFrame, threshold: float = 0.85) -> Tuple[List[bool], List[bool], List[str], List[str]]:
    """Performs bidirectional fuzzy matching between two dataframes."""
    gt_entries = gt_df['entry'].astype(str).tolist()
    llm_entries = llm_df['entry'].astype(str).tolist()
    gt_ids = gt_df['id'].astype(str).tolist()
    llm_ids = llm_df['id'].astype(str).tolist()

    gt_matches = [False] * len(gt_entries)
    llm_matches = [False] * len(llm_entries)
    gt_match_ids = ['—'] * len(gt_entries)
    llm_match_ids = ['—'] * len(llm_entries)

    # Use a set to track matched indices to avoid re-matching
    used_llm_indices = set()
    used_gt_indices = set()

    # Match from GT to LLM
    for i, gt_entry in enumerate(gt_entries):
        best_score = -1
        best_idx = -1
        for j, llm_entry in enumerate(llm_entries):
            if j in used_llm_indices:
                continue
            score = Levenshtein.normalized_similarity(gt_entry, llm_entry)
            if score > best_score:
                best_score = score
                best_idx = j
        if best_score >= threshold and best_idx != -1:
            gt_matches[i] = True
            llm_matches[best_idx] = True
            gt_match_ids[i] = llm_ids[best_idx]
            llm_match_ids[best_idx] = gt_ids[i]
            used_llm_indices.add(best_idx)
            used_gt_indices.add(i)

    # Match from LLM to GT (for any LLM entries not yet matched)
    for j, llm_entry in enumerate(llm_entries):
        if j in used_llm_indices:
            continue
        best_score = -1
        best_idx = -1
        for i, gt_entry in enumerate(gt_entries):
            if i in used_gt_indices:
                continue
            score = Levenshtein.normalized_similarity(llm_entry, gt_entry)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_score >= threshold and best_idx != -1:
            llm_matches[j] = True
            gt_matches[best_idx] = True
            llm_match_ids[j] = gt_ids[best_idx]
            # Ensure the GT match ID is also updated if it wasn't before
            if gt_match_ids[best_idx] == '—':
                gt_match_ids[best_idx] = llm_ids[j]
            used_llm_indices.add(j)
            used_gt_indices.add(i)
            
    return gt_matches, llm_matches, gt_match_ids, llm_match_ids

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

def make_metrics_html(gt_matches: int, llm_matches: int, gt_len: int, llm_len: int) -> str:
    """Creates an HTML snippet for performance metrics."""
    gt_unmatched = gt_len - gt_matches
    llm_unmatched = llm_len - llm_matches
    total = gt_len + llm_len
    total_matched = gt_matches
    match_rate = (total_matched / gt_len * 100) if gt_len else 0
    return (
        f'<div class="metrics">'
        f'<b>GT Matched:</b> {gt_matches}/{gt_len} &nbsp; '
        f'<b>LLM Matched:</b> {llm_matches}/{llm_len} &nbsp; '
        f'<b>GT Unmatched:</b> {gt_unmatched} &nbsp; '
        f'<b>LLM Unmatched:</b> {llm_unmatched} &nbsp; '
        f'<b>Match Rate (GT perspective):</b> {match_rate:.2f}%<br>'
        f'<span style="font-size:0.9em; color:#555;">Match Rate = (GT Matched / GT Total Entries)</span>'
        f'</div>'
    )
    
def make_pair_section_html(gt_df, llm_df, gt_matches, llm_matches, gt_match_ids, llm_match_ids, filename_stem) -> str:
    """Creates an HTML section for a single pair of files."""
    metrics_html = make_metrics_html(sum(gt_matches), sum(llm_matches), len(gt_df), len(llm_df))
    gt_table_html = make_table_html(gt_df, gt_matches, gt_match_ids, 'Ground Truth')
    llm_table_html = make_table_html(llm_df, llm_matches, llm_match_ids, 'LLM Output')
    return (
        f'<section class="pair-section">'
        f'<h2>File Pair: <span class="filename">{html_escape(filename_stem)}</span></h2>'
        f'{metrics_html}'
        f'<div class="table-container">{gt_table_html}{llm_table_html}</div>'
        f'</section>'
    )

def make_full_html(title: str, sections_html: str, summary_html: str) -> str:
    """Constructs the final HTML report."""
    css = """
    <style>
        body { font-family: sans-serif; margin: 0; background-color: #f4f4f9; color: #333; }
        .container { max-width: 1200px; margin: auto; padding: 20px; }
        h1, h2 { color: #444; }
        h1 { text-align: center; }
        .pair-section { background: #fff; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 20px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .metrics { margin-bottom: 15px; font-size: 1.1em; }
        .table-container { display: flex; flex-wrap: wrap; gap: 20px; }
        .benchmark-table { flex: 1; min-width: 400px; border-collapse: collapse; }
        .benchmark-table th, .benchmark-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .benchmark-table th { background-color: #f2f2f2; }
        .benchmark-table caption { font-weight: bold; margin-bottom: 10px; font-size: 1.2em; }
        .filename { font-family: monospace; background: #eee; padding: 2px 5px; border-radius: 4px; }
        .summary-section { margin-top: 30px; padding: 20px; background: #fff; border-radius: 8px; }
    </style>
    """
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f'<title>{title}</title>{css}</head><body><div class="container">'
        f'<h1>{title}</h1>{summary_html}{sections_html}</div></body></html>'
    )

def extract_year_from_filename(filename: str) -> str:
    match = re.search(r'(\d{4})', filename)
    return match.group(1) if match else ''

def compute_levenshtein_stats(gt_text: str, llm_text: str):
    ops = Levenshtein.editops(gt_text, llm_text)
    ins = sum(1 for op in ops if op[0] == 'insert')
    del_ = sum(1 for op in ops if op[0] == 'delete')
    sub = sum(1 for op in ops if op[0] == 'replace')
    return ins, del_, sub

def make_summary_table_html(summary_rows):
    table = [
        '<table class="summary-table">',
        '<caption style="font-weight: bold; margin-bottom: 18px; font-size: 1.2em;">File-level CER and Edit Statistics</caption>',
        '<tr><th>File</th><th>Year</th><th>CER (Unnorm)</th><th>CER (Norm)</th><th>Words (GT)</th><th>Words (LLM)</th><th>Chars (GT)</th><th>Chars (LLM)</th><th>Insertions</th><th>Deletions</th><th>Substitutions</th></tr>'
    ]
    for row in summary_rows:
        table.append(
            f'<tr>'
            f'<td>{html_escape(row["file"])}<br></td>'
            f'<td>{html_escape(row["year"])}<br></td>'
            f'<td>{row["cer"]:.2%}</td>'
            f'<td>{row["cer_normalized"]:.2%}</td>'
            f'<td>{row["words_gt"]}</td>'
            f'<td>{row["words_llm"]}</td>'
            f'<td>{row["chars_gt"]}</td>'
            f'<td>{row["chars_llm"]}</td>'
            f'<td>{row["ins"]}</td>'
            f'<td>{row["del"]}</td>'
            f'<td>{row["sub"]}</td>'
            f'</tr>'
        )
    table.append('</table>')
    return '\n'.join(table)

def make_interactive_cer_graph(summary_rows):
    years = [int(row['year']) for row in summary_rows]
    cers = [row['cer'] for row in summary_rows]
    files = [row['file'] for row in summary_rows]
    return f'''
<div id="cer-graph" style="height:400px;"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
var data = [{{
    x: {years},
    y: {cers},
    text: {files},
    type: 'scatter',
    mode: 'markers+lines',
    marker: {{ size: 12 }},
    hovertemplate: 'File: %{{text}}<br>Year: %{{x}}<br>CER: %{{y:.2%}}<extra></extra>'
}}];
var layout = {{
    title: 'CER by Year',
    xaxis: {{ title: 'Year', tickangle: -90, dtick: 1 }},
    yaxis: {{ title: 'CER', tickformat: ',.0%' }},
    margin: {{ t: 40, b: 120 }}
}};
Plotly.newPlot('cer-graph', data, layout);
</script>
'''

def make_side_by_side_diff(gt_text, llm_text, file, year, cer):
    # Normalize whitespace for comparison to reduce false positives
    # but keep original text for display
    gt_lines = gt_text.splitlines()
    llm_lines = llm_text.splitlines()
    
    # Normalize whitespace for comparison
    gt_lines_normalized = [line.rstrip() for line in gt_lines]
    llm_lines_normalized = [line.rstrip() for line in llm_lines]
    
    diff_html = difflib.HtmlDiff(wrapcolumn=80).make_table(
        gt_lines_normalized,
        llm_lines_normalized,
        fromdesc='<span class="diff-table-header">Ground Truth</span>',
        todesc='<span class="diff-table-header">LLM Output</span>',
        context=False,  # Show all changes, not just context
        numlines=2
    )
    return f'<section class="diff-section"><h2 class="diff-file-heading">{html_escape(file)}</h2><h3 class="diff-cer">CER: {cer:.2%}</h3>{diff_html}</section>'

# --- Main Comparison Logic ---

def run_comparison(llm_csv_dir: Path, gt_xlsx_dir: Path, output_dir: Path, fuzzy_threshold: float = 0.85, comparison_type: str = "perfect"):
    """
    Runs the full comparison logic: fuzzy matching and diffing.
    Generates HTML reports and a JSON summary.
    
    Args:
        llm_csv_dir: Directory containing LLM-generated CSV files
        gt_xlsx_dir: Directory containing ground truth Excel files
        output_dir: Directory to save comparison results
        fuzzy_threshold: Threshold for fuzzy matching (default: 0.85)
        comparison_type: Type of comparison ("perfect" or "student")
    """
    logging.info(f"Starting {comparison_type} comparison for data in {llm_csv_dir.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_stems = get_file_stems(gt_xlsx_dir, 'xlsx')
    llm_stems = get_file_stems(llm_csv_dir, 'csv')
    common_stems = sorted(list(gt_stems.intersection(llm_stems)))

    if not common_stems:
        logging.warning(f"No common files found between {comparison_type} ground truth and LLM directories. Skipping comparison.")
        return None

    # --- Fuzzy Matching ---
    pair_sections_html = []
    total_gt_entries, total_llm_entries = 0, 0
    total_gt_matched, total_llm_matched = 0, 0
    full_gt_text, full_llm_text = "", ""

    for stem in common_stems:
        logging.info(f"Processing {comparison_type} pair: {stem}")
        gt_df = load_gt_file(gt_xlsx_dir / f"{stem}.xlsx")
        llm_df = load_llm_file(llm_csv_dir / f"{stem}.csv")

        if gt_df.empty or llm_df.empty:
            logging.warning(f"Skipping {comparison_type} pair {stem} due to empty dataframe after loading.")
            continue

        full_gt_text += "\n".join(gt_df['entry'].tolist()) + "\n"
        full_llm_text += "\n".join(llm_df['entry'].tolist()) + "\n"

        gt_matches, llm_matches, gt_match_ids, llm_match_ids = match_entries_fuzzy(gt_df, llm_df, fuzzy_threshold)

        pair_sections_html.append(make_pair_section_html(gt_df, llm_df, gt_matches, llm_matches, gt_match_ids, llm_match_ids, stem))
        
        total_gt_entries += len(gt_df)
        total_llm_entries += len(llm_df)
        total_gt_matched += sum(gt_matches)
        total_llm_matched += sum(llm_matches)

    # --- Aggregation and Report Generation ---
    overall_match_rate = (total_gt_matched / total_gt_entries * 100) if total_gt_entries > 0 else 0
    
    summary_html = (
        f'<div class="summary-section">'
        f'<h2>Overall Summary - {comparison_type.title()} Comparison</h2>'
        f'<p><b>Total GT Entries:</b> {total_gt_entries}</p>'
        f'<p><b>Total LLM Entries:</b> {total_llm_entries}</p>'
        f'<p><b>Total GT Matched:</b> {total_gt_matched}</p>'
        f'<p><b>Total LLM Matched:</b> {total_llm_matched}</p>'
        f'<p><b>Overall Match Rate (GT perspective):</b> {overall_match_rate:.2f}%</p>'
        f'</div>'
    )
    
    fuzzy_html_content = make_full_html(f"Fuzzy Matching Report - {comparison_type.title()}", "".join(pair_sections_html), summary_html)
    fuzzy_report_path = output_dir / "fuzzy_report.html"
    fuzzy_report_path.write_text(fuzzy_html_content, encoding='utf-8')
    logging.info(f"Fuzzy report saved to {fuzzy_report_path}")

    # --- CER and Diffing (Normalized) ---
    summary_rows = []
    diff_sections = []
    
    # Style section with improved diff table responsiveness
    style = '''<style>
    body{font-family:Segoe UI,Arial,sans-serif;background:#f4f4f9;color:#222;}
    .container{max-width:1200px;margin:auto;padding:30px;}
    .summary-section{margin-top:30px;padding:24px 32px;background:#fff;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.04);}
    .summary-table{width:100%;border-collapse:collapse;margin-bottom:30px;font-size:1.05em;}
    .summary-table th,.summary-table td{border:1px solid #ddd;padding:10px 8px;text-align:left;}
    .summary-table th{background:#e9e9f2;font-weight:600;}
    .summary-table caption{font-weight: bold; margin-bottom: 18px; font-size: 1.2em;}
    .diff-section{background:#f9fafc;border:1px solid #dbe2ea;border-radius:14px;margin-bottom:40px;padding:28px 22px;box-shadow:0 4px 16px rgba(0,0,0,0.06);overflow-x:auto;}
    .diff-file-heading{font-size:1.35em;font-weight:700;margin-bottom:8px;letter-spacing:0.5px;}
    .diff-cer{font-size:1.08em;font-weight:600;margin-bottom:18px;}
    table.diff{font-size:1em;max-width:100%;width:100%;word-break:break-word;background:#fff;border-radius:8px;border:1.5px solid #e0e0e0;box-shadow:0 1px 4px rgba(0,0,0,0.03);margin-bottom:0;}
    td.diff_header{background:#f2f2f2;font-weight:bold;font-size:1.08em;}
    td.diff_next{background:#e9e9f2;}
    span.diff_add{background:#d4f8e8;color:#228b22;}
    span.diff_sub{background:#ffe0e0;color:#b22222;}
    span.diff_chg{background:#fff7cc;color:#b8860b;}
    td{white-space:pre-wrap;}
    .diff-table-header{font-size:1.08em;font-weight:700;letter-spacing:0.2px;}
    table.diff tr:hover td{background:#f6f8fa;transition:background 0.2s;}
    .normalization-notice{background:#e3f2fd;border:1px solid #2196f3;border-radius:8px;padding:16px;margin-bottom:24px;color:#1565c0;}
    </style>'''
    
    mathjax_script = '<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>\n<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'
    cer_definition = '''<div class="summary-section"><h2>Character Error Rate (CER) Definition</h2><p>$$\\mathrm{CER} = \\frac{\\text{Levenshtein distance}}{\\text{number of characters in ground truth}}$$<br>Insertions, deletions, and substitutions are counted as edit operations. Lower CER means higher similarity.</p><p><strong>Unnormalized CER:</strong> Uses original text with all characters, case, linebreaks, and formatting preserved.</p><p><strong>Normalized CER:</strong> Uses text normalized to ASCII letters (a-z), digits (0-9), lowercase, no linebreaks, and trimmed whitespace (academic standard).</p></div>'''

    for stem in common_stems:
        gt_df = load_gt_file(gt_xlsx_dir / f"{stem}.xlsx")
        llm_df = load_llm_file(llm_csv_dir / f"{stem}.csv")
        if gt_df.empty or llm_df.empty:
            continue
            
        # Original text for comparison (unnormalized)
        gt_text = "\n".join(gt_df['entry'].tolist())
        llm_text = "\n".join(llm_df['entry'].tolist())
        
        # Normalized text for normalized CER calculation
        gt_df_normalized = create_normalized_dataframe(gt_df)
        llm_df_normalized = create_normalized_dataframe(llm_df)
        gt_text_normalized = " ".join(gt_df_normalized['entry'].tolist())
        llm_text_normalized = " ".join(llm_df_normalized['entry'].tolist())
        
        # Calculate CERs
        cer_unnormalized = Levenshtein.normalized_distance(gt_text, llm_text)
        cer_normalized = Levenshtein.normalized_distance(gt_text_normalized, llm_text_normalized)
        
        # Get Levenshtein stats
        ins, del_, sub = compute_levenshtein_stats(gt_text, llm_text)
        
        year = extract_year_from_filename(stem)
        
        # Summary
        summary_rows.append({
            'file': stem,
            'year': year,
            'cer': cer_unnormalized,
            'cer_normalized': cer_normalized,
            'words_gt': len(gt_text.split()),
            'words_llm': len(llm_text.split()),
            'chars_gt': len(gt_text),
            'chars_llm': len(llm_text),
            'ins': ins,
            'del': del_,
            'sub': sub
        })
        
        # Create diff section
        diff_sections.append(make_side_by_side_diff(gt_text, llm_text, stem, year, cer_unnormalized))

    # Generate diff report
    if summary_rows:
        summary_table_html = '<div style="margin-top: 32px;">' + make_summary_table_html(summary_rows) + '</div>'
        cer_graph_html = make_interactive_cer_graph(summary_rows)
        
        # Calculate average CER by concatenating all files
        all_gt_text = ""
        all_llm_text = ""
        all_gt_text_normalized = ""
        all_llm_text_normalized = ""
        
        for stem in common_stems:
            gt_df = load_gt_file(gt_xlsx_dir / f"{stem}.xlsx")
            llm_df = load_llm_file(llm_csv_dir / f"{stem}.csv")
            if gt_df.empty or llm_df.empty:
                continue
                
            all_gt_text += "\n".join(gt_df['entry'].tolist()) + "\n"
            all_llm_text += "\n".join(llm_df['entry'].tolist()) + "\n"
            
            gt_df_normalized = create_normalized_dataframe(gt_df)
            llm_df_normalized = create_normalized_dataframe(llm_df)
            all_gt_text_normalized += " ".join(gt_df_normalized['entry'].tolist()) + " "
            all_llm_text_normalized += " ".join(llm_df_normalized['entry'].tolist()) + " "
        
        avg_cer_unnormalized = Levenshtein.normalized_distance(all_gt_text, all_llm_text)
        avg_cer_normalized = Levenshtein.normalized_distance(all_gt_text_normalized, all_llm_text_normalized)
        
        normalization_notice = '''
        <div class="normalization-notice">
            <strong>Text Processing Notice:</strong> This report shows results with original text including all characters, case, linebreaks, and formatting preserved.
        </div>
        '''
        
        # Add average CER section
        avg_cer_html = f'''
        <div class="summary-section">
            <h2>Average Character Error Rate (CER)</h2>
            <p><strong>Unnormalized CER:</strong> {avg_cer_unnormalized:.2%} (calculated by concatenating all files)</p>
            <p><strong>Normalized CER:</strong> {avg_cer_normalized:.2%} (calculated by concatenating all normalized files)</p>
            <p><em>Note: Average CER is computed by concatenating all files and calculating the overall CER, not by averaging individual file CERs.</em></p>
        </div>
        '''
        
        diff_legend_html = '''
        <div class="diff-legend" style="margin: 36px 0 24px 0; padding: 18px 24px; background: #f8f8fc; border-radius: 8px; border: 1px solid #e0e0e0; max-width: 700px;">
          <strong>Legend for Side-by-Side Comparison:</strong>
          <ul style="margin: 10px 0 0 20px; padding: 0; font-size: 1em;">
            <li><span style="background:#d4f8e8; color:#228b22; padding:2px 6px; border-radius:3px;">Insertion</span>: Text present in the LLM output but not in the ground truth.</li>
            <li><span style="background:#ffe0e0; color:#b22222; padding:2px 6px; border-radius:3px;">Deletion</span>: Text present in the ground truth but not in the LLM output.</li>
            <li><span style="background:#fff7cc; color:#b8860b; padding:2px 6px; border-radius:3px;">Substitution</span>: Text that differs between the ground truth and the LLM output.</li>
          </ul>
        </div>
        '''

        full_html = (
            '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
            '<title>Diff Report</title>'
            '{mathjax_script}'
            '{style}'
            '</head><body><div class="container">'
            '{normalization_notice}{cer_definition}{summary_table_html}{avg_cer_html}{cer_graph_html}{diff_legend_html}{diff_sections}'
            '</div></body></html>'
        ).format(
            mathjax_script=mathjax_script,
            style=style,
            normalization_notice=normalization_notice,
            cer_definition=cer_definition,
            summary_table_html=summary_table_html,
            avg_cer_html=avg_cer_html,
            cer_graph_html=cer_graph_html,
            diff_legend_html=diff_legend_html,
            diff_sections=''.join(diff_sections)
        )
        diff_report_path = output_dir / "diff_report.html"
        diff_report_path.write_text(full_html, encoding='utf-8')
        logging.info(f"Diff report saved to {diff_report_path}")

    # Calculate overall CER (use concatenated approach)
    overall_cer = 0
    if summary_rows:
        # Use the concatenated CER calculation
        overall_cer = avg_cer_unnormalized

    # --- Return Results for JSON Generation ---
    results = {
        'comparison_type': comparison_type,
        'overall_match_rate': round(overall_match_rate, 2),
        'character_error_rate': round(overall_cer * 100, 2),
        'total_gt_entries': total_gt_entries,
        'total_llm_entries': total_llm_entries,
        'total_gt_matched': total_gt_matched,
        'total_llm_matched': total_llm_matched,
        'fuzzy_threshold': fuzzy_threshold,
        'common_files_processed': len(common_stems),
        'files_with_results': [row['file'] for row in summary_rows]
    }
    
    return results 