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
        '<caption>File-level CER and Edit Statistics</caption>',
        '<tr><th>File</th><th>Year</th><th>CER</th><th>Words (GT)</th><th>Words (LLM)</th><th>Chars (GT)</th><th>Chars (LLM)</th><th>Insertions</th><th>Deletions</th><th>Substitutions</th></tr>'
    ]
    for row in summary_rows:
        table.append(
            f'<tr>'
            f'<td>{html_escape(row["file"])}<br></td>'
            f'<td>{html_escape(row["year"])}<br></td>'
            f'<td>{row["cer"]:.2%}</td>'
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
    years = [row['year'] for row in summary_rows]
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
    xaxis: {{ title: 'Year', tickangle: -90 }},
    yaxis: {{ title: 'CER', tickformat: ',.0%' }},
    margin: {{ t: 40, b: 120 }}
}};
Plotly.newPlot('cer-graph', data, layout);
</script>
'''

def make_side_by_side_diff(gt_text, llm_text, file, year, cer):
    diff_html = difflib.HtmlDiff(wrapcolumn=80).make_table(
        gt_text.splitlines(),
        llm_text.splitlines(),
        fromdesc='Ground Truth',
        todesc='LLM Output',
        context=True,
        numlines=2
    )
    return f'<section class="diff-section"><h2>{html_escape(file)} ({year})</h2><h3>CER: {cer:.2%}</h3>{diff_html}</section>'

# --- Main Comparison Logic ---

def run_comparison(llm_csv_dir: Path, gt_xlsx_dir: Path, output_dir: Path, fuzzy_threshold: float = 0.85):
    """
    Runs the full comparison logic: fuzzy matching and diffing.
    Generates HTML reports and a JSON summary.
    """
    logging.info(f"Starting comparison for data in {llm_csv_dir.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_stems = get_file_stems(gt_xlsx_dir, 'xlsx')
    llm_stems = get_file_stems(llm_csv_dir, 'csv')
    common_stems = sorted(list(gt_stems.intersection(llm_stems)))

    if not common_stems:
        logging.warning("No common files found between ground truth and LLM directories. Aborting.")
        return

    # --- Fuzzy Matching ---
    pair_sections_html = []
    total_gt_entries, total_llm_entries = 0, 0
    total_gt_matched, total_llm_matched = 0, 0
    full_gt_text, full_llm_text = "", ""

    for stem in common_stems:
        logging.info(f"Processing pair: {stem}")
        gt_df = load_gt_file(gt_xlsx_dir / f"{stem}.xlsx")
        llm_df = load_llm_file(llm_csv_dir / f"{stem}.csv")

        if gt_df.empty or llm_df.empty:
            logging.warning(f"Skipping pair {stem} due to empty dataframe after loading.")
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
        f'<h2>Overall Summary</h2>'
        f'<p><b>Total GT Entries:</b> {total_gt_entries}</p>'
        f'<p><b>Total LLM Entries:</b> {total_llm_entries}</p>'
        f'<p><b>Total GT Matched:</b> {total_gt_matched}</p>'
        f'<p><b>Total LLM Matched:</b> {total_llm_matched}</p>'
        f'<p><b>Overall Match Rate (GT perspective):</b> {overall_match_rate:.2f}%</p>'
        f'</div>'
    )
    
    fuzzy_html_content = make_full_html("Fuzzy Matching Report", "".join(pair_sections_html), summary_html)
    fuzzy_report_path = output_dir / "fuzzy_report.html"
    fuzzy_report_path.write_text(fuzzy_html_content, encoding='utf-8')
    logging.info(f"Fuzzy report saved to {fuzzy_report_path}")

    # --- CER and Diffing ---
    summary_rows = []
    diff_sections = []
    cer_definition = '<div class="summary-section"><h2>Character Error Rate (CER) Definition</h2><p>CER = (Levenshtein distance) / (number of characters in ground truth). Insertions, deletions, and substitutions are counted as edit operations. Lower CER means higher similarity.</p></div>'

    for stem in common_stems:
        gt_df = load_gt_file(gt_xlsx_dir / f"{stem}.xlsx")
        llm_df = load_llm_file(llm_csv_dir / f"{stem}.csv")
        if gt_df.empty or llm_df.empty:
            continue
        gt_text = "\n".join(gt_df['entry'].tolist())
        llm_text = "\n".join(llm_df['entry'].tolist())
        cer = Levenshtein.normalized_distance(gt_text, llm_text)
        ins, del_, sub = compute_levenshtein_stats(gt_text, llm_text)
        year = extract_year_from_filename(stem)
        summary_rows.append({
            'file': stem,
            'year': year,
            'cer': cer,
            'words_gt': len(gt_text.split()),
            'words_llm': len(llm_text.split()),
            'chars_gt': len(gt_text),
            'chars_llm': len(llm_text),
            'ins': ins,
            'del': del_,
            'sub': sub
        })
        diff_sections.append(make_side_by_side_diff(gt_text, llm_text, stem, year, cer))

    summary_table_html = make_summary_table_html(summary_rows)
    cer_graph_html = make_interactive_cer_graph(summary_rows)
    full_html = (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f'<title>Diff Report</title>'
        f'<style>'
        f'body{{{{font-family:sans-serif;background:#f4f4f9;color:#333;}}}}'
        f'.container{{{{max-width:1200px;margin:auto;padding:20px;}}}}'
        f'.summary-table{{{{width:100%;border-collapse:collapse;margin-bottom:30px;}}}}'
        f'.summary-table th,.summary-table td{{{{border:1px solid #ddd;padding:8px;text-align:left;}}}}'
        f'.summary-table th{{{{background:#f2f2f2;}}}}'
        f'.diff-section{{{{background:#fff;border:1px solid #ddd;border-radius:8px;margin-bottom:20px;padding:15px;box-shadow:0 2px 4px rgba(0,0,0,0.05);}}}}'
        f'</style></head><body><div class="container">'
        f'{cer_definition}{summary_table_html}{cer_graph_html}{"".join(diff_sections)}'
        f'</div></body></html>'
    )
    diff_report_path = output_dir / "diff_report.html"
    diff_report_path.write_text(full_html, encoding='utf-8')
    logging.info(f"Diff report saved to {diff_report_path}")

    # --- JSON Results ---
    results = {
        'overall_match_rate': round(overall_match_rate, 2),
        'character_error_rate': round(cer * 100, 2),
        'total_gt_entries': total_gt_entries,
        'total_llm_entries': total_llm_entries,
        'total_gt_matched': total_gt_matched,
        'total_llm_matched': total_llm_matched,
        'fuzzy_threshold': fuzzy_threshold,
        'common_files_processed': len(common_stems)
    }
    
    results_path = output_dir / "results.json"
    with results_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    logging.info(f"JSON results saved to {results_path}") 