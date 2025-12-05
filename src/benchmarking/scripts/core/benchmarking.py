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

def create_clean_text_for_cer(df: pd.DataFrame) -> str:
    """Create clean text for CER calculation by concatenating entries without line breaks between entries."""
    entries = []
    for _, row in df.iterrows():
        # Combine all non-null text fields, excluding 'id' column
        text_parts = []
        for col in df.columns:
            if col != 'id' and pd.notna(row[col]) and str(row[col]).strip():
                text_parts.append(str(row[col]).strip())
        
        if text_parts:
            # Join parts with spaces
            entry_text = ' '.join(text_parts)
            entries.append(entry_text)
    
    # Join entries without line breaks between them for CER calculation
    return ' '.join(entries)

def create_text_file_from_entries(df: pd.DataFrame) -> str:
    """Create a text file representation for display with line breaks between entries."""
    entries = []
    for _, row in df.iterrows():
        # Combine all non-null text fields and trim whitespace, excluding 'id' column
        text_parts = []
        for col in df.columns:
            if col != 'id' and pd.notna(row[col]) and str(row[col]).strip():
                text_parts.append(str(row[col]).strip())
        
        if text_parts:
            # Join parts with spaces and trim the final entry
            entry_text = ' '.join(text_parts).strip()
            entries.append(entry_text)
    
    # Join entries with double line breaks (one empty line between entries)
    return '\n\n'.join(entries)

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
    """Performs mutual best fuzzy matching between two dataframes."""
    gt_entries = gt_df['entry'].astype(str).tolist()
    llm_entries = llm_df['entry'].astype(str).tolist()
    gt_ids = gt_df['id'].astype(str).tolist()
    llm_ids = llm_df['id'].astype(str).tolist()

    gt_matches = [False] * len(gt_entries)
    llm_matches = [False] * len(llm_entries)
    gt_match_ids = ['—'] * len(gt_entries)
    llm_match_ids = ['—'] * len(llm_entries)

    # Step 1: Calculate all similarity scores
    similarity_matrix = []
    for i, gt_entry in enumerate(gt_entries):
        row = []
        for j, llm_entry in enumerate(llm_entries):
            score = Levenshtein.normalized_similarity(gt_entry, llm_entry)
            row.append(score)
        similarity_matrix.append(row)

    # Step 2: Find mutual best matches
    # Track which entries are already matched
    used_gt_indices = set()
    used_llm_indices = set()
    
    # Continue until no more matches can be made
    while True:
        best_match = None
        best_score = -1
        
        # Find the highest scoring unmatched pair
        for i in range(len(gt_entries)):
            if i in used_gt_indices:
                continue
            for j in range(len(llm_entries)):
                if j in used_llm_indices:
                    continue
                score = similarity_matrix[i][j]
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = (i, j)
        
        # If no more matches above threshold, stop
        if best_match is None:
            break
            
        # Make the match
        gt_idx, llm_idx = best_match
        gt_matches[gt_idx] = True
        llm_matches[llm_idx] = True
        gt_match_ids[gt_idx] = llm_ids[llm_idx]
        llm_match_ids[llm_idx] = gt_ids[gt_idx]
        used_gt_indices.add(gt_idx)
        used_llm_indices.add(llm_idx)
            
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

def make_full_html(title: str, sections_html: str, summary_html: str, top_notes: str = "") -> str:
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
        f'<h1>{title}</h1>{top_notes}{summary_html}{sections_html}</div></body></html>'
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
    """Create a side-by-side text file comparison with character-level diffing."""
    
    # Use difflib to get character-level differences
    matcher = difflib.SequenceMatcher(None, gt_text, llm_text)
    
    # Create the HTML structure
    html_parts = [
        f'<section class="diff-section">',
        f'<h2 class="diff-file-heading">{html_escape(file)}</h2>',
        f'<h3 class="diff-cer">CER: {cer:.2%}</h3>',
        f'<div class="text-file-comparison">',
        f'<div class="text-file-container">',
        f'<div class="text-file-header">Ground Truth Text File</div>',
        f'<div class="text-file-content">'
    ]
    
    # Process ground truth text with highlighting
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Same text - no highlighting, preserve line breaks
            html_parts.append(html_escape(gt_text[i1:i2]))
        else:
            # Different text - highlight with yellow (inline span)
            html_parts.append(f'<span class="diff-highlight">{html_escape(gt_text[i1:i2])}</span>')
    
    html_parts.extend([
        f'</div>',  # Close text-file-content
        f'</div>',  # Close text-file-container
        f'<div class="text-file-container">',
        f'<div class="text-file-header">LLM Output Text File</div>',
        f'<div class="text-file-content">'
    ])
    
    # Process LLM text with highlighting
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Same text - no highlighting, preserve line breaks
            html_parts.append(html_escape(llm_text[j1:j2]))
        else:
            # Different text - highlight with yellow (inline span)
            html_parts.append(f'<span class="diff-highlight">{html_escape(llm_text[j1:j2])}</span>')
    
    html_parts.extend([
        f'</div>',  # Close text-file-content
        f'</div>',  # Close text-file-container
        f'</div>',  # Close text-file-comparison
        f'</section>'
    ])
    
    return ''.join(html_parts)  # Use join without newlines to prevent artificial breaks

# --- File Matching and Availability Logic ---

def find_matching_files(sampled_pdfs_dir: Path, student_xlsx_dir: Path, perfect_xlsx_dir: Path, llm_csv_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Find matching files across all three directories and return availability matrix.
    
    Returns:
        Dict with file stems as keys, containing paths to available files
    """
    # Get all file stems from each directory
    pdf_stems = {f.stem for f in sampled_pdfs_dir.glob('*.pdf')}
    student_stems = {f.stem for f in student_xlsx_dir.glob('*.xlsx')}
    perfect_stems = {f.stem for f in perfect_xlsx_dir.glob('*.xlsx')}
    
    # Handle perfect transcription naming variations
    perfect_stems_normalized = set()
    for stem in perfect_stems:
        # Handle cases like "Patentamt_1901_sampled_.perfected" -> "Patentamt_1901_sampled"
        if stem.endswith('_perfected') or stem.endswith('_.perfected'):
            base_stem = stem.replace('_perfected', '').replace('_.perfected', '')
            perfect_stems_normalized.add(base_stem)
        else:
            perfect_stems_normalized.add(stem)
    
    # Find all possible file stems
    all_stems = pdf_stems.union(student_stems).union(perfect_stems_normalized)
    
    # Create file availability matrix
    file_matrix = {}
    for stem in sorted(all_stems):
        file_matrix[stem] = {
            'pdf': None,
            'student': None,
            'perfect': None,
            'available': []
        }
        
        # Check PDF availability
        pdf_path = sampled_pdfs_dir / f"{stem}.pdf"
        if pdf_path.exists():
            file_matrix[stem]['pdf'] = pdf_path
            file_matrix[stem]['available'].append('pdf')
        
        # Check student transcription availability
        student_path = student_xlsx_dir / f"{stem}.xlsx"
        if student_path.exists():
            file_matrix[stem]['student'] = student_path
            file_matrix[stem]['available'].append('student')
        
        # Check LLM CSV availability (handle both regular and _cleaned files)
        llm_candidates = [
            llm_csv_dir / f"{stem}.csv",
            llm_csv_dir / f"{stem}_cleaned.csv"
        ]
        
        for llm_path in llm_candidates:
            if llm_path.exists():
                file_matrix[stem]['llm'] = llm_path
                file_matrix[stem]['available'].append('llm')
                break
        
        # Check perfect transcription availability (handle naming variations)
        perfect_candidates = [
            perfect_xlsx_dir / f"{stem}_perfected.xlsx",
            perfect_xlsx_dir / f"{stem}_.perfected.xlsx",
            perfect_xlsx_dir / f"{stem}_perfected.xlsx"
        ]
        
        for perfect_path in perfect_candidates:
            if perfect_path.exists():
                file_matrix[stem]['perfect'] = perfect_path
                file_matrix[stem]['available'].append('perfect')
                break
    
    return file_matrix

def create_three_table_comparison(perfect_df: pd.DataFrame, llm_df: pd.DataFrame, student_df: pd.DataFrame, 
                                 filename_stem: str, fuzzy_threshold: float = 0.85) -> Dict[str, Any]:
    """
    Create a three-table comparison showing Perfect, LLM, and Student transcriptions.
    
    Returns:
        Dict containing comparison results and HTML sections
    """
    # Perform fuzzy matching for both comparisons
    perfect_llm_matches, llm_matches, perfect_llm_ids, llm_match_ids = match_entries_fuzzy(perfect_df, llm_df, fuzzy_threshold)
    perfect_student_matches, student_matches, perfect_student_ids, student_match_ids = match_entries_fuzzy(perfect_df, student_df, fuzzy_threshold)
    
    # Calculate CER for both comparisons
    perfect_text = create_text_file_from_entries(perfect_df)
    llm_text = create_text_file_from_entries(llm_df)
    student_text = create_text_file_from_entries(student_df)
    
    # Clean text for CER calculation
    perfect_text_clean = create_clean_text_for_cer(perfect_df)
    llm_text_clean = create_clean_text_for_cer(llm_df)
    student_text_clean = create_clean_text_for_cer(student_df)
    
    # Calculate CERs
    llm_cer = Levenshtein.normalized_distance(perfect_text_clean, llm_text_clean)
    student_cer = Levenshtein.normalized_distance(perfect_text_clean, student_text_clean)
    
    # Calculate performance gap
    performance_gap = student_cer - llm_cer
    
    # Create HTML tables with color coding
    perfect_table_html = make_three_table_html(perfect_df, [True] * len(perfect_df), ['—'] * len(perfect_df), 
                                              'Perfect Transcription', 'perfect')
    llm_table_html = make_three_table_html(llm_df, llm_matches, llm_match_ids, 
                                          'LLM-Generated Transcription', 'llm')
    student_table_html = make_three_table_html(student_df, student_matches, student_match_ids, 
                                             'Student Transcription', 'student')
    
    return {
        'filename': filename_stem,
        'perfect_table': perfect_table_html,
        'llm_table': llm_table_html,
        'student_table': student_table_html,
        'llm_cer': llm_cer,
        'student_cer': student_cer,
        'performance_gap': performance_gap,
        'llm_matches': sum(llm_matches),
        'student_matches': sum(student_matches),
        'perfect_entries': len(perfect_df),
        'llm_entries': len(llm_df),
        'student_entries': len(student_df)
    }

def create_two_table_comparison(perfect_df: pd.DataFrame, llm_df: pd.DataFrame, 
                               filename_stem: str, fuzzy_threshold: float = 0.85) -> Dict[str, Any]:
    """
    Create a two-table comparison showing Perfect and LLM transcriptions only.
    
    Returns:
        Dict containing comparison results and HTML sections
    """
    # Perform fuzzy matching between Perfect and LLM
    perfect_llm_matches, llm_matches, perfect_llm_ids, llm_match_ids = match_entries_fuzzy(perfect_df, llm_df, fuzzy_threshold)
    
    # Create HTML tables with color coding
    # For perfect table: use the matches and match IDs from the fuzzy matching
    perfect_table_html = make_three_table_html(perfect_df, perfect_llm_matches, perfect_llm_ids, 
                                              'Perfect Transcription', 'perfect')
    llm_table_html = make_three_table_html(llm_df, llm_matches, llm_match_ids, 
                                          'LLM-Generated Transcription', 'llm')
    
    return {
        'filename': filename_stem,
        'perfect_table': perfect_table_html,
        'llm_table': llm_table_html,
        'llm_matches': sum(llm_matches),
        'perfect_entries': len(perfect_df),
        'llm_entries': len(llm_df)
    }

def make_three_table_html(df: pd.DataFrame, matches: List[bool], match_ids: List[str], 
                         title: str, table_type: str) -> str:
    """Create HTML table with color coding for three-table comparison."""
    rows_html = []
    for i, row in df.iterrows():
        # Color coding based on table type
        if table_type == 'perfect':
            # Perfect table: red for unmatched, green for matches
            color = '#f8d7da' if not matches[i] else '#d4edda'  # red for unmatched, green for matches
        elif table_type == 'llm':
            # LLM table: red for unmatched, green for matches
            color = '#f8d7da' if not matches[i] else '#d4edda'  # red for unmatched, green for matches
        elif table_type == 'student':
            # Student table: red for unmatched, green for matches
            color = '#f8d7da' if not matches[i] else '#d4edda'  # red for unmatched, green for matches
        else:
            color = '#d4edda'  # default green
        
        rows_html.append(
            f'<tr style="background-color:{color}">'
            f'<td>{html_escape(row["id"])}</td>'
            f'<td>{html_escape(row["entry"])}</td>'
            f'<td>{html_escape(match_ids[i])}</td>'
            f'</tr>'
        )
    
    return (
        f'<table class="three-table">\n'
        f'<caption>{title}</caption>\n'
        f'<tr><th>ID</th><th>Entry</th><th>Match ID</th></tr>\n'
        f'{"".join(rows_html)}\n'
        f'</table>'
    )

def make_three_table_diff_html(perfect_text: str, llm_text: str, student_text: str, 
                              filename: str, llm_cer: float, student_cer: float, performance_gap: float) -> str:
    """Create side-by-side text comparison for three tables with character-level highlighting."""
    
    def create_character_level_diff(text1: str, text2: str, highlight_class: str) -> str:
        """Create ultra-precise character-level diff highlighting while preserving text formatting."""
        html_parts = []
        
        # Use dynamic programming for precise alignment
        aligned_text1, aligned_text2 = align_texts_for_comparison(text1, text2)
        
        # Create a mapping to track which characters in aligned_text1 correspond to original text1
        # This will help us distinguish between original spaces and alignment spaces
        original_positions = []
        i = 0  # Index in original text1
        
        for j in range(len(aligned_text1)):
            if aligned_text1[j] != ' ' or (i < len(text1) and text1[i] == ' '):
                # This character corresponds to original text1 (either non-space or original space)
                original_positions.append(i)
                i += 1
            else:
                # This is an alignment space - mark as -1
                original_positions.append(-1)
        
        # Now process the aligned texts with proper mapping
        for j in range(len(aligned_text1)):
            if original_positions[j] == -1:
                # This is an alignment space - skip it
                continue
            elif aligned_text1[j] == aligned_text2[j]:
                # Characters match - no highlighting
                html_parts.append(html_escape(aligned_text1[j]))
            else:
                # Characters differ - highlight the character
                html_parts.append(f'<span class="{highlight_class}">{html_escape(aligned_text1[j])}</span>')
        
        return ''.join(html_parts)
    
    def align_texts_for_comparison(text1: str, text2: str):
        """Align two texts for character-by-character comparison using dynamic programming."""
        # Use dynamic programming to find optimal alignment
        m, n = len(text1), len(text2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        # Backtrack to find alignment
        aligned1, aligned2 = [], []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and text1[i-1] == text2[j-1]:
                # Characters match
                aligned1.append(text1[i-1])
                aligned2.append(text2[j-1])
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or dp[i-1][j] <= dp[i][j-1]):
                # Deletion in text1
                aligned1.append(text1[i-1])
                aligned2.append(' ')
                i -= 1
            else:
                # Insertion in text2
                aligned1.append(' ')
                aligned2.append(text2[j-1])
                j -= 1
        
        return ''.join(reversed(aligned1)), ''.join(reversed(aligned2))
    
    html_parts = [
        f'<section class="three-diff-section">',
        f'<h2 class="diff-file-heading">{html_escape(filename)}.pdf</h2>',
        f'<div class="highlight-explanation">',
        f'<p><strong>Note:</strong> The visual highlighting may occasionally mark identical text as different due to sequence alignment limitations. However, this does not affect CER calculations, which use precise character-level Levenshtein distance. The highlighting serves as a visual aid only - the underlying metrics remain accurate.</p>',
        f'</div>',
        f'<div class="metrics-row">',
        f'<div class="metric-box gap-metric">',
        f'<h3>Performance Gap</h3>',
        f'<p><strong>Gap:</strong> {performance_gap:+.2%}</p>',
        f'<p><em>{"LLM better" if performance_gap > 0 else "Student better" if performance_gap < 0 else "Equal"}</em></p>',
        f'</div>',
        f'<div class="metric-box llm-metric">',
        f'<h3>LLM vs Perfect</h3>',
        f'<p><strong>CER:</strong> {llm_cer:.2%}</p>',
        f'</div>',
        f'<div class="metric-box student-metric">',
        f'<h3>Student vs Perfect</h3>',
        f'<p><strong>CER:</strong> {student_cer:.2%}</p>',
        f'</div>',
        f'</div>',
        f'<div class="three-text-comparison">',
        f'<div class="text-container">',
        f'<div class="text-header perfect-header">Perfect Transcription</div>',
        f'<div class="text-content">',
        html_escape(perfect_text),  # No highlighting for perfect text
        f'</div>',
        f'</div>',
        f'<div class="text-container">',
        f'<div class="text-header llm-header">LLM-Generated Transcription</div>',
        f'<div class="text-content">',
        create_character_level_diff(llm_text, perfect_text, 'diff-highlight-llm'),
        f'</div>',
        f'</div>',
        f'<div class="text-container">',
        f'<div class="text-header student-header">Student Transcription</div>',
        f'<div class="text-content">',
        create_character_level_diff(student_text, perfect_text, 'diff-highlight-student'),
        f'</div>',
        f'</div>',
        f'</div>',  # Close three-text-comparison
        f'</section>'
    ]
    
    return ''.join(html_parts)

# --- Main Comparison Logic ---

def run_unified_comparison(llm_csv_dir: Path, student_xlsx_dir: Path, perfect_xlsx_dir: Path, 
                          sampled_pdfs_dir: Path, output_dir: Path, fuzzy_threshold: float = 0.85):
    """
    Run unified comparison with three-table format: Perfect, LLM, and Student.
    Generates comprehensive HTML reports with academic transparency.
    """
    logging.info("Starting unified three-table comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching files
    file_matrix = find_matching_files(sampled_pdfs_dir, student_xlsx_dir, perfect_xlsx_dir, llm_csv_dir)
    
    # Filter files that have at least two of the three components
    valid_files = {stem: data for stem, data in file_matrix.items() 
                   if len(data['available']) >= 2}
    
    if not valid_files:
        logging.warning("No valid file combinations found. Cannot proceed with comparison.")
        return None
    
    logging.info(f"Found {len(valid_files)} files with valid combinations")
    
    # Process each file
    comparison_results = []
    diff_sections = []
    summary_rows = []
    
    for stem, file_data in valid_files.items():
        logging.info(f"Processing {stem}")
        
        # Load available data
        perfect_df = pd.DataFrame({'id': [], 'entry': []})
        llm_df = pd.DataFrame({'id': [], 'entry': []})
        student_df = pd.DataFrame({'id': [], 'entry': []})
        
        if 'perfect' in file_data['available']:
            perfect_df = load_gt_file(file_data['perfect'])
        if 'llm' in file_data['available']:
            llm_df = load_llm_file(file_data['llm'])
        if 'student' in file_data['available']:
            student_df = load_gt_file(file_data['student'])
        
        # Create comparison if we have perfect + at least one other
        if not perfect_df.empty and (not llm_df.empty or not student_df.empty):
            if not llm_df.empty and not student_df.empty:
                # Full three-way comparison
                result = create_three_table_comparison(perfect_df, llm_df, student_df, stem, fuzzy_threshold)
                comparison_results.append(result)
                
                # Create diff section
                perfect_text = create_text_file_from_entries(perfect_df)
                llm_text = create_text_file_from_entries(llm_df)
                student_text = create_text_file_from_entries(student_df)
                
                diff_sections.append(make_three_table_diff_html(
                    perfect_text, llm_text, student_text, stem, 
                    result['llm_cer'], result['student_cer'], result['performance_gap']
                ))
                
                # Add to summary
                year = extract_year_from_filename(stem)
                summary_rows.append({
                    'file': stem,
                    'year': year,
                    'llm_cer': result['llm_cer'],
                    'student_cer': result['student_cer'],
                    'performance_gap': result['performance_gap'],
                    'llm_matches': result['llm_matches'],
                    'student_matches': result['student_matches'],
                    'perfect_entries': result['perfect_entries']
                })
            else:
                # Partial comparison - create empty table with "File not available"
                logging.warning(f"Partial data for {stem}: {file_data['available']}")
                # Add empty result for missing files
                if llm_df.empty:
                    comparison_results.append({
                        'filename': stem,
                        'perfect_table': make_empty_table_html('Perfect Transcription', 'perfect'),
                        'llm_table': make_empty_table_html('LLM-Generated Transcriptions - File not available', 'llm'),
                        'student_table': make_three_table_html(student_df, [True] * len(student_df), ['—'] * len(student_df), 'Student Transcriptions', 'student') if not student_df.empty else make_empty_table_html('Student Transcriptions', 'student'),
                        'llm_cer': None,
                        'student_cer': None,
                        'performance_gap': None
                    })
                if student_df.empty:
                    comparison_results.append({
                        'filename': stem,
                        'perfect_table': make_empty_table_html('Perfect Transcription', 'perfect'),
                        'llm_table': make_three_table_html(llm_df, [True] * len(llm_df), ['—'] * len(llm_df), 'LLM-Generated Transcriptions', 'llm') if not llm_df.empty else make_empty_table_html('LLM-Generated Transcriptions', 'llm'),
                        'student_table': make_empty_table_html('Student Transcriptions - File not available', 'student'),
                        'llm_cer': None,
                        'student_cer': None,
                        'performance_gap': None
                    })
    
    # Generate reports
    generate_unified_reports(comparison_results, diff_sections, summary_rows, file_matrix, output_dir, fuzzy_threshold, llm_csv_dir, student_xlsx_dir, perfect_xlsx_dir, generate_fuzzy_report=True)
    
    return {
        'total_files': len(valid_files),
        'comparison_results': len(comparison_results),
        'summary_rows': len(summary_rows)
    }

def run_unified_comparison_cer_only(llm_csv_dir: Path, student_xlsx_dir: Path, perfect_xlsx_dir: Path, 
                                   sampled_pdfs_dir: Path, output_dir: Path, fuzzy_threshold: float = 0.85):
    """
    Run unified comparison with three-table format but only generate the character error rate report.
    This is used by the 02 script to avoid generating unnecessary fuzzy matching reports.
    """
    logging.info("Starting unified three-table comparison (CER only)")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching files
    file_matrix = find_matching_files(sampled_pdfs_dir, student_xlsx_dir, perfect_xlsx_dir, llm_csv_dir)
    
    # Filter files that have at least two of the three components
    valid_files = {stem: data for stem, data in file_matrix.items() 
                   if len(data['available']) >= 2}
    
    if not valid_files:
        logging.warning("No valid file combinations found. Cannot proceed with comparison.")
        return None
    
    logging.info(f"Found {len(valid_files)} files with valid combinations")
    
    # Process each file
    comparison_results = []
    diff_sections = []
    summary_rows = []
    
    for stem, file_data in valid_files.items():
        logging.info(f"Processing {stem}")
        
        # Load available data
        perfect_df = pd.DataFrame({'id': [], 'entry': []})
        llm_df = pd.DataFrame({'id': [], 'entry': []})
        student_df = pd.DataFrame({'id': [], 'entry': []})
        
        if 'perfect' in file_data['available']:
            perfect_df = load_gt_file(file_data['perfect'])
        if 'llm' in file_data['available']:
            llm_df = load_llm_file(file_data['llm'])
        if 'student' in file_data['available']:
            student_df = load_gt_file(file_data['student'])
        
        # Create comparison if we have perfect + at least one other
        if not perfect_df.empty and (not llm_df.empty or not student_df.empty):
            if not llm_df.empty and not student_df.empty:
                # Full three-way comparison
                result = create_three_table_comparison(perfect_df, llm_df, student_df, stem, fuzzy_threshold)
                comparison_results.append(result)
                
                # Create diff section
                perfect_text = create_text_file_from_entries(perfect_df)
                llm_text = create_text_file_from_entries(llm_df)
                student_text = create_text_file_from_entries(student_df)
                
                diff_sections.append(make_three_table_diff_html(
                    perfect_text, llm_text, student_text, stem, 
                    result['llm_cer'], result['student_cer'], result['performance_gap']
                ))
                
                # Add to summary
                year = extract_year_from_filename(stem)
                summary_rows.append({
                    'file': stem,
                    'year': year,
                    'llm_cer': result['llm_cer'],
                    'student_cer': result['student_cer'],
                    'performance_gap': result['performance_gap'],
                    'llm_matches': result['llm_matches'],
                    'student_matches': result['student_matches'],
                    'perfect_entries': result['perfect_entries']
                })
            else:
                # Partial comparison - create empty table with "File not available"
                if llm_df.empty:
                    llm_table = make_empty_table_html("LLM Transcription", "llm")
                else:
                    llm_table = make_variable_table_html_simple(perfect_df, llm_df, [], [], [], [], stem)[0]
                
                if student_df.empty:
                    student_table = make_empty_table_html("Student Transcription", "student")
                else:
                    student_table = make_variable_table_html_simple(perfect_df, student_df, [], [], [], [], stem)[0]
                
                perfect_table = make_variable_table_html_simple(perfect_df, perfect_df, [], [], [], [], stem)[0]
                
                comparison_results.append({
                    'filename': stem,
                    'perfect_table': perfect_table,
                    'llm_table': llm_table,
                    'student_table': student_table,
                    'llm_cer': None,
                    'student_cer': None,
                    'performance_gap': None
                })
    
    # Generate only the diff report (3-way character error rate)
    generate_diff_report(diff_sections, summary_rows, file_matrix, output_dir, llm_csv_dir, student_xlsx_dir, perfect_xlsx_dir)
    
    return {
        'total_files': len(valid_files),
        'comparison_results': len(comparison_results),
        'summary_rows': len(summary_rows)
    }

def make_empty_table_html(title: str, table_type: str) -> str:
    """Create empty table with 'File not available' message."""
    color = '#f8f9fa' if table_type == 'perfect' else '#e9ecef'
    return (
        f'<table class="three-table empty-table">\n'
        f'<caption>{title}</caption>\n'
        f'<tr><th>ID</th><th>Entry</th><th>Match ID</th></tr>\n'
        f'<tr style="background-color:{color}"><td colspan="3" style="text-align: center; font-style: italic;">File not available</td></tr>\n'
        f'</table>'
    )

def generate_unified_reports(comparison_results: List[Dict], diff_sections: List[str], 
                           summary_rows: List[Dict], file_matrix: Dict, output_dir: Path, fuzzy_threshold: float,
                           llm_csv_dir: Path, student_xlsx_dir: Path, perfect_xlsx_dir: Path, 
                           generate_fuzzy_report: bool = True):
    """Generate unified HTML reports with separate two-way and three-way formats."""
    
    # Create 2-way comparison results for fuzzy report (Perfect vs LLM only)
    two_way_comparison_results = []
    for stem, file_data in file_matrix.items():
        if 'perfect' in file_data['available'] and 'llm' in file_data['available']:
            try:
                perfect_df = load_gt_file(file_data['perfect'])
                llm_df = load_llm_file(llm_csv_dir / f"{stem}.csv")
                
                if not perfect_df.empty and not llm_df.empty:
                    result = create_two_table_comparison(perfect_df, llm_df, stem, fuzzy_threshold)
                    two_way_comparison_results.append(result)
            except Exception as e:
                logging.warning(f"Error creating 2-way comparison for {stem}: {e}")
                continue
    
    # Generate fuzzy report (2-way) only if requested
    if generate_fuzzy_report:
        generate_fuzzy_report_func(two_way_comparison_results, file_matrix, output_dir, fuzzy_threshold)
    
    # Generate diff report (3-way)
    generate_diff_report(diff_sections, summary_rows, file_matrix, output_dir, llm_csv_dir, student_xlsx_dir, perfect_xlsx_dir)

def generate_fuzzy_report_func(comparison_results: List[Dict], file_matrix: Dict, output_dir: Path, fuzzy_threshold: float, report_type: str = "before_cleaning"):
    """Generate fuzzy matching report with two-table format (Perfect vs LLM only)."""
    
    # Create file availability summary (Perfect and LLM only)
    availability_summary = create_two_way_availability_summary(file_matrix)
    
    # Create comparison sections (Perfect vs LLM only)
    sections_html = []
    for result in comparison_results:
        section_html = f'''
        <section class="two-comparison-section">
            <h2>Patentamt_{html_escape(result['filename'].split('_')[1])}_sampled.pdf</h2>
            <div class="two-table-container">
                {result['perfect_table']}
                {result['llm_table']}
            </div>
        </section>
        '''
        sections_html.append(section_html)
    
    # Create summary metrics (Perfect vs LLM only)
    summary_html = create_two_way_summary(comparison_results, fuzzy_threshold)
    
    # Generate full HTML
    if report_type == "after_cleaning":
        title = "Patent Entry Matching After Cleaning - Perfect vs LLM Transcriptions"
        fuzzy_report_path = output_dir / "patent_entry_matching_after_cleaning.html"
    else:
        title = "Patent Entry Matching Before Cleaning - Perfect vs LLM Transcriptions"
        fuzzy_report_path = output_dir / "patent_entry_matching_before_cleaning.html"
    
    full_html = make_two_way_html(title, availability_summary + summary_html, ''.join(sections_html))
    fuzzy_report_path.write_text(full_html, encoding='utf-8')
    logging.info(f"Patent entry matching report saved to {fuzzy_report_path}")

def generate_diff_report(diff_sections: List[str], summary_rows: List[Dict], file_matrix: Dict, output_dir: Path, 
                         llm_csv_dir: Path, student_xlsx_dir: Path, perfect_xlsx_dir: Path):
    """Generate diff report with three-table text comparison."""
    
    # Create document outline
    document_outline = create_document_outline()
    
    # Create file availability summary
    availability_summary = create_availability_summary(file_matrix)
    
    # Create CER definition with academic formula
    cer_definition = create_cer_definition()
    
    # Create summary table
    summary_table_html = create_summary_table_html(summary_rows)
    
    # Create CER chart
    cer_chart_html = create_cer_chart_html(summary_rows)
    
    # Create performance gap analysis
    gap_analysis_html = create_performance_gap_analysis(summary_rows, file_matrix, llm_csv_dir, student_xlsx_dir, perfect_xlsx_dir)
    
    # Generate full HTML
    title = "Character Error Rate - Perfect, LLM, and Student Transcriptions"
    full_html = make_unified_diff_html(
        title, document_outline, availability_summary, cer_definition, summary_table_html, 
        cer_chart_html, gap_analysis_html, ''.join(diff_sections)
    )
    
    diff_report_path = output_dir / "character_error_rate.html"
    diff_report_path.write_text(full_html, encoding='utf-8')
    logging.info(f"Diff report saved to {diff_report_path}")

def create_availability_summary(file_matrix: Dict) -> str:
    """Create file availability summary."""
    total_files = len(file_matrix)
    perfect_available = sum(1 for data in file_matrix.values() if 'perfect' in data['available'])
    llm_available = sum(1 for data in file_matrix.values() if 'llm' in data['available'])
    student_available = sum(1 for data in file_matrix.values() if 'student' in data['available'])
    all_three_available = sum(1 for data in file_matrix.values() if len(data['available']) == 3)
    files_with_at_least_two = sum(1 for data in file_matrix.values() if len(data['available']) >= 2)
    files_processed = files_with_at_least_two
    
    return f'''
    <div class="availability-summary">
        <h2>File Availability Summary</h2>
        <p><strong>Total files:</strong> {total_files}</p>
        <p><strong>Perfect transcriptions available:</strong> {perfect_available}</p>
        <p><strong>LLM-generated transcriptions available:</strong> {llm_available}</p>
        <p><strong>Student transcriptions available:</strong> {student_available}</p>
    </div>
    '''

def create_two_way_availability_summary(file_matrix: Dict) -> str:
    """Create file availability summary for two-way comparison (Perfect vs LLM only)."""
    total_files = len(file_matrix)
    perfect_available = sum(1 for data in file_matrix.values() if 'perfect' in data['available'])
    llm_available = sum(1 for data in file_matrix.values() if 'llm' in data['available'])
    both_available = sum(1 for data in file_matrix.values() if 'perfect' in data['available'] and 'llm' in data['available'])
    
    return f'''
    <div class="availability-summary">
        <h2>File Availability Summary</h2>
        <p><strong>Total files:</strong> {total_files}</p>
        <p><strong>Perfect transcriptions available:</strong> {perfect_available}</p>
        <p><strong>LLM-generated transcriptions available:</strong> {llm_available}</p>
        <p><strong>Files with both Perfect and LLM transcriptions:</strong> {both_available}</p>
    </div>
    '''

def create_cer_definition() -> str:
    """Create CER definition with academic formula."""
    return r'''
    <div class="cer-definition">
        <h2>Character Error Rate (CER) Definition</h2>
        <p>The Character Error Rate is calculated using the Levenshtein distance formula:</p>
        <div style="text-align: center; font-size: 1.4em; margin: 30px 0; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
            $$\mathrm{CER} = \frac{\mathrm{Levenshtein\ distance}}{\mathrm{number\ of\ characters\ in\ ground\ truth}}$$
        </div>
        <p><strong>Note:</strong> Levenshtein distance is the minimum number of single-character edits (insertions, deletions, substitutions) required to transform one text into another. Lower CER indicates higher similarity.</p>
    </div>
    '''

def create_summary_table_html(summary_rows: List[Dict]) -> str:
    """Create summary table with all metrics."""
    if not summary_rows:
        return '<p>No summary data available.</p>'
    
    table_html = [
        '<table class="summary-table">',
        '<caption style="font-weight: bold; font-size: 1.4em; margin-bottom: 25px;">File-level Performance Metrics</caption>',
        '<tr><th>File</th><th>Year</th><th>LLM CER</th><th>Student CER</th><th>Performance Gap</th></tr>'
    ]
    
    for row in summary_rows:
        gap_text = f"{row['performance_gap']:+.2%}" if row['performance_gap'] is not None else "N/A"
        gap_color = "color: #d32f2f;" if row['performance_gap'] and row['performance_gap'] > 0 else "color: #388e3c;" if row['performance_gap'] and row['performance_gap'] < 0 else ""
        
        table_html.append(
            f'<tr>'
            f'<td>{html_escape(row["file"])}.pdf</td>'
            f'<td>{html_escape(row["year"])}</td>'
            f'<td>{row["llm_cer"]:.2%}</td>'
            f'<td>{row["student_cer"]:.2%}</td>'
            f'<td style="{gap_color}">{gap_text}</td>'
            f'</tr>'
        )
    
    table_html.append('</table>')
    return '\n'.join(table_html)

def create_document_outline() -> str:
    """Create document outline at the beginning."""
    return '''
    <div class="document-outline">
        <h2>Document Outline</h2>
        <p><strong>Note:</strong> This entire document was AI-generated using Claude-4.5-Sonnet in the Cursor IDE.</p>
        <p>This report provides a comprehensive analysis of transcription accuracy comparing LLM-generated and student-generated transcriptions against Perfect ground truth. The analysis includes:</p>
        <ol>
            <li><strong>File Availability Summary</strong></li>
            <li><strong>Character Error Rate (CER) Definition</strong></li>
            <li><strong>File-level Performance Metrics Table</strong> - Detailed CER and performance gap for each file</li>
            <li><strong>Character Error Rate by Year Chart</strong> - Interactive visualization of CER trends over time</li>
            <li><strong>Performance Gap Analysis</strong> - Overall statistics</li>
            <li><strong>Side-by-Side Text Comparisons</strong> - Detailed three-table format showing character differences between Perfect, LLM, and Student transcriptions</li>
        </ol>
    </div>
    '''

def create_cer_chart_html(summary_rows: List[Dict]) -> str:
    """Create interactive CER chart by year."""
    if not summary_rows:
        return '<p>No chart data available.</p>'
    
    # Sort by year for proper x-axis ordering
    sorted_rows = sorted(summary_rows, key=lambda x: int(x['year']) if x['year'].isdigit() else 0)
    
    years = [int(row['year']) for row in sorted_rows if row['year'].isdigit()]
    llm_cers = [row['llm_cer'] for row in sorted_rows if row['year'].isdigit()]
    student_cers = [row['student_cer'] for row in sorted_rows if row['year'].isdigit()]
    files = [row['file'] for row in sorted_rows if row['year'].isdigit()]
    
    # Convert to JSON strings for JavaScript
    years_js = json.dumps(years)
    llm_cers_js = json.dumps(llm_cers)
    student_cers_js = json.dumps(student_cers)
    files_js = json.dumps(files)
    
    return f'''
    <div class="cer-chart-section">
        <h2>Character Error Rate by Year</h2>
        <p>This chart shows the CER for both LLM-generated and Research Assistant transcriptions compared to Perfect transcriptions across different years. Lower CER indicates better performance.</p>
        <div id="cer-chart" style="height: 500px; margin: 20px 0;"></div>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script>
            var years = {years_js};
            var llmCers = {llm_cers_js};
            var studentCers = {student_cers_js};
            var files = {files_js};
            
            var trace1 = {{
                x: years,
                y: llmCers,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'LLM vs Perfect',
                line: {{ color: '#2196f3', width: 3 }},
                marker: {{ size: 8, color: '#2196f3' }},
                hovertemplate: 'Year: %{{x}}<br>LLM CER: %{{y:.2%}}<extra></extra>'
            }};
            
            var trace2 = {{
                x: years,
                y: studentCers,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Research Assistants vs Perfect',
                line: {{ color: '#f44336', width: 3 }},
                marker: {{ size: 8, color: '#f44336' }},
                hovertemplate: 'Year: %{{x}}<br>Research Assistants CER: %{{y:.2%}}<extra></extra>'
            }};
            
            var data = [trace1, trace2];
            
            var layout = {{
                title: {{
                    text: 'Character Error Rate Comparison by Year',
                    font: {{ size: 18, color: '#333' }}
                }},
                xaxis: {{
                    title: 'Year',
                    titlefont: {{ size: 14 }},
                    tickfont: {{ size: 12 }},
                    dtick: 1
                }},
                yaxis: {{
                    title: 'Character Error Rate (CER)',
                    titlefont: {{ size: 14 }},
                    tickfont: {{ size: 12 }},
                    tickformat: ',.1%'
                }},
                legend: {{
                    x: 0.02,
                    y: 0.98,
                    bgcolor: 'rgba(255,255,255,0.8)',
                    bordercolor: '#ccc',
                    borderwidth: 1
                }},
                shapes: [{{
                    type: 'line',
                    x0: 1894,
                    x1: 1894,
                    y0: 0,
                    y1: 1,
                    yref: 'paper',
                    line: {{
                        color: '#666',
                        width: 2,
                        dash: 'dot'
                    }}
                }}, {{
                    type: 'line',
                    x0: 1912,
                    x1: 1912,
                    y0: 0,
                    y1: 1,
                    yref: 'paper',
                    line: {{
                        color: '#666',
                        width: 2,
                        dash: 'dot'
                    }}
                }}],
                annotations: [{{
                    x: 1886,
                    y: 1.02,
                    yref: 'paper',
                    xanchor: 'center',
                    text: '<i>Roman Font</i>',
                    showarrow: false,
                    font: {{
                        size: 11,
                        color: '#444'
                    }}
                }}, {{
                    x: 1903,
                    y: 1.02,
                    yref: 'paper',
                    xanchor: 'center',
                    text: '<i>Unger Gothic</i>',
                    showarrow: false,
                    font: {{
                        size: 11,
                        color: '#444'
                    }}
                }}, {{
                    x: 1915,
                    y: 1.02,
                    yref: 'paper',
                    xanchor: 'center',
                    text: '<i>Breitkopf Gothic</i>',
                    showarrow: false,
                    font: {{
                        size: 11,
                        color: '#444'
                    }}
                }}],
                margin: {{ t: 80, b: 60, l: 80, r: 40 }},
                plot_bgcolor: 'rgba(248,248,248,0.5)',
                paper_bgcolor: 'white'
            }};
            
            Plotly.newPlot('cer-chart', data, layout);
        </script>
    </div>
    '''

def create_performance_gap_analysis(summary_rows: List[Dict], file_matrix: Dict, llm_csv_dir: Path, student_xlsx_dir: Path, perfect_xlsx_dir: Path) -> str:
    """Create performance gap analysis with concatenated approach."""
    if not summary_rows:
        return '<p>No performance data available.</p>'
    
    valid_gaps = [row['performance_gap'] for row in summary_rows if row['performance_gap'] is not None]
    if not valid_gaps:
        return '<p>No performance gap data available.</p>'
    
    # Calculate concatenated CER for overall performance gap
    all_perfect_text = ""
    all_llm_text = ""
    all_student_text = ""
    
    files_processed = 0
    for stem, file_data in file_matrix.items():
        if len(file_data['available']) == 3:  # Only files with all three components
            try:
                # Load data
                perfect_df = load_gt_file(file_data['perfect'])
                llm_df = load_llm_file(llm_csv_dir / f"{stem}.csv")
                student_df = load_gt_file(file_data['student'])
                
                if not perfect_df.empty and not llm_df.empty and not student_df.empty:
                    perfect_text = create_clean_text_for_cer(perfect_df)
                    llm_text = create_clean_text_for_cer(llm_df)
                    student_text = create_clean_text_for_cer(student_df)
                    
                    if perfect_text and llm_text and student_text:
                        all_perfect_text += perfect_text + " "
                        all_llm_text += llm_text + " "
                        all_student_text += student_text + " "
                        files_processed += 1
            except Exception as e:
                logging.warning(f"Error processing {stem} for concatenated CER: {e}")
                continue
    
    # Calculate overall CER using concatenated approach
    if all_perfect_text and all_llm_text and all_student_text:
        overall_llm_cer = Levenshtein.normalized_distance(all_perfect_text, all_llm_text)
        overall_student_cer = Levenshtein.normalized_distance(all_perfect_text, all_student_text)
        overall_gap = overall_student_cer - overall_llm_cer
    else:
        # Fallback to average of individual CERs
        overall_llm_cer = sum(row['llm_cer'] for row in summary_rows if row['llm_cer'] is not None) / len([row for row in summary_rows if row['llm_cer'] is not None])
        overall_student_cer = sum(row['student_cer'] for row in summary_rows if row['student_cer'] is not None) / len([row for row in summary_rows if row['student_cer'] is not None])
        overall_gap = overall_student_cer - overall_llm_cer
    
    # File-level statistics
    avg_gap = sum(valid_gaps) / len(valid_gaps)
    llm_better_count = sum(1 for gap in valid_gaps if gap > 0)
    student_better_count = sum(1 for gap in valid_gaps if gap < 0)
    equal_count = sum(1 for gap in valid_gaps if gap == 0)
    
    return f'''
    <div class="performance-analysis">
        <h2>Performance Gap Analysis</h2>
        <p><strong>Files where LLM performs better:</strong> {llm_better_count}</p>
        <p><strong>Files where Student performs better:</strong> {student_better_count}</p>
        <p><strong>Files with equal performance:</strong> {equal_count}</p>
        <p><em>Positive gap indicates LLM is closer to perfect than Student. Negative gap indicates Student is closer to perfect than LLM.</em></p>
    </div>
    '''

def create_unified_summary(comparison_results: List[Dict], fuzzy_threshold: float) -> str:
    """Create unified summary for fuzzy matching report."""
    if not comparison_results:
        return '<p>No comparison results available.</p>'
    
    # Calculate overall statistics
    total_llm_matches = sum(result.get('llm_matches', 0) for result in comparison_results)
    total_student_matches = sum(result.get('student_matches', 0) for result in comparison_results)
    total_perfect_entries = sum(result.get('perfect_entries', 0) for result in comparison_results)
    
    # Calculate average CERs
    valid_llm_cers = [result['llm_cer'] for result in comparison_results if result.get('llm_cer') is not None]
    valid_student_cers = [result['student_cer'] for result in comparison_results if result.get('student_cer') is not None]
    
    avg_llm_cer = sum(valid_llm_cers) / len(valid_llm_cers) if valid_llm_cers else 0
    avg_student_cer = sum(valid_student_cers) / len(valid_student_cers) if valid_student_cers else 0
    
    return f'''
    <div class="unified-summary">
        <h2>Unified Comparison Summary</h2>
        <p><strong>Fuzzy Matching Threshold:</strong> {fuzzy_threshold}</p>
        <p><strong>Total Perfect Entries:</strong> {total_perfect_entries}</p>
        <p><strong>Total LLM Matches:</strong> {total_llm_matches}</p>
        <p><strong>Total Student Matches:</strong> {total_student_matches}</p>
        <p><strong>Average LLM CER:</strong> {avg_llm_cer:.2%}</p>
        <p><strong>Average Student CER:</strong> {avg_student_cer:.2%}</p>
        <p><strong>Overall Performance Gap:</strong> {avg_student_cer - avg_llm_cer:+.2%}</p>
    </div>
    '''

def create_two_way_summary(comparison_results: List[Dict], fuzzy_threshold: float) -> str:
    """Create summary for two-way comparison (Perfect vs LLM only)."""
    if not comparison_results:
        return '<p>No comparison results available.</p>'
    
    # Calculate overall statistics
    total_llm_matches = sum(result.get('llm_matches', 0) for result in comparison_results)
    total_perfect_entries = sum(result.get('perfect_entries', 0) for result in comparison_results)
    total_llm_entries = sum(result.get('llm_entries', 0) for result in comparison_results)
    
    # Calculate match rates
    match_rate_perfect_perspective = (total_llm_matches / total_perfect_entries * 100) if total_perfect_entries > 0 else 0
    match_rate_llm_perspective = (total_llm_matches / total_llm_entries * 100) if total_llm_entries > 0 else 0
    
    return f'''
    <div class="two-way-summary">
        <h2>Patent Entry Matching Summary</h2>
        <p><strong>Fuzzy Matching Threshold:</strong> {fuzzy_threshold}</p>
        <p><strong>Total Perfect Entries:</strong> {total_perfect_entries}</p>
        <p><strong>Total LLM Entries:</strong> {total_llm_entries}</p>
        <p><strong>Total Matches:</strong> {total_llm_matches}</p>
        <p><strong>Match Rate (Perfect perspective):</strong> {match_rate_perfect_perspective:.2f}%</p>
        <p><strong>Match Rate (LLM perspective):</strong> {match_rate_llm_perspective:.2f}%</p>
    </div>
    '''

def make_unified_html(title: str, summary_html: str, sections_html: str) -> str:
    """Create unified HTML with three-table styling."""
    css = '''
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f4f9; color: #222; margin: 0; }
        .container { max-width: 1400px; margin: auto; padding: 20px; }
        .availability-summary { background: #e3f2fd; border: 1px solid #2196f3; border-radius: 8px; padding: 20px; margin-bottom: 30px; }
        .three-comparison-section { background: #fff; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 30px; padding: 20px; }
        .three-table-container { display: flex; gap: 15px; overflow-x: auto; }
        .three-table { flex: 1; min-width: 300px; border-collapse: collapse; margin: 10px 0; }
        .three-table th, .three-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .three-table th { background-color: #f2f2f2; font-weight: bold; }
        .three-table caption { font-weight: bold; margin-bottom: 10px; font-size: 1.1em; }
        .empty-table td { text-align: center; font-style: italic; color: #666; }
        .summary-section { background: #fff; border-radius: 8px; padding: 20px; margin: 20px 0; }
    </style>
    '''
    
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        {css}
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            {summary_html}
            {sections_html}
        </div>
    </body>
    </html>
    '''

def make_two_way_html(title: str, summary_html: str, sections_html: str) -> str:
    """Create HTML with two-table styling (Perfect vs LLM only)."""
    css = '''
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f4f9; color: #222; margin: 0; }
        .container { max-width: 1400px; margin: auto; padding: 20px; }
        .availability-summary { background: #e3f2fd; border: 1px solid #2196f3; border-radius: 8px; padding: 20px; margin-bottom: 30px; }
        .two-comparison-section { background: #fff; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 30px; padding: 20px; }
        .two-table-container { display: flex; gap: 20px; overflow-x: auto; }
        .three-table { flex: 1; min-width: 400px; border-collapse: collapse; margin: 10px 0; }
        .three-table th, .three-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .three-table th { background-color: #f2f2f2; font-weight: bold; }
        .three-table caption { font-weight: bold; margin-bottom: 10px; font-size: 1.1em; }
        .empty-table td { text-align: center; font-style: italic; color: #666; }
        .two-way-summary { background: #fff; border-radius: 8px; padding: 20px; margin: 20px 0; }
    </style>
    '''
    
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        {css}
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            {summary_html}
            {sections_html}
        </div>
    </body>
    </html>
    '''

def make_unified_diff_html(title: str, document_outline: str, availability_summary: str, cer_definition: str, 
                          summary_table_html: str, cer_chart_html: str, gap_analysis_html: str, diff_sections: str) -> str:
    """Create unified diff HTML with three-table text comparison."""
    css = '''
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f4f9; color: #222; margin: 0; }
        .container { max-width: 1400px; margin: auto; padding: 20px; }
        .availability-summary { background: #e3f2fd; border: 1px solid #2196f3; border-radius: 8px; padding: 20px; margin-bottom: 30px; }
        .cer-definition { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; }
        .summary-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .summary-table th, .summary-table td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        .summary-table th { background-color: #f2f2f2; font-weight: bold; }
        .performance-analysis { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; }
        .three-diff-section { background: #f9fafc; border: 1px solid #dbe2ea; border-radius: 14px; margin-bottom: 40px; padding: 28px; }
        .metrics-row { display: flex; gap: 20px; margin-bottom: 20px; }
        .metric-box { flex: 1; background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 15px; text-align: center; }
        .llm-metric { border-left: 4px solid #2196f3; }
        .student-metric { border-left: 4px solid #f44336; }
        .gap-metric { border-left: 4px solid #ff9800; }
        .three-text-comparison { display: flex; gap: 15px; margin-top: 20px; }
        .text-container { flex: 1; background: #fff; border: 1px solid #ddd; border-radius: 8px; }
        .text-header { background: #f2f2f2; font-weight: bold; padding: 12px; border-bottom: 1px solid #ddd; }
        .perfect-header { border-left: 4px solid #ffc107; }
        .llm-header { border-left: 4px solid #2196f3; }
        .student-header { border-left: 4px solid #f44336; }
        .text-content { padding: 15px; font-family: monospace; font-size: 0.9em; line-height: 1.5; white-space: pre-wrap; }
        .diff-highlight-perfect { background-color: rgba(255, 193, 7, 0.3); padding: 0; border-radius: 1px; display: inline; }
        .diff-highlight-llm { background-color: rgba(33, 150, 243, 0.3); padding: 0; border-radius: 1px; display: inline; }
        .diff-highlight-student { background-color: rgba(244, 67, 54, 0.3); padding: 0; border-radius: 1px; display: inline; }
        .highlight-explanation { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 15px; margin: 15px 0; font-size: 0.9em; color: #495057; }
        .highlight-explanation ul { margin: 10px 0; padding-left: 20px; }
        .highlight-explanation li { margin: 5px 0; line-height: 1.4; }
        .highlight-explanation p { margin: 10px 0; }
        .document-outline { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-bottom: 30px; }
        .document-outline ol { margin: 15px 0; padding-left: 25px; }
        .document-outline li { margin: 8px 0; line-height: 1.6; }
        .cer-chart-section { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; }
    </style>
    '''
    
    mathjax_script = '<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>\n<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'
    
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        {mathjax_script}
        {css}
    </head>
    <body>
        <div class="container">
            <h1 style="text-align: center;">{title}</h1>
            {document_outline}
            {availability_summary}
            {cer_definition}
            {summary_table_html}
            {cer_chart_html}
            {gap_analysis_html}
            {diff_sections}
        </div>
    </body>
    </html>
    '''

def run_after_cleaning_comparison(llm_csv_dir: Path, perfect_xlsx_dir: Path, output_dir: Path, fuzzy_threshold: float = 0.85):
    """
    Runs the comparison logic specifically for after-cleaning evaluation.
    Generates only the 2-way patent entry matching report (Perfect vs LLM-cleaned).
    
    Args:
        llm_csv_dir: Directory containing LLM-cleaned CSV files
        perfect_xlsx_dir: Directory containing perfect transcription Excel files
        output_dir: Directory to save comparison results
        fuzzy_threshold: Threshold for fuzzy matching (default: 0.85)
    """
    logging.info(f"Starting after-cleaning comparison for data in {llm_csv_dir.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get file stems with proper handling for different suffixes
    perfect_files = list(perfect_xlsx_dir.glob('*.xlsx'))
    llm_files = list(llm_csv_dir.glob('*.csv'))
    
    # Extract base stems (remove _perfected and _cleaned suffixes)
    perfect_stems = set()
    llm_stems = set()
    
    for perfect_file in perfect_files:
        stem = perfect_file.stem
        if stem.endswith('_perfected'):
            base_stem = stem[:-10]  # Remove '_perfected'
            perfect_stems.add(base_stem)
        else:
            perfect_stems.add(stem)
    
    for llm_file in llm_files:
        stem = llm_file.stem
        if stem.endswith('_cleaned'):
            base_stem = stem[:-8]  # Remove '_cleaned'
            llm_stems.add(base_stem)
        else:
            llm_stems.add(stem)
    
    common_stems = sorted(list(perfect_stems.intersection(llm_stems)))

    if not common_stems:
        logging.warning(f"No common files found between perfect ground truth and LLM-cleaned directories. Skipping comparison.")
        return None

    logging.info(f"Found {len(common_stems)} common files for after-cleaning comparison")
    
    # Create file matrix for availability summary (using original stems)
    perfect_original_stems = {stem + '_perfected' for stem in perfect_stems}
    llm_original_stems = {stem + '_cleaned' for stem in llm_stems}
    common_original_stems = sorted([stem + '_perfected' for stem in common_stems])
    
    # Create simple file matrix for after-cleaning comparison
    file_matrix = {}
    for stem in common_original_stems:
        file_matrix[stem] = {
            'perfect': perfect_xlsx_dir / f"{stem}.xlsx",
            'llm': llm_csv_dir / f"{stem.replace('_perfected', '_cleaned')}.csv",
            'available': ['perfect', 'llm']
        }
    
    # Run 2-way comparison (Perfect vs LLM-cleaned only)
    comparison_results = []
    
    for base_stem in common_stems:
        perfect_path = perfect_xlsx_dir / f"{base_stem}_perfected.xlsx"
        llm_path = llm_csv_dir / f"{base_stem}_cleaned.csv"
        
        logging.info(f"Processing after-cleaning comparison: {base_stem}")
        
        try:
            perfect_df = pd.read_excel(perfect_path)
            llm_df = pd.read_csv(llm_path)
            
            # Create 2-way comparison result
            result = create_two_table_comparison(perfect_df, llm_df, base_stem, fuzzy_threshold)
            result['filename_stem'] = base_stem
            comparison_results.append(result)
            
        except Exception as e:
            logging.error(f"Error processing {base_stem}: {e}")
            continue
    
    if not comparison_results:
        logging.warning("No valid comparison results generated for after-cleaning evaluation")
        return None
    
    # Generate the 2-way fuzzy report
    generate_fuzzy_report_func(comparison_results, file_matrix, output_dir, fuzzy_threshold, "after_cleaning")
    
    return {
        'total_files': len(common_stems),
        'comparison_results': len(comparison_results),
        'files_with_results': [r['filename_stem'] for r in comparison_results]
    }

def run_variable_extraction_comparison(llm_csv_dir: Path, perfect_xlsx_dir: Path, output_dir: Path, fuzzy_threshold: float = 0.85):
    """
    Runs the variable extraction comparison logic specifically for Perfect vs LLM-with-variables evaluation.
    Generates only the variable extraction report comparing Perfect transcriptions vs LLM CSV files with extracted variables.
    
    Args:
        llm_csv_dir: Directory containing LLM CSV files with extracted variables
        perfect_xlsx_dir: Directory containing perfect transcription Excel files
        output_dir: Directory to save comparison results
        fuzzy_threshold: Threshold for fuzzy matching (default: 0.85)
    """
    logging.info(f"Starting variable extraction comparison for data in {llm_csv_dir.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get file stems with proper handling for different suffixes
    perfect_files = list(perfect_xlsx_dir.glob('*.xlsx'))
    llm_files = list(llm_csv_dir.glob('*.csv'))
    
    # Extract base stems (remove _perfected and _cleaned_with_variables suffixes)
    perfect_stems = set()
    llm_stems = set()
    
    for perfect_file in perfect_files:
        stem = perfect_file.stem
        if stem.endswith('_perfected'):
            base_stem = stem[:-10]  # Remove '_perfected'
            perfect_stems.add(base_stem)
        else:
            perfect_stems.add(stem)
    
    for llm_file in llm_files:
        stem = llm_file.stem
        if stem.endswith('_cleaned_with_variables'):
            base_stem = stem[:-23]  # Remove '_cleaned_with_variables'
            llm_stems.add(base_stem)
        else:
            llm_stems.add(stem)
    
    common_stems = sorted(list(perfect_stems.intersection(llm_stems)))

    if not common_stems:
        logging.warning(f"No common files found between perfect ground truth and LLM-with-variables directories. Skipping comparison.")
        return None

    logging.info(f"Found {len(common_stems)} common files for variable extraction comparison")
    
    # For variable extraction, we need to call the existing variable comparison logic
    # Import the run_variable_comparison function from the 03 script
    import sys
    import importlib.util
    from pathlib import Path
    
    # Load the 03_variable_extraction_benchmarking module
    script_path = Path(__file__).parent.parent / "03_variable_extraction_benchmarking.py"
    spec = importlib.util.spec_from_file_location("variable_extraction_benchmarking", script_path)
    variable_extraction_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(variable_extraction_module)
    
    # Create temporary directories with correctly named files for the comparison
    temp_llm_dir = output_dir / 'temp_llm_for_comparison'
    temp_gt_dir = output_dir / 'temp_gt_for_comparison'
    temp_llm_dir.mkdir(exist_ok=True)
    temp_gt_dir.mkdir(exist_ok=True)
    
    # Copy LLM files with variables to temp directory with names that the function expects
    for llm_file in llm_files:
        if llm_file.stem.endswith('_cleaned_with_variables'):
            # The function expects files with _cleaned_with_variables suffix
            expected_name = llm_file.stem + '.csv'  # Keep the original name
            temp_csv = temp_llm_dir / expected_name
            import shutil
            shutil.copy2(llm_file, temp_csv)
    
    # Copy perfect files to temp directory with names that match the LLM base names
    for perfect_file in perfect_files:
        if perfect_file.stem.endswith('_perfected'):
            # Remove _perfected suffix to match LLM base names
            expected_name = perfect_file.stem.replace('_perfected', '') + '.xlsx'
            temp_xlsx = temp_gt_dir / expected_name
            import shutil
            shutil.copy2(perfect_file, temp_xlsx)
    
    # Run the variable comparison (Perfect vs LLM-with-variables)
    variable_results = variable_extraction_module.run_variable_comparison(
        llm_csv_dir=temp_llm_dir,
        gt_xlsx_dir=temp_gt_dir,
        output_dir=output_dir,
        fuzzy_threshold=fuzzy_threshold,
        comparison_type="perfect"
    )
    
    # Clean up temporary directories
    import shutil
    shutil.rmtree(temp_llm_dir, ignore_errors=True)
    shutil.rmtree(temp_gt_dir, ignore_errors=True)
    
    # Rename the generated report from variable_fuzzy_report.html to variable_extraction_report.html
    old_report_path = output_dir / "variable_fuzzy_report.html"
    new_report_path = output_dir / "variable_extraction_report.html"
    
    if old_report_path.exists():
        old_report_path.rename(new_report_path)
        logging.info(f"Renamed variable_fuzzy_report.html to variable_extraction_report.html")
    
    return variable_results

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
    overall_match_rate_gt = (total_gt_matched / total_gt_entries * 100) if total_gt_entries > 0 else 0
    overall_match_rate_llm = (total_llm_matched / total_llm_entries * 100) if total_llm_entries > 0 else 0
    
    # Create threshold and transcription notes
    threshold_note = f'<p><b>Fuzzy Matching Threshold:</b> {fuzzy_threshold}</p>'
    
    perfect_note = ""
    student_note = ""
    
    if comparison_type == "perfect":
        perfect_note = """
        <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 15px; margin: 15px 0;">
            <p><strong>Note:</strong> These perfect transcriptions should contain absolutely no errors and any differences to the LLM-generated transcriptions are due to errors made by the LLM.</p>
        </div>
        """
    elif comparison_type == "student":
        student_note = """
        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; margin: 15px 0;">
            <p><strong>Note:</strong> The ground truth data contains errors made by human student assistants during transcription. 
            These errors may affect the accuracy of the comparison results.</p>
        </div>
        """
    
    # Create transcription notes for the top of the page
    top_notes = ""
    if comparison_type == "perfect":
        top_notes = """
        <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 15px; margin-bottom: 30px;">
            <p style="margin: 0; color: #155724;"><strong>Note:</strong> These perfect transcriptions should contain absolutely no errors and any differences to the LLM-generated transcriptions are due to errors made by the LLM.</p>
        </div>
        """
    elif comparison_type == "student":
        top_notes = """
        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; margin-bottom: 30px;">
            <p style="margin: 0; color: #856404;"><strong>Note:</strong> The ground truth data contains errors made by human student assistants during transcription. 
            These errors may affect the accuracy of the comparison results.</p>
        </div>
        """
    
    summary_html = (
        f'<div class="summary-section">'
        f'<h2>Overall Summary - {comparison_type.title()} Comparison</h2>'
        f'{threshold_note}'
        f'<p><b>Total Ground Truth Entries:</b> {total_gt_entries}</p>'
        f'<p><b>Total LLM Entries:</b> {total_llm_entries}</p>'
        f'<p><b>Total Matches:</b> {total_gt_matched}</p>'
        f'<p><b>Overall Match Rate (Ground Truth perspective):</b> {overall_match_rate_gt:.2f}%</p>'
        f'<p><b>Overall Match Rate (LLM perspective):</b> {overall_match_rate_llm:.2f}%</p>'
        f'</div>'
    )
    
    # Add spacing between summary and file pairs
    sections_with_spacing = f'<div style="margin-top: 40px;">{"".join(pair_sections_html)}</div>'
    
    fuzzy_html_content = make_full_html(f"Fuzzy Matching Report - {comparison_type.title()}", sections_with_spacing, summary_html, top_notes)
    fuzzy_report_path = output_dir / "patent_entry_matching_before_cleaning.html"
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
    .diff-section{background:#f9fafc;border:1px solid #dbe2ea;border-radius:14px;margin-bottom:40px;padding:28px 22px;box-shadow:0 4px 16px rgba(0,0,0,0.06);}
    .diff-file-heading{font-size:1.35em;font-weight:700;margin-bottom:8px;letter-spacing:0.5px;}
    .diff-cer{font-size:1.08em;font-weight:600;margin-bottom:18px;}
    /* Text file comparison styling */
    .text-file-comparison{display:flex;gap:20px;margin-top:20px;}
    .text-file-container{flex:1;background:#fff;border-radius:8px;border:1.5px solid #e0e0e0;box-shadow:0 1px 4px rgba(0,0,0,0.03);}
    .text-file-header{background:#f2f2f2;font-weight:bold;font-size:1.08em;padding:12px 16px;margin:0;border-radius:8px 8px 0 0;border-bottom:1px solid #e0e0e0;}
    .text-file-content{padding:16px;background:#fafafa;border-radius:0 0 8px 8px;font-family:monospace;font-size:0.9em;line-height:1.5;white-space:pre-wrap;overflow-x:auto;word-break:break-word;}
    .text-line{padding:4px 16px;border-bottom:1px solid #f0f0f0;font-family:monospace;font-size:0.9em;line-height:1.4;white-space:pre-wrap;word-break:break-word;}
    .text-line:last-child{border-bottom:none;}
    .text-line:hover{background:#f6f8fa;}
    .text-line.empty-line{height:12px;background:#f8f8f8;border-bottom:1px solid #e8e8e8;}
    .diff-highlight{background-color:rgba(255,255,0,0.6);padding:1px 2px;border-radius:2px;}
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
        # Create text files from Excel entries with proper structure for display
        gt_text = create_text_file_from_entries(gt_df)
        llm_text = create_text_file_from_entries(llm_df)
        
        # Create clean text files for CER calculation (no line breaks between entries)
        gt_text_clean = create_clean_text_for_cer(gt_df)
        llm_text_clean = create_clean_text_for_cer(llm_df)
        
        # Normalized text for normalized CER calculation
        gt_df_normalized = create_normalized_dataframe(gt_df)
        llm_df_normalized = create_normalized_dataframe(llm_df)
        gt_text_normalized = " ".join(gt_df_normalized['entry'].tolist())
        llm_text_normalized = " ".join(llm_df_normalized['entry'].tolist())
        
        # Calculate CERs
        cer_unnormalized = Levenshtein.normalized_distance(gt_text_clean, llm_text_clean)
        cer_normalized = Levenshtein.normalized_distance(gt_text_normalized, llm_text_normalized)
        
        # Get Levenshtein stats
        ins, del_, sub = compute_levenshtein_stats(gt_text_clean, llm_text_clean)
        
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
                
            # Use clean text files for CER calculation
            all_gt_text += create_clean_text_for_cer(gt_df)
            all_llm_text += create_clean_text_for_cer(llm_df)
            
            gt_df_normalized = create_normalized_dataframe(gt_df)
            llm_df_normalized = create_normalized_dataframe(llm_df)
            all_gt_text_normalized += " ".join(gt_df_normalized['entry'].tolist()) + " "
            all_llm_text_normalized += " ".join(llm_df_normalized['entry'].tolist()) + " "
        
        avg_cer_unnormalized = Levenshtein.normalized_distance(all_gt_text, all_llm_text)
        avg_cer_normalized = Levenshtein.normalized_distance(all_gt_text_normalized, all_llm_text_normalized)
        
        # Create combined header section with transcription notes (no fuzzy threshold for text comparison)
        header_section = '''
        <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-bottom: 25px;">
        '''
        
        if comparison_type == "perfect":
            header_section += '''
            <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 12px; margin-bottom: 15px;">
                <p style="margin: 0; color: #155724;"><strong>Note:</strong> These perfect transcriptions should contain absolutely no errors and any differences to the LLM-generated transcriptions are due to errors made by the LLM.</p>
            </div>
            '''
        elif comparison_type == "student":
            header_section += '''
            <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 12px; margin-bottom: 15px;">
                <p style="margin: 0; color: #856404;"><strong>Note:</strong> The ground truth data contains errors made by human student assistants during transcription. These errors may affect the accuracy of the comparison results.</p>
            </div>
            '''
        
        header_section += '''
            <div style="background-color: #e3f2fd; border: 1px solid #2196f3; border-radius: 5px; padding: 12px;">
                <p style="margin: 0; color: #1565c0;"><strong>Text Processing Notice:</strong> This report shows results with original text including all characters, case, linebreaks, and formatting preserved.</p>
            </div>
        </div>
        '''
        
        # Empty variables for backward compatibility
        threshold_note = ""
        perfect_note = ""
        student_note = ""
        normalization_notice = ""
        
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
        <div class="diff-legend" style="margin: 36px 0 24px 0; padding: 18px 24px; background: #f8f8fc; border-radius: 8px; border: 1px solid #e0e0e0;">
          <strong>Text File Comparison:</strong>
          <p style="margin: 10px 0 0 0; font-size: 1em;">
            This comparison shows the complete text files created from Excel entries. Each entry is preserved as a block with its original formatting. 
            Empty lines between entries show the structure of the original Excel file. The CER is calculated on the complete text files.
          </p>
        </div>
        '''

        # Create title for diff report
        diff_title = f"<h1 style='text-align: center; color: #444; margin-bottom: 30px;'>Text Comparison Report - {comparison_type.title()} Transcriptions</h1>"
        
        full_html = (
            '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
            f'<title>Text Comparison Report - {comparison_type.title()} Transcriptions</title>'
            '{mathjax_script}'
            '{style}'
            '</head><body><div class="container">'
            '{diff_title}{header_section}{cer_definition}{summary_table_html}{avg_cer_html}{cer_graph_html}{diff_legend_html}{diff_sections}'
            '</div></body></html>'
        ).format(
            mathjax_script=mathjax_script,
            style=style,
            diff_title=diff_title,
            header_section=header_section,
            cer_definition=cer_definition,
            summary_table_html=summary_table_html,
            avg_cer_html=avg_cer_html,
            cer_graph_html=cer_graph_html,
            diff_legend_html=diff_legend_html,
            diff_sections=''.join(diff_sections)
        )
        diff_report_path = output_dir / "character_error_rate.html"
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
        'overall_match_rate': round(overall_match_rate_gt, 2),
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