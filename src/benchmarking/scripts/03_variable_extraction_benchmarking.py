#!/usr/bin/env python3

import argparse
import logging
import json
import os
import pandas as pd
from pathlib import Path
import sys
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types
from rapidfuzz.distance import Levenshtein

# Core modules are now in the same directory
from core.benchmarking import run_comparison
from create_dashboard import create_dashboard

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define project root
project_root = Path(__file__).parent.parent.parent.parent
BENCHMARKING_ROOT = project_root / 'data' / 'benchmarking'
PROMPTS_DIR = project_root / 'src' / 'benchmarking' / 'prompts' / '03_variable_extraction'
ENV_PATH = project_root / 'config' / '.env'

# Load environment
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM config
MAX_OUTPUT_TOKENS = 8192
MAX_WORKERS = 20
MAX_RETRIES = 3

MODELS = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-lite']

# Variable fields to extract and compare
VARIABLE_FIELDS = ['patent_id', 'name', 'location', 'description', 'date']

# --- LLM Processing Functions ---

def load_prompt(prompt_name: str) -> str:
    """Load the variable extraction prompt."""
    prompt_path = PROMPTS_DIR / prompt_name
    return prompt_path.read_text(encoding="utf-8")

def parse_response(text: str) -> dict:
    """Parse JSON response from LLM."""
    # Strip away any backticks that might wrap the text
    candidate = re.sub(r"^`+|`+$", "", text.strip())

    try:
        # Attempt to parse the cleaned text as JSON
        return json.loads(candidate)
    except json.JSONDecodeError:
        logging.warning(f"Failed to parse JSON response: {candidate[:100]}...")
        # Return default structure with "NaN" values if parsing fails
        return {field: "NaN" for field in VARIABLE_FIELDS}

def call_llm(entry: str, prompt_template: str, model_name: str) -> dict:
    """Call the LLM to extract variables from an entry."""
    client = genai.Client(api_key=API_KEY)
    prompt = f"{prompt_template.strip()}\n{entry.strip()}"
    
    # Configure model-specific settings
    config_args = {
        "temperature": 0.0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_mime_type": "application/json",
    }
    
    # For gemini-2.5 models, set thinking_config
    if "2.5" in model_name:
        if "lite" in model_name:
            # For lite model: no thinking, keep original max_output_tokens
            config_args["thinking_config"] = types.ThinkingConfig(
                thinking_budget=0,
                include_thoughts=False
            )
        else:
            # For other 2.5 models: use thinking config
            if "pro" in model_name:
                thinking_budget = 32768
            else:
                thinking_budget = 24576
            config_args["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget,
                include_thoughts=True
            )
    
    config = types.GenerateContentConfig(**config_args)
    
    # Retry logic for API failures only
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt],
                config=config,
            )
            if not response or not response.text:
                logging.warning(f"Empty response from {model_name}")
                return {field: "NaN" for field in VARIABLE_FIELDS}
            return parse_response(response.text)
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if this is an API failure that should be retried
            is_api_failure = (
                "429" in error_msg or 
                "rate limit" in error_msg.lower() or 
                "resource exhausted" in error_msg.lower() or
                "timeout" in error_msg.lower() or
                "connection" in error_msg.lower() or
                "network" in error_msg.lower()
            )
            
            if is_api_failure and attempt < max_retries - 1:
                retry_after = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logging.warning(f"API failure for {model_name} (attempt {attempt + 1}/{max_retries}): {error_msg}. Waiting {retry_after}s...")
                time.sleep(retry_after)
                continue
            else:
                # Either not an API failure or max retries reached
                logging.error(f"LLM call failed for {model_name} (attempt {attempt + 1}/{max_retries}): {error_msg}")
                
                # Check for specific error types
                if "thinking" in error_msg.lower():
                    logging.error(f"Thinking budget configuration issue for {model_name}")
                elif "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                    logging.error(f"Model {model_name} not found or not available")
                
                return {field: "NaN" for field in VARIABLE_FIELDS}
    
    return {field: "NaN" for field in VARIABLE_FIELDS}

def process_llm(df, prompt_template, model_name):
    """Process all entries in a dataframe using LLM."""
    results = [None] * len(df)
    complete_failures = 0
    partial_failures = 0
    failed_rows = []  # Complete failures (exceptions)
    partial_failure_rows = []  # Partial failures (some variables NaN)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(call_llm, row["entry"], prompt_template, model_name): idx
            for idx, row in df.iterrows()
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                if result and all(result.get(field, "NaN") != "NaN" for field in VARIABLE_FIELDS):
                    results[idx] = result
                    logging.info(f"Row {idx+1}: Successfully extracted variables")
                else:
                    # Partial failure - some variables are NaN
                    partial_failures += 1
                    partial_failure_rows.append((idx, df.iloc[idx]["id"]))
                    results[idx] = result
                    logging.warning(f"Row {idx+1}: Some variables failed to extract")
            except Exception as e:
                # Complete failure - exception occurred
                results[idx] = {field: "NaN" for field in VARIABLE_FIELDS}
                complete_failures += 1
                failed_rows.append((idx, df.iloc[idx]["id"]))
                logging.error(f"Row {idx+1}: Exception during LLM processing: {e}")
    
    return results, complete_failures, failed_rows, partial_failures, partial_failure_rows

def process_single_csv(csv_path: Path, output_dir: Path, prompt_template: str, model_name: str) -> bool:
    """Process a single CSV file for variable extraction."""
    try:
        logging.info(f"Processing CSV: {csv_path.name}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        if "entry" not in df.columns:
            logging.error(f"Column 'entry' not found in {csv_path.name}")
            return False
        
        # Process with LLM
        results, complete_failures, failed_rows, partial_failures, partial_failure_rows = process_llm(df, prompt_template, model_name)
        
        # Add extracted variables to dataframe
        for field in VARIABLE_FIELDS:
            df[field] = [result.get(field, "NaN") for result in results]
        
        # Save processed CSV
        output_filename = f"{csv_path.stem}_with_variables.csv"
        output_path = output_dir / output_filename
        df.to_csv(output_path, index=False)
        
        # Create logs directory if it doesn't exist
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Save runtime summary to logs folder
        summary_filename = f"{csv_path.stem}_cleaned_with_variables.txt"
        summary_path = logs_dir / summary_filename
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Total rows processed: {len(df)}\n")
            f.write(f"Complete LLM failures (exceptions): {complete_failures}\n")
            f.write(f"Partial LLM failures (some variables NaN): {partial_failures}\n")
            f.write(f"Total failures: {complete_failures + partial_failures}\n\n")
            
            if failed_rows:
                f.write("Complete failures - rows with exceptions (id):\n")
                for _, id_val in failed_rows:
                    f.write(f"{id_val}\n")
                f.write("\n")
            
            if partial_failure_rows:
                f.write("Partial failures - rows with some variables NaN (id):\n")
                for _, id_val in partial_failure_rows:
                    f.write(f"{id_val}\n")
                f.write("\n")
        
        logging.info(f"Saved processed CSV to: {output_path}")
        logging.info(f"Saved runtime summary to: {summary_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error processing {csv_path.name}: {e}")
        return False

# --- Variable Comparison Functions ---

def load_gt_variables(gt_xlsx_path: Path) -> pd.DataFrame:
    """Load ground truth variables from Excel file with whitespace trimming and validation."""
    try:
        df = pd.read_excel(gt_xlsx_path)
        
        # Trim whitespace from all column names
        df.columns = df.columns.str.strip()
        
        # Check if required variable columns exist after trimming
        missing_variables = []
        for field in VARIABLE_FIELDS:
            if field not in df.columns:
                missing_variables.append(field)
                df[field] = "NaN"
            else:
                # Special handling for patent_id to remove float residuals
                if field == "patent_id":
                    # Convert to string, remove .0 if present, then trim
                    df[field] = df[field].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                else:
                    # Ensure all other variable fields are strings and trim whitespace
                    df[field] = df[field].astype(str).str.strip()
        
        # Store missing variables info for summary
        df.attrs['missing_variables'] = missing_variables
        
        return df
    except Exception as e:
        logging.error(f"Error loading GT file {gt_xlsx_path}: {e}")
        return pd.DataFrame()

def load_llm_variables(llm_csv_path: Path) -> pd.DataFrame:
    """Load LLM extracted variables from CSV file with whitespace trimming."""
    try:
        df = pd.read_csv(llm_csv_path)
        
        # Trim whitespace from all column names
        df.columns = df.columns.str.strip()
        
        # Check if variable columns exist, if not create them with NaN
        for field in VARIABLE_FIELDS:
            if field not in df.columns:
                df[field] = "NaN"
            else:
                # Special handling for patent_id to remove float residuals
                if field == "patent_id":
                    # Convert to string, remove .0 if present, then trim
                    df[field] = df[field].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                else:
                    # Ensure all other variable fields are strings and trim whitespace
                    df[field] = df[field].astype(str).str.strip()
        return df
    except Exception as e:
        logging.error(f"Error loading LLM file {llm_csv_path}: {e}")
        return pd.DataFrame()

def match_entries_fuzzy(gt_df: pd.DataFrame, llm_df: pd.DataFrame, threshold: float = 0.85):
    """Performs mutual best fuzzy matching between two dataframes based on entry field."""
    gt_entries = gt_df['entry'].astype(str).tolist()
    llm_entries = llm_df['entry'].astype(str).tolist()
    gt_ids = gt_df['id'].astype(str).tolist()
    llm_ids = llm_df['id'].astype(str).tolist()

    gt_matches = [False] * len(gt_entries)
    llm_matches = [False] * len(llm_entries)
    gt_match_ids = ['—'] * len(gt_entries)
    llm_match_ids = ['—'] * len(llm_entries)

    # Calculate all similarity scores
    similarity_matrix = []
    for i, gt_entry in enumerate(gt_entries):
        row = []
        for j, llm_entry in enumerate(llm_entries):
            score = Levenshtein.normalized_similarity(gt_entry, llm_entry)
            row.append(score)
        similarity_matrix.append(row)

    # Find mutual best matches
    used_gt_indices = set()
    used_llm_indices = set()
    
    while True:
        best_match = None
        best_score = -1
        
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
        
        if best_match is None:
            break
            
        gt_idx, llm_idx = best_match
        gt_matches[gt_idx] = True
        llm_matches[llm_idx] = True
        gt_match_ids[gt_idx] = llm_ids[llm_idx]
        llm_match_ids[llm_idx] = gt_ids[gt_idx]
        used_gt_indices.add(gt_idx)
        used_llm_indices.add(llm_idx)
            
    return gt_matches, llm_matches, gt_match_ids, llm_match_ids

def compare_variables(gt_value: str, llm_value: str, threshold: float = 0.85) -> bool:
    """Compare two variable values using fuzzy matching."""
    if pd.isna(gt_value) or pd.isna(llm_value):
        return False
    
    # Ensure both values are strings and handle any conversion errors
    try:
        gt_str = str(gt_value).strip()
        llm_str = str(llm_value).strip()
    except (TypeError, ValueError):
        return False
    
    if gt_str == "" or llm_str == "":
        return False
    
    # Special handling for patent_id: remove .0 from LLM value if present
    if gt_str.isdigit() and llm_str.endswith('.0'):
        llm_str = llm_str[:-2]  # Remove '.0' suffix
    
    similarity = Levenshtein.normalized_similarity(gt_str, llm_str)
    return similarity >= threshold



def calculate_global_threshold_sensitivity(all_gt_dfs: list, all_llm_dfs: list, all_matched_pairs: list) -> dict:
    """Calculate match rates for different thresholds across all files combined."""
    if not all_matched_pairs:
        return {}
    
    # Calculate for each threshold
    thresholds = [round(t * 0.1, 1) for t in range(11)]  # 0.0, 0.1, ..., 1.0
    results = {}
    
    for threshold in thresholds:
        total_cells = len(all_matched_pairs) * len(VARIABLE_FIELDS)
        matched_cells = 0
        variable_stats = {field: {'total': 0, 'matched': 0} for field in VARIABLE_FIELDS}
        
        for file_idx, gt_idx, llm_idx in all_matched_pairs:
            gt_df = all_gt_dfs[file_idx]
            llm_df = all_llm_dfs[file_idx]
            
            for field in VARIABLE_FIELDS:
                gt_value = str(gt_df.iloc[gt_idx].get(field, "NaN"))
                llm_value = str(llm_df.iloc[llm_idx].get(field, "NaN"))
                
                is_match = compare_variables(gt_value, llm_value, threshold)
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
        
        results[threshold] = {
            'overall_rate': overall_rate,
            'variable_rates': variable_rates
        }
    
    return results

def create_global_threshold_table(threshold_sensitivity: dict) -> str:
    """Create HTML table for global threshold sensitivity analysis."""
    if not threshold_sensitivity:
        return ""
    
    # Create threshold sensitivity table
    threshold_table_rows = []
    for threshold in sorted(threshold_sensitivity.keys()):
        data = threshold_sensitivity[threshold]
        row_cells = [f'<td>{threshold}</td>']
        row_cells.append(f'<td>{data["overall_rate"]:.2f}%</td>')
        for field in VARIABLE_FIELDS:
            row_cells.append(f'<td>{data["variable_rates"][field]:.2f}%</td>')
        threshold_table_rows.append(f"<tr>{''.join(row_cells)}</tr>")
    
    threshold_header = ''.join([f'<th>{field}</th>' for field in VARIABLE_FIELDS])
    threshold_table_html = f"""
    <div class="global-threshold-analysis">
        <h2>Global Threshold Sensitivity Analysis</h2>
        <p>This table shows how match rates change across all files for different similarity thresholds.</p>
        <table class="benchmark-table">
            <tr><th>Threshold</th><th>Overall</th>{threshold_header}</tr>
            {''.join(threshold_table_rows)}
        </table>
    </div>
    """
    
    return threshold_table_html

def make_variable_table_html_simple(gt_df: pd.DataFrame, llm_df: pd.DataFrame, gt_matches: list, 
                                   llm_matches: list, gt_match_ids: list, llm_match_ids: list, 
                                   filename_stem: str) -> tuple:
    """Create HTML table for variable comparison (simplified version without individual threshold analysis)."""
    # Get matched pairs
    matched_pairs = []
    for i, gt_matched in enumerate(gt_matches):
        if gt_matched:
            llm_idx = int(gt_match_ids[i]) - 1  # Convert back to 0-based index
            if llm_idx < len(llm_df):
                matched_pairs.append((i, llm_idx))
    
    if not matched_pairs:
        return f"<p>No matches found for {filename_stem}</p>", "", {}
    
    # Calculate variable-level statistics for current threshold (0.85)
    total_cells = len(matched_pairs) * len(VARIABLE_FIELDS)
    matched_cells = 0
    variable_stats = {field: {'total': 0, 'matched': 0} for field in VARIABLE_FIELDS}
    
    # Build table rows
    table_rows = []
    for gt_idx, llm_idx in matched_pairs:
        row_cells = []
        for field in VARIABLE_FIELDS:
            gt_value = str(gt_df.iloc[gt_idx].get(field, "NaN"))
            llm_value = str(llm_df.iloc[llm_idx].get(field, "NaN"))
            
            is_match = compare_variables(gt_value, llm_value)
            variable_stats[field]['total'] += 1
            if is_match:
                variable_stats[field]['matched'] += 1
                matched_cells += 1
                bg_color = "#d4edda"  # Green
            else:
                bg_color = "#f8d7da"  # Red
            
            # Ensure safe HTML display - just show values separated by /
            safe_gt_value = str(gt_value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
            safe_llm_value = str(llm_value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
            cell_content = f"{safe_gt_value} / {safe_llm_value}"
            row_cells.append(f'<td style="background-color:{bg_color}">{cell_content}</td>')
        
        table_rows.append(f"<tr>{''.join(row_cells)}</tr>")
    
    # Calculate statistics
    overall_match_rate = (matched_cells / total_cells * 100) if total_cells > 0 else 0
    variable_match_rates = {}
    for field in VARIABLE_FIELDS:
        stats = variable_stats[field]
        rate = (stats['matched'] / stats['total'] * 100) if stats['total'] > 0 else 0
        variable_match_rates[field] = rate
    
    # Create metrics HTML
    metrics_html = f"""
    <div class="metrics">
        <b>Total Cells:</b> {total_cells} &nbsp; 
        <b>Matched Cells:</b> {matched_cells} &nbsp; 
        <b>Overall Match Rate:</b> {overall_match_rate:.2f}%<br>
        <b>Variable Match Rates:</b> 
        {', '.join([f'{field}: {rate:.2f}%' for field, rate in variable_match_rates.items()])}
    </div>
    """
    
    # Create table HTML with legend at the beginning
    header_cells = ''.join([f'<th>{field}</th>' for field in VARIABLE_FIELDS])
    table_html = f"""
    <div class="table-legend">
        <p><strong>Legend:</strong> Values in each cell show "Ground Truth / LLM Generated"</p>
    </div>
    <table class="benchmark-table">
        <caption>Variable Comparison - {filename_stem}</caption>
        <tr>{header_cells}</tr>
        {''.join(table_rows)}
    </table>
    """
    
    return metrics_html, table_html, {
        'total_cells': total_cells,
        'matched_cells': matched_cells,
        'overall_match_rate': overall_match_rate,
        'variable_match_rates': variable_match_rates
    }

def run_variable_comparison(llm_csv_dir: Path, gt_xlsx_dir: Path, output_dir: Path, 
                          fuzzy_threshold: float = 0.85, comparison_type: str = "perfect"):
    """Run variable-specific comparison between LLM and ground truth."""
    logging.info(f"Starting {comparison_type} variable comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_stems = {f.stem for f in gt_xlsx_dir.glob('*.xlsx')}
    logging.info(f"Found {len(gt_stems)} ground truth files: {sorted(gt_stems)}")
    
    # Extract base stems from LLM files (remove '_cleaned_with_variables' suffix)
    llm_stems = set()
    for f in llm_csv_dir.glob('*_with_variables.csv'):
        stem = f.stem
        if stem.endswith('_cleaned_with_variables'):
            base_stem = stem.replace('_cleaned_with_variables', '')
            llm_stems.add(base_stem)
        else:
            llm_stems.add(stem)
    logging.info(f"Found {len(llm_stems)} LLM files: {sorted(llm_stems)}")
    
    common_stems = sorted(list(gt_stems.intersection(llm_stems)))
    logging.info(f"Found {len(common_stems)} common files: {common_stems}")

    if not common_stems:
        logging.warning(f"No common files found for {comparison_type} comparison")
        return None

    # Collect all matched pairs across all files for threshold sensitivity analysis
    all_matched_pairs = []
    all_gt_dfs = []
    all_llm_dfs = []
    
    # First pass: collect all data and perform fuzzy matching
    for stem in common_stems:
        logging.info(f"Processing {comparison_type} variable pair: {stem}")
        
        gt_df = load_gt_variables(gt_xlsx_dir / f"{stem}.xlsx")
        # Find the corresponding LLM file with the correct suffix
        llm_file_pattern = f"{stem}_cleaned_with_variables.csv"
        llm_files = list(llm_csv_dir.glob(llm_file_pattern))
        if not llm_files:
            logging.warning(f"No LLM file found for pattern: {llm_file_pattern}")
            continue
        llm_df = load_llm_variables(llm_files[0])
        
        if gt_df.empty or llm_df.empty:
            continue
        
        # Track failed LLM extractions (rows where all variables are NaN)
        failed_rows = []
        for idx, row in llm_df.iterrows():
            all_nan = all(str(row.get(field, "NaN")).strip() == "NaN" for field in VARIABLE_FIELDS)
            if all_nan:
                failed_rows.append(row.get('id', idx + 1))
        

        
        # Perform fuzzy matching on entry field
        gt_matches, llm_matches, gt_match_ids, llm_match_ids = match_entries_fuzzy(
            gt_df, llm_df, fuzzy_threshold
        )
        
        # Collect matched pairs for global threshold analysis
        for i, gt_matched in enumerate(gt_matches):
            if gt_matched:
                llm_idx = int(gt_match_ids[i]) - 1  # Convert back to 0-based index
                if llm_idx < len(llm_df):
                    all_matched_pairs.append((len(all_gt_dfs), i, llm_idx))
        
        all_gt_dfs.append(gt_df)
        all_llm_dfs.append(llm_df)
    
    # Calculate global threshold sensitivity across all files
    global_threshold_sensitivity = calculate_global_threshold_sensitivity(
        all_gt_dfs, all_llm_dfs, all_matched_pairs
    )
    
    # Create global threshold sensitivity table
    global_threshold_table_html = create_global_threshold_table(global_threshold_sensitivity)
    
    # Aggregate statistics
    total_cells = 0
    total_matched_cells = 0
    total_variable_stats = {field: {'total': 0, 'matched': 0} for field in VARIABLE_FIELDS}
    file_sections = []
    
    # Second pass: create individual file sections
    for i, stem in enumerate(common_stems):
        gt_df = all_gt_dfs[i]
        llm_df = all_llm_dfs[i]
        
        # Perform fuzzy matching on entry field
        gt_matches, llm_matches, gt_match_ids, llm_match_ids = match_entries_fuzzy(
            gt_df, llm_df, fuzzy_threshold
        )
        
        # Create variable comparison table (without individual threshold analysis)
        metrics_html, table_html, stats = make_variable_table_html_simple(
            gt_df, llm_df, gt_matches, llm_matches, gt_match_ids, llm_match_ids, stem
        )
        
        # Aggregate statistics
        total_cells += stats['total_cells']
        total_matched_cells += stats['matched_cells']
        for field in VARIABLE_FIELDS:
            total_variable_stats[field]['total'] += stats['variable_match_rates'][field] * stats['total_cells'] / 100
            total_variable_stats[field]['matched'] += stats['variable_match_rates'][field] * stats['total_cells'] / 100 * stats['variable_match_rates'][field] / 100
        
        # Create file section
        file_section = f"""
        <section class="pair-section">
            <h2>File: <span class="filename">{stem}</span></h2>
            {metrics_html}
            <div class="table-container">
                {table_html}
            </div>
        </section>
        """
        file_sections.append(file_section)
    
    # Calculate overall statistics
    overall_match_rate = (total_matched_cells / total_cells * 100) if total_cells > 0 else 0
    overall_variable_rates = {}
    for field in VARIABLE_FIELDS:
        stats = total_variable_stats[field]
        rate = (stats['matched'] / stats['total'] * 100) if stats['total'] > 0 else 0
        overall_variable_rates[field] = rate
    
    # Create overall summary
    threshold_note = f"<p><b>Fuzzy Matching Threshold:</b> {fuzzy_threshold}</p>"
    
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
    
    summary_html = f"""
    <div class="summary-section">
        <h2>Overall Variable Extraction Summary - {comparison_type.title()}</h2>
        {threshold_note}
        <p><b>Total Cells:</b> {total_cells}</p>
        <p><b>Total Matched Cells:</b> {total_matched_cells}</p>
        <p><b>Overall Match Rate:</b> {overall_match_rate:.2f}%</p>
        <p><b>Variable Match Rates:</b></p>
        <ul>
            {''.join([f'<li><b>{field}:</b> {rate:.2f}%</li>' for field, rate in overall_variable_rates.items()])}
        </ul>
    </div>
    """
    
    # Add spacing between summary and file sections
    sections_with_spacing = f'<div style="margin-top: 40px;">{"".join(file_sections)}</div>'
    
    # Generate HTML report with global threshold table at the top
    html_content = make_full_html(
        f"Variable Extraction Report - {comparison_type.title()}",
        global_threshold_table_html,
        sections_with_spacing,
        summary_html,
        top_notes
    )
    
    report_path = output_dir / "variable_fuzzy_report.html"
    report_path.write_text(html_content, encoding='utf-8')
    logging.info(f"Variable report saved to {report_path}")
    
    # Return results for JSON summary
    return {
        'total_cells': total_cells,
        'matched_cells': total_matched_cells,
        'overall_match_rate': overall_match_rate,
        'variable_match_rates': overall_variable_rates,
        'files_processed': len(common_stems)
    }

def make_full_html(title: str, global_threshold_html: str, sections_html: str, summary_html: str, top_notes: str = "") -> str:
    """Create full HTML report."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
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
        .table-container {{ margin-top: 20px; }}
        .table-legend {{ margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff; }}
        .table-legend p {{ margin: 0; font-weight: 500; color: #495057; }}
        .benchmark-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        .benchmark-table th, .benchmark-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .benchmark-table th {{ background-color: #f2f2f2; font-weight: bold; }}
        .benchmark-table caption {{ font-weight: bold; margin-bottom: 10px; font-size: 1.2em; color: #2c3e50; }}
        .filename {{ font-family: monospace; background: #eee; padding: 2px 5px; border-radius: 4px; }}
        .summary-section {{ margin-top: 30px; padding: 20px; background: #fff; border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        {top_notes}
        {global_threshold_html}
        {summary_html}
        {sections_html}
    </div>
</body>
</html>"""

# --- Main Benchmarking Function ---

def run_single_benchmark(dataset_cleaning_model: str, dataset_cleaning_prompt: str, model: str, prompt: str):
    """
    Executes the full benchmarking pipeline for variable extraction for a single model and prompt combination.
    """
    logging.info(f"--- Starting variable extraction benchmark ---")
    logging.info(f"Input: model=[{dataset_cleaning_model}] prompt=[{dataset_cleaning_prompt}]")
    logging.info(f"Processing: model=[{model}] prompt=[{prompt}]")

    # Load the variable extraction prompt
    try:
        prompt_template = load_prompt(prompt)
    except Exception as e:
        logging.error(f"Failed to load prompt file {prompt}: {e}")
        return

    # Define input and output directories
    dataset_cleaning_prompt_stem = Path(dataset_cleaning_prompt).stem
    
    # Check if the prerequisite dataset cleaning step has been completed
    base_cleaning_dir = BENCHMARKING_ROOT / 'results' / '02_dataset_cleaning' / dataset_cleaning_model / dataset_cleaning_prompt_stem
    if not base_cleaning_dir.exists():
        logging.error(f"Prerequisite dataset cleaning results not found: {base_cleaning_dir}")
        logging.error(f"Please run the dataset cleaning benchmark first for model: {dataset_cleaning_model}, prompt: {dataset_cleaning_prompt}")
        return
    
    input_dir = base_cleaning_dir / 'llm_csv'
    if not input_dir.exists():
        logging.error(f"Input directory not found: {input_dir}")
        logging.error(f"Please ensure the dataset cleaning step completed successfully")
        return
    
    # New structure: variable extraction results go to model/prompt subfolders
    prompt_stem = Path(prompt).stem
    run_output_dir = BENCHMARKING_ROOT / 'results' / '03_variable_extraction' / model / prompt_stem
    llm_csv_output_dir = run_output_dir / 'llm_csv'
    perfect_comparison_dir = run_output_dir / 'perfect_transcriptions_xlsx'
    student_comparison_dir = run_output_dir / 'student_transcriptions_xlsx'
    
    run_output_dir.mkdir(parents=True, exist_ok=True)
    llm_csv_output_dir.mkdir(exist_ok=True)
    perfect_comparison_dir.mkdir(exist_ok=True)
    student_comparison_dir.mkdir(exist_ok=True)
    
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output will be saved in: {run_output_dir}")

    # 1. Process CSV files from the previous benchmarking step
    csv_files = list(input_dir.glob('*_cleaned.csv'))
    if not csv_files:
        logging.error(f"No cleaned CSV files found in {input_dir}. Cannot proceed.")
        return

    logging.info(f"Found {len(csv_files)} CSV files to process.")
    processed_count = 0
    skipped_count = 0
    
    # Filter out CSV files that already have corresponding processed files
    csvs_to_process = []
    for csv_path in csv_files:
        processed_csv_path = llm_csv_output_dir / f"{csv_path.stem}_with_variables.csv"
        
        if processed_csv_path.exists():
            logging.info(f"Processed CSV already exists for {csv_path.name}, skipping.")
            skipped_count += 1
        else:
            csvs_to_process.append(csv_path)
    
    # Process remaining CSV files sequentially
    if csvs_to_process:
        logging.info(f"Processing {len(csvs_to_process)} CSV files sequentially...")
        
        for csv_path in csvs_to_process:
            success = process_single_csv(csv_path, llm_csv_output_dir, prompt_template, model)
            if success:
                processed_count += 1
            else:
                logging.error(f"Failed to process: {csv_path.name}")
    
    logging.info(f"CSV processing complete. Processed: {processed_count}, Skipped: {skipped_count}")
    logging.info("--- Starting variable comparison phase. ---")

    # 2. Run variable comparisons against both ground truth types
    all_results = {}
    
    # Perfect transcriptions comparison
    perfect_gt_dir = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'perfect_transcriptions_xlsx'
    if perfect_gt_dir.exists():
        logging.info("Running variable comparison against perfect transcriptions...")
        perfect_results = run_variable_comparison(
            llm_csv_dir=llm_csv_output_dir,
            gt_xlsx_dir=perfect_gt_dir,
            output_dir=perfect_comparison_dir,
            comparison_type="perfect"
        )
        if perfect_results:
            all_results['perfect'] = perfect_results
            logging.info("Perfect transcriptions variable comparison completed.")
        else:
            logging.warning("No perfect transcriptions variable comparison results generated.")
    else:
        logging.warning(f"Perfect transcriptions directory not found: {perfect_gt_dir}")
    
    # Student transcriptions comparison
    student_gt_dir = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'student_transcriptions_xlsx'
    if student_gt_dir.exists():
        logging.info("Running variable comparison against student transcriptions...")
        student_results = run_variable_comparison(
            llm_csv_dir=llm_csv_output_dir,
            gt_xlsx_dir=student_gt_dir,
            output_dir=student_comparison_dir,
            comparison_type="student"
        )
        if student_results:
            all_results['student'] = student_results
            logging.info("Student transcriptions variable comparison completed.")
        else:
            logging.warning("No student transcriptions variable comparison results generated.")
    else:
        logging.warning(f"Student transcriptions directory not found: {student_gt_dir}")
    
    # 3. Generate combined results.json at prompt level
    if all_results:
        combined_results = {
            'model': model,
            'prompt': prompt_stem,
            'timestamp': pd.Timestamp.now().isoformat(),
            'perfect': all_results.get('perfect', {}),
            'student': all_results.get('student', {}),
            'summary': {
                'perfect_match_rate': all_results.get('perfect', {}).get('overall_match_rate', 0),
                'student_match_rate': all_results.get('student', {}).get('overall_match_rate', 0),
                'perfect_cells': all_results.get('perfect', {}).get('total_cells', 0),
                'student_cells': all_results.get('student', {}).get('total_cells', 0),
                'files_processed': (
                    all_results.get('perfect', {}).get('files_processed', 0) +
                    all_results.get('student', {}).get('files_processed', 0)
                )
            }
        }
        
        results_path = run_output_dir / "results.json"
        with results_path.open('w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=4)
        logging.info(f"Combined results saved to {results_path}")
    else:
        logging.warning("No comparison results generated. Check if ground truth files exist.")
    
    logging.info(f"--- Variable extraction benchmark finished ---")

# --- Main Function ---

def main():
    """
    Main function to parse arguments and orchestrate the benchmarking runs.
    """
    parser = argparse.ArgumentParser(description="Run the variable extraction benchmarking pipeline.")
    parser.add_argument(
        '--dataset_cleaning_model',
        type=str,
        choices=MODELS,
        help='The name of the model used in the previous dataset cleaning step.'
    )
    parser.add_argument(
        '--dataset_cleaning_prompt',
        type=str,
        help='The filename of the prompt used in the previous dataset cleaning step (e.g., "cleaning_v0.0_prompt.txt").'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=MODELS,
        help='The name of the model to use for variable extraction.'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='The filename of the prompt to use for variable extraction (e.g., "variable_extraction_v0.0_prompt.txt").'
    )
    
    args = parser.parse_args()

    if args.dataset_cleaning_model and args.dataset_cleaning_prompt and args.model and args.prompt:
        run_single_benchmark(args.dataset_cleaning_model, args.dataset_cleaning_prompt, args.model, args.prompt)
        logging.info("--- Single benchmark run complete. ---")
        logging.info(f"To generate/update the main dashboard, run: python src/benchmarking/scripts/create_dashboard.py")
    else:
        parser.print_help()
        logging.warning("Please specify all required arguments: --dataset_cleaning_model, --dataset_cleaning_prompt, --model, and --prompt.")
        sys.exit(1)

if __name__ == "__main__":
    main() 