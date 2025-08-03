import argparse
import logging
import json
import os
import pandas as pd
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

# Core modules are now in the same directory
from core.benchmarking import run_comparison
from create_dashboard import create_dashboard

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define project root
project_root = Path(__file__).parent.parent.parent.parent
BENCHMARKING_ROOT = project_root / 'data' / 'benchmarking'
PROMPTS_DIR = project_root / 'src' / 'benchmarking' / 'prompts' / '02_dataset_cleaning'
ENV_PATH = project_root / 'config' / '.env'

# Load environment
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM config
MAX_OUTPUT_TOKENS = 128
MAX_WORKERS = 8

MODELS = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-pro']

# --- LLM Processing Functions (adapted from complete_patent.py) ---

def load_prompt(prompt_name: str) -> str:
    """Load the dataset cleaning prompt."""
    prompt_path = PROMPTS_DIR / prompt_name
    return prompt_path.read_text(encoding="utf-8")

def call_llm(entry: str, prompt_template: str, model_name: str) -> str:
    """Call the LLM to classify an entry as complete (1) or truncated (0)."""
    client = genai.Client(api_key=API_KEY)
    prompt = f"{prompt_template}\n{entry.strip()}"
    
    # Configure model-specific settings
    config_args = {
        "temperature": 0.0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    
    # For gemini-2.5 models, set thinking_config with minimum thinking_budget
    if "2.5" in model_name:
        config_args["thinking_config"] = types.ThinkingConfig(
            thinking_budget=128,  # Minimum required for 2.5 models
            include_thoughts=True
        )
    
    config = types.GenerateContentConfig(**config_args)
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config=config,
        )
        if not response or not response.text:
            return "LLM failed"
        text = response.text.strip()
        if text == "1" or text == "0":
            return text
        return "LLM failed"
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        return "LLM failed"

def process_llm(df, prompt_template, model_name):
    """Process all entries in a dataframe using LLM."""
    results = [None] * len(df)
    failures = 0
    failed_rows = []  # To store (idx, id) for rows that fail twice
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(call_llm, row["entry"], prompt_template, model_name): idx
            for idx, row in df.iterrows()
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                if result == "LLM failed":
                    # Retry once
                    logging.info(f"Row {idx+1}: LLM failed, retrying...")
                    result_retry = call_llm(df.iloc[idx]["entry"], prompt_template, model_name)
                    if result_retry == "LLM failed":
                        failures += 1
                        failed_rows.append((idx, df.iloc[idx]["id"]))
                        results[idx] = "LLM failed"
                        logging.error(f"Row {idx+1}: LLM failed after retry.")
                    else:
                        results[idx] = result_retry
                        logging.info(f"Row {idx+1}: LLM retry result = {result_retry}")
                else:
                    results[idx] = result
                    logging.info(f"Row {idx+1}: LLM result = {result}")
            except Exception as e:
                results[idx] = "LLM failed"
                failures += 1
                failed_rows.append((idx, df.iloc[idx]["id"]))
                logging.error(f"Row {idx+1}: Exception during LLM processing: {e}")
    return results, failures, failed_rows

def postprocess_and_save(df, csv_path, summary_path, failed_rows):
    """Post-process the dataframe and save the cleaned CSV."""
    df_clean = df.copy().reset_index(drop=True)
    merged_isolated = 0
    pair_count = 0
    run_gt2_count = 0
    failed_count = (df_clean["complete_patent"] == "LLM failed").sum()

    mask = (df_clean["complete_patent"] == "0")
    runs = []
    run_start = None
    for i, val in enumerate(mask):
        if val:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                runs.append((run_start, i-1))
                run_start = None
    if run_start is not None:
        runs.append((run_start, len(mask)-1))

    to_remove = set()
    for start, end in runs:
        length = end - start + 1
        if length == 1:
            # Isolated incomplete row: merge with row below if possible
            if end + 1 < len(df_clean):
                # Merge entries
                df_clean.at[start, "entry"] = df_clean.at[start, "entry"] + " " + df_clean.at[end+1, "entry"]
                # For merged rows, set complete_patent to 1 (complete) since we're creating a complete entry
                df_clean.at[start, "complete_patent"] = "1"
                # Remove the row below
                to_remove.add(end+1)
                merged_isolated += 1
        elif length == 2:
            # Pair: leave as is, count
            pair_count += 1
        elif length > 2:
            # Run >2: leave as is, count
            run_gt2_count += 1

    # Remove rows (sort descending so index stays valid)
    to_remove = sorted(to_remove, reverse=True)
    for idx in to_remove:
        df_clean = df_clean.drop(idx)
    df_clean = df_clean.reset_index(drop=True)

    # Reassign ids sequentially starting from 1
    df_clean["id"] = range(1, len(df_clean)+1)

    # Keep complete_patent column in the CSV for transparency
    df_clean.to_csv(csv_path, index=False)
    logging.info(f"Saved cleaned csv to: {csv_path}")

    # Write summary file
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Isolated incomplete rows merged with below: {merged_isolated}\n")
        f.write(f"Pairs of incomplete rows: {pair_count}\n")
        f.write(f"Runs of >2 incomplete rows: {run_gt2_count}\n")
        f.write(f"LLM failures: {failed_count}\n")
        if failed_rows:
            f.write("\nFailed rows (id):\n")
            for _, id_val in failed_rows:
                f.write(f"{id_val}\n")
    logging.info(f"Saved summary file to: {summary_path}")

    # Summary
    logging.info("")
    logging.info("Summary".center(60, "-"))
    logging.info(f"LLM failures: {failed_count}")
    logging.info(f"Isolated incomplete rows merged with below: {merged_isolated}")
    logging.info(f"Pairs of incomplete rows: {pair_count}")
    logging.info(f"Runs of >2 incomplete rows: {run_gt2_count}")
    logging.info(f"Final row count: {len(df_clean)}")
    logging.info("-"*60)

    return df_clean

def process_single_csv(csv_path: Path, output_dir: Path, prompt_template: str, model_name: str) -> bool:
    """Process a single CSV file for dataset cleaning."""
    try:
        logging.info(f"Processing CSV: {csv_path.name}")
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Verify we have the essential columns
        if "id" not in df.columns or "entry" not in df.columns:
            logging.error(f"Missing essential columns 'id' or 'entry' in {csv_path.name}")
            return False

        # LLM processing
        results, failures, failed_rows = process_llm(df, prompt_template, model_name)
        df["complete_patent"] = results

        # Output paths
        filestem = csv_path.stem
        csv_output_path = output_dir / f"{filestem}_cleaned.csv"
        
        # Create logs directory if it doesn't exist
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        summary_path = logs_dir / f"{filestem}.txt"

        # Post-process and save
        postprocess_and_save(df, csv_output_path, summary_path, failed_rows)
        
        logging.info(f"Successfully processed: {csv_path.name}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing {csv_path.name}: {e}")
        return False

# --- Main Functions ---

def run_single_benchmark(dataset_construction_model: str, dataset_construction_prompt: str, model: str, prompt: str):
    """
    Executes the full benchmarking pipeline for dataset cleaning for a single model and prompt combination.
    """
    logging.info(f"--- Starting dataset cleaning benchmark ---")
    logging.info(f"Input: model=[{dataset_construction_model}] prompt=[{dataset_construction_prompt}]")
    logging.info(f"Processing: model=[{model}] prompt=[{prompt}]")

    # Load the dataset cleaning prompt
    try:
        prompt_template = load_prompt(prompt)
    except Exception as e:
        logging.error(f"Failed to load prompt file {prompt}: {e}")
        return

    # Define input and output directories
    dataset_construction_prompt_stem = Path(dataset_construction_prompt).stem
    
    # Check if the prerequisite dataset construction step has been completed
    base_construction_dir = BENCHMARKING_ROOT / 'results' / '01_dataset_construction' / dataset_construction_model / dataset_construction_prompt_stem
    if not base_construction_dir.exists():
        logging.error(f"Prerequisite dataset construction results not found: {base_construction_dir}")
        logging.error(f"Please run the dataset construction benchmark first for model: {dataset_construction_model}, prompt: {dataset_construction_prompt}")
        return
    
    input_dir = base_construction_dir / 'llm_csv'
    if not input_dir.exists():
        logging.error(f"Input directory not found: {input_dir}")
        logging.error(f"Please ensure the dataset construction step completed successfully")
        return
    
    # New structure: dataset cleaning results go to model/prompt subfolders
    prompt_stem = Path(prompt).stem
    run_output_dir = BENCHMARKING_ROOT / 'results' / '02_dataset_cleaning' / model / prompt_stem
    llm_csv_output_dir = run_output_dir / 'llm_csv'
    perfect_comparison_dir = run_output_dir / 'perfect_transcriptions_xlsx'
    student_comparison_dir = run_output_dir / 'student_transcriptions_xlsx'
    
    run_output_dir.mkdir(parents=True, exist_ok=True)
    llm_csv_output_dir.mkdir(exist_ok=True)
    perfect_comparison_dir.mkdir(exist_ok=True)
    student_comparison_dir.mkdir(exist_ok=True)
    
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output will be saved in: {run_output_dir}")

    # Check if input directory exists
    if not input_dir.exists():
        logging.error(f"Input directory not found: {input_dir}. Cannot proceed.")
        return

    # 1. Process CSV files from the previous benchmarking step
    csv_files = list(input_dir.glob('*.csv'))
    if not csv_files:
        logging.error(f"No CSV files found in {input_dir}. Cannot proceed.")
        return

    logging.info(f"Found {len(csv_files)} CSV files to process.")
    processed_count = 0
    skipped_count = 0
    
    # Filter out CSV files that already have corresponding cleaned files
    csvs_to_process = []
    for csv_path in csv_files:
        cleaned_csv_path = llm_csv_output_dir / f"{csv_path.stem}_cleaned.csv"
        
        if cleaned_csv_path.exists():
            logging.info(f"Cleaned CSV already exists for {csv_path.name}, skipping.")
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
    logging.info("--- Starting comparison phase. ---")

    # 2. Run comparisons against both ground truth types
    all_results = {}
    
    # Create temporary directory with correctly named LLM files for comparison
    temp_llm_dir = run_output_dir / 'temp_llm_for_comparison'
    temp_llm_dir.mkdir(exist_ok=True)
    
    # Copy cleaned CSV files to temp directory with original names (without _cleaned suffix)
    for cleaned_csv in llm_csv_output_dir.glob('*_cleaned.csv'):
        original_name = cleaned_csv.stem.replace('_cleaned', '') + '.csv'
        temp_csv = temp_llm_dir / original_name
        import shutil
        shutil.copy2(cleaned_csv, temp_csv)
    
    # Perfect transcriptions comparison
    perfect_gt_dir = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'perfect_transcriptions_xlsx'
    if perfect_gt_dir.exists():
        logging.info("Running comparison against perfect transcriptions...")
        perfect_results = run_comparison(
            llm_csv_dir=temp_llm_dir,
            gt_xlsx_dir=perfect_gt_dir,
            output_dir=perfect_comparison_dir,
            comparison_type="perfect"
        )
        if perfect_results:
            all_results['perfect'] = perfect_results
            logging.info("Perfect transcriptions comparison completed.")
        else:
            logging.warning("No perfect transcriptions comparison results generated.")
    else:
        logging.warning(f"Perfect transcriptions directory not found: {perfect_gt_dir}")
    
    # Student transcriptions comparison
    student_gt_dir = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'student_transcriptions_xlsx'
    if student_gt_dir.exists():
        logging.info("Running comparison against student transcriptions...")
        student_results = run_comparison(
            llm_csv_dir=temp_llm_dir,
            gt_xlsx_dir=student_gt_dir,
            output_dir=student_comparison_dir,
            comparison_type="student"
        )
        if student_results:
            all_results['student'] = student_results
            logging.info("Student transcriptions comparison completed.")
        else:
            logging.warning("No student transcriptions comparison results generated.")
    else:
        logging.warning(f"Student transcriptions directory not found: {student_gt_dir}")
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_llm_dir, ignore_errors=True)
    
    # 3. Generate combined results.json at prompt level
    if all_results:
        combined_results = {
            'model': model,
            'prompt': prompt_stem,
            'timestamp': pd.Timestamp.now().isoformat(),
            'perfect': all_results.get('perfect', {}),
            'student': all_results.get('student', {}),
            'summary': {
                'perfect_cer': all_results.get('perfect', {}).get('character_error_rate', 0),
                'student_cer': all_results.get('student', {}).get('character_error_rate', 0),
                'perfect_match_rate': all_results.get('perfect', {}).get('overall_match_rate', 0),
                'student_match_rate': all_results.get('student', {}).get('overall_match_rate', 0),
                'files_processed': len(set(
                    all_results.get('perfect', {}).get('files_with_results', []) +
                    all_results.get('student', {}).get('files_with_results', [])
                ))
            }
        }
        
        results_path = run_output_dir / "results.json"
        with results_path.open('w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=4)
        logging.info(f"Combined results saved to {results_path}")
    else:
        logging.warning("No comparison results generated. Check if ground truth files exist.")
    
    logging.info(f"--- Dataset cleaning benchmark finished ---")

def main():
    """
    Main function to parse arguments and orchestrate the benchmarking runs.
    """
    parser = argparse.ArgumentParser(description="Run the dataset cleaning benchmarking pipeline.")
    parser.add_argument(
        '--dataset_construction_model',
        type=str,
        choices=MODELS,
        help='The name of the model used in the previous dataset construction step.'
    )
    parser.add_argument(
        '--dataset_construction_prompt',
        type=str,
        help='The filename of the prompt used in the previous dataset construction step (e.g., "construction_v0.4_prompt.txt").'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=MODELS,
        help='The name of the model to use for dataset cleaning.'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='The filename of the prompt to use for dataset cleaning (e.g., "cleaning_v0.0_prompt.txt").'
    )

    
    args = parser.parse_args()

    if args.dataset_construction_model and args.dataset_construction_prompt and args.model and args.prompt:
        run_single_benchmark(args.dataset_construction_model, args.dataset_construction_prompt, args.model, args.prompt)
        logging.info("--- Single dataset cleaning benchmark run complete. ---")
    else:
        parser.print_help()
        logging.warning("Please specify --dataset_construction_model, --dataset_construction_prompt, --model, and --prompt.")

if __name__ == "__main__":
    main() 