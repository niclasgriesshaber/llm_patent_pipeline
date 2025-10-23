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
from core.benchmarking import run_after_cleaning_comparison, run_unified_comparison

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

MODELS = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-lite']

# --- LLM Processing Functions (adapted from complete_patent.py) ---

def is_special_volume_csv(csv_path: Path) -> bool:
    """Check if the CSV is from 1878 or 1879 special volumes."""
    filename = csv_path.name
    return '1878' in filename or '1879' in filename

def load_prompt(prompt_name: str) -> str:
    """Load the dataset cleaning prompt."""
    prompt_path = PROMPTS_DIR / prompt_name
    return prompt_path.read_text(encoding="utf-8")

def get_prompt_for_csv(csv_path: Path, prompt_name: str) -> str:
    """Get the appropriate prompt text for a CSV file."""
    if is_special_volume_csv(csv_path):
        # Use special volumes prompt for 1878/1879
        special_prompt_file = PROMPTS_DIR / 'special_volumes_prompt.txt'
        if not special_prompt_file.exists():
            logging.error(f"Special volumes prompt file not found: {special_prompt_file}. Aborting.")
            raise FileNotFoundError(f"Special volumes prompt file not found: {special_prompt_file}")
        return special_prompt_file.read_text(encoding='utf-8')
    else:
        # Use regular prompt for other years
        prompt_file = PROMPTS_DIR / prompt_name
        if not prompt_file.exists():
            logging.error(f"Prompt file not found: {prompt_file}. Aborting.")
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        return prompt_file.read_text(encoding='utf-8')

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
        if "lite" in model_name:
            # For lite model: no thinking, minimal output tokens
            config_args["max_output_tokens"] = 1
            config_args["thinking_config"] = types.ThinkingConfig(
                thinking_budget=0,
                include_thoughts=False
            )
        else:
            # For other 2.5 models: use thinking config
            thinking_budget = 128  # Minimum required for other 2.5 models
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
                return "LLM failed"
            text = response.text.strip()
            
            # Debug: Log the actual response for troubleshooting
            logging.info(f"Raw response from {model_name}: '{text}'")
            
            # Check for exact matches first
            if text == "1" or text == "0":
                return text
            
            # Check for responses that contain the expected values
            if "1" in text and "0" not in text:
                logging.info(f"Extracted '1' from response: '{text}'")
                return "1"
            elif "0" in text and "1" not in text:
                logging.info(f"Extracted '0' from response: '{text}'")
                return "0"
            elif "1" in text and "0" in text:
                # If both are present, check which comes first or is more prominent
                if text.find("1") < text.find("0"):
                    logging.info(f"Extracted '1' (appears first) from response: '{text}'")
                    return "1"
                else:
                    logging.info(f"Extracted '0' (appears first) from response: '{text}'")
                    return "0"
            
            logging.warning(f"Unexpected response from {model_name}: '{text}'")
            return "LLM failed"
            
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
                logging.warning(f"API failure for {model_name} (attempt {attempt + 1}/{max_retries}): {error_msg}")
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            else:
                # Either not an API failure or max retries reached
                logging.error(f"LLM call failed for {model_name} (attempt {attempt + 1}/{max_retries}): {error_msg}")
                
                # Check for specific error types
                if "429" in error_msg or "rate limit" in error_msg.lower() or "resource exhausted" in error_msg.lower():
                    logging.warning(f"Rate limit detected for {model_name}, consider reducing MAX_WORKERS")
                elif "thinking" in error_msg.lower():
                    logging.error(f"Thinking budget configuration issue for {model_name}")
                elif "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                    logging.error(f"Model {model_name} not found or not available")
                
                return "LLM failed"
    
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
                results[idx] = result
                if result == "LLM failed":
                    failures += 1
                    failed_rows.append((idx, df.iloc[idx]["id"]))
                    logging.error(f"Row {idx+1}: LLM failed after all retries.")
                else:
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

def process_single_csv(csv_path: Path, output_dir: Path, prompt_name: str, model_name: str) -> bool:
    """Process a single CSV file for dataset cleaning."""
    try:
        # Get the appropriate prompt for this CSV
        prompt_template = get_prompt_for_csv(csv_path, prompt_name)
        prompt_type = "special volumes" if is_special_volume_csv(csv_path) else "regular"
        logging.info(f"Processing CSV: {csv_path.name} with {prompt_type} prompt")
        
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

def run_single_benchmark(dataset_construction_model: str, dataset_construction_prompt: str, model: str, prompt: str, threshold: float = 0.85):
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
    
    run_output_dir.mkdir(parents=True, exist_ok=True)
    llm_csv_output_dir.mkdir(exist_ok=True)
    
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
            success = process_single_csv(csv_path, llm_csv_output_dir, prompt, model)
            if success:
                processed_count += 1
            else:
                logging.error(f"Failed to process: {csv_path.name}")
    
    logging.info(f"CSV processing complete. Processed: {processed_count}, Skipped: {skipped_count}")
    logging.info("--- Starting after-cleaning comparison phase. ---")

    # 2. Run after-cleaning comparison (Perfect vs LLM-cleaned only)
    perfect_gt_dir = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'perfect_transcriptions_xlsx'
    if perfect_gt_dir.exists():
        logging.info("Running after-cleaning comparison against perfect transcriptions...")
        after_cleaning_results = run_after_cleaning_comparison(
            llm_csv_dir=llm_csv_output_dir,
            perfect_xlsx_dir=perfect_gt_dir,
            output_dir=run_output_dir,
            fuzzy_threshold=threshold
        )
        if after_cleaning_results:
            logging.info("After-cleaning comparison completed successfully.")
        else:
            logging.warning("No after-cleaning comparison results generated.")
    else:
        logging.warning(f"Perfect transcriptions directory not found: {perfect_gt_dir}")
    
    # 3. Run 3-way character error rate comparison (Perfect vs LLM-cleaned vs Student)
    student_xlsx_dir = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'student_transcriptions_xlsx'
    sampled_pdfs_dir = BENCHMARKING_ROOT / 'input_data' / 'sampled_pdfs'
    
    if student_xlsx_dir.exists() and perfect_gt_dir.exists() and sampled_pdfs_dir.exists():
        logging.info("Running 3-way character error rate comparison with cleaned data...")
        
        # Use the cleaned LLM CSV files for the 3-way comparison
        # Call a custom function that only generates the character error rate report
        from core.benchmarking import run_unified_comparison_cer_only
        
        unified_results = run_unified_comparison_cer_only(
            llm_csv_dir=llm_csv_output_dir,
            student_xlsx_dir=student_xlsx_dir,
            perfect_xlsx_dir=perfect_gt_dir,
            sampled_pdfs_dir=sampled_pdfs_dir,
            output_dir=run_output_dir,
            fuzzy_threshold=threshold
        )
        
        if unified_results:
            logging.info("3-way character error rate comparison completed successfully.")
            logging.info(f"Total files processed: {unified_results.get('total_files', 0)}")
            logging.info(f"Comparison results generated: {unified_results.get('comparison_results', 0)}")
        else:
            logging.warning("No 3-way character error rate comparison results generated.")
    else:
        logging.warning(f"Required directories not found for 3-way comparison:")
        if not student_xlsx_dir.exists():
            logging.warning(f"  - Student transcriptions: {student_xlsx_dir}")
        if not perfect_gt_dir.exists():
            logging.warning(f"  - Perfect transcriptions: {perfect_gt_dir}")
        if not sampled_pdfs_dir.exists():
            logging.warning(f"  - Sampled PDFs: {sampled_pdfs_dir}")
    
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
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Fuzzy matching threshold for patent entry matching (0.0-1.0). Default: 0.85'
    )

    
    args = parser.parse_args()
    
    # Validate threshold parameter
    if not (0.0 <= args.threshold <= 1.0):
        logging.error(f"Threshold must be between 0.0 and 1.0, got: {args.threshold}")
        sys.exit(1)

    if args.dataset_construction_model and args.dataset_construction_prompt and args.model and args.prompt:
        run_single_benchmark(args.dataset_construction_model, args.dataset_construction_prompt, args.model, args.prompt, args.threshold)
        logging.info("--- Single dataset cleaning benchmark run complete. ---")
    else:
        parser.print_help()
        logging.warning("Please specify --dataset_construction_model, --dataset_construction_prompt, --model, and --prompt.")

if __name__ == "__main__":
    main() 