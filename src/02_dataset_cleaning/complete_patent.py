import os
import sys
import argparse
import logging
import pandas as pd
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
COMPLETE_CSVS = PROJECT_ROOT / "data" / "01_dataset_construction" / "complete_csvs"
CLEANED_XLSX_TEMP = PROJECT_ROOT / "data" / "02_dataset_cleaning" / "cleaned_xlsx"
CLEANED_CSVS = PROJECT_ROOT / "data" / "02_dataset_cleaning" / "cleaned_csvs"
PROMPT_PATH = PROJECT_ROOT / "src" / "02_dataset_cleaning" / "prompt.txt"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

# Load environment
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM config - Updated to support gemini-2.5-flash-lite
FULL_MODEL_NAME = "gemini-2.5-flash-lite"  # Updated default model
MAX_OUTPUT_TOKENS = 128
MAX_WORKERS = 8

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

# Global token tracking
total_input_tokens = 0
total_thought_tokens = 0
total_candidate_tokens = 0
total_failed_calls = 0  # Track failed API calls for cost monitoring

def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")

def call_llm(entry: str, prompt_template: str) -> tuple[str, dict]:
    global total_input_tokens, total_thought_tokens, total_candidate_tokens, total_failed_calls
    
    client = genai.Client(api_key=API_KEY)
    prompt = f"{prompt_template}\n{entry.strip()}"
    
    # Configure model-specific settings
    config_args = {
        "temperature": 0.0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    
    # For gemini-2.5 models, set thinking_config with minimum thinking_budget
    if "2.5" in FULL_MODEL_NAME:
        if "lite" in FULL_MODEL_NAME:
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
    
    # Retry logic for any failure to get valid response (0 or 1)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=FULL_MODEL_NAME,
                contents=[prompt],
                config=config,
            )
            if not response or not response.text:
                logging.warning(f"Empty response from {FULL_MODEL_NAME}")
                if attempt < max_retries - 1:
                    logging.warning(f"Retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    continue
                else:
                    total_failed_calls += 1
                    return "LLM failed", {}
            
            # Extract token usage
            usage = getattr(response, 'usage_metadata', None)
            if usage:
                # Extract token counts with robust handling of "N/A" and non-numeric values
                ptk_raw = getattr(usage, 'prompt_token_count', None)
                ctk_raw = getattr(usage, 'candidates_token_count', None)
                ttk_raw = getattr(usage, 'thinking_token_count', None)
                totk_raw = getattr(usage, 'total_token_count', None)
                
                # Convert to integers, treating "N/A" and non-numeric values as 0
                try:
                    ptk = int(ptk_raw) if ptk_raw is not None and ptk_raw != "N/A" else 0
                except (ValueError, TypeError):
                    ptk = 0
                    logging.warning(f"Invalid prompt_token_count value: {ptk_raw}, using 0")
                
                try:
                    ctk = int(ctk_raw) if ctk_raw is not None and ctk_raw != "N/A" else 0
                except (ValueError, TypeError):
                    ctk = 0
                    logging.warning(f"Invalid candidates_token_count value: {ctk_raw}, using 0")
                
                try:
                    ttk = int(ttk_raw) if ttk_raw is not None and ttk_raw != "N/A" else 0
                except (ValueError, TypeError):
                    ttk = 0
                    logging.warning(f"Invalid thinking_token_count value: {ttk_raw}, using 0")
                
                try:
                    totk = int(totk_raw) if totk_raw is not None and totk_raw != "N/A" else 0
                except (ValueError, TypeError):
                    totk = 0
                    logging.warning(f"Invalid total_token_count value: {totk_raw}, using 0")
                
                # Log if we encountered any "N/A" values
                if any(val == "N/A" for val in [ptk_raw, ctk_raw, ttk_raw, totk_raw]):
                    logging.info(f"Encountered 'N/A' token values: prompt={ptk_raw}, candidate={ctk_raw}, thoughts={ttk_raw}, total={totk_raw}")
                
                # Update global token counts
                total_input_tokens += ptk
                total_candidate_tokens += ctk
                total_thought_tokens += ttk
                
                token_info = {
                    'prompt_tokens': ptk,
                    'thoughts_tokens': ttk,
                    'candidate_tokens': ctk,
                    'total_tokens': totk
                }
            else:
                token_info = {}
            
            text = response.text.strip()
            
            # Debug: Log the actual response for troubleshooting
            logging.info(f"Raw response from {FULL_MODEL_NAME}: '{text}'")
            
            # Log token usage for successful API calls
            if usage:
                logging.info(f"Token usage: prompt={ptk}, candidate={ctk}, thoughts={ttk}, total={totk}")
            
            # Check for exact matches first
            if text == "1" or text == "0":
                return text, token_info
            
            # Check for responses that contain the expected values
            if "1" in text and "0" not in text:
                logging.info(f"Extracted '1' from response: '{text}'")
                return "1", token_info
            elif "0" in text and "1" not in text:
                logging.info(f"Extracted '0' from response: '{text}'")
                return "0", token_info
            elif "1" in text and "0" in text:
                # If both are present, check which comes first or is more prominent
                if text.find("1") < text.find("0"):
                    logging.info(f"Extracted '1' (appears first) from response: '{text}'")
                    return "1", token_info
                else:
                    logging.info(f"Extracted '0' (appears first) from response: '{text}'")
                    return "0", token_info
            
            logging.warning(f"Unexpected response from {FULL_MODEL_NAME}: '{text}'")
            if attempt < max_retries - 1:
                logging.warning(f"Retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            else:
                total_failed_calls += 1
                return "LLM failed", token_info
            
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
                logging.warning(f"API failure for {FULL_MODEL_NAME} (attempt {attempt + 1}/{max_retries}): {error_msg}")
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            else:
                # Either not an API failure or max retries reached
                logging.error(f"LLM call failed for {FULL_MODEL_NAME} (attempt {attempt + 1}/{max_retries}): {error_msg}")
                
                # Check for specific error types
                if "429" in error_msg or "rate limit" in error_msg.lower() or "resource exhausted" in error_msg.lower():
                    logging.warning(f"Rate limit detected for {FULL_MODEL_NAME}, consider reducing MAX_WORKERS")
                elif "thinking" in error_msg.lower():
                    logging.error(f"Thinking budget configuration issue for {FULL_MODEL_NAME}")
                elif "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                    logging.error(f"Model {FULL_MODEL_NAME} not found or not available")
                
                total_failed_calls += 1
                return "LLM failed", {}
    
    total_failed_calls += 1
    return "LLM failed", {}

def create_processing_log(logs_dir: Path, filestem: str, csv_name: str, row_count: int, 
                         processing_time: float, max_workers: int) -> None:
    """Create a JSON log file with processing information."""
    log_file = logs_dir / f"{filestem}_cleaned_logs.json"
    
    log_data = {
        "file_name": csv_name,
        "model": FULL_MODEL_NAME,
        "number_of_rows": row_count,
        "total_input_tokens": total_input_tokens,
        "total_thought_tokens": total_thought_tokens,
        "total_candidate_tokens": total_candidate_tokens,
        "total_tokens": total_input_tokens + total_thought_tokens + total_candidate_tokens,  # Add total tokens
        "total_failed_calls": total_failed_calls,  # Track failed calls for cost monitoring
        "processing_time_seconds": round(processing_time, 2),
        "max_workers": max_workers
    }
    
    try:
        with log_file.open("w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Processing log saved to: {log_file}")
    except Exception as e:
        logging.error(f"Failed to write processing log {log_file}: {e}")

def process_llm(df, prompt_template):
    results = [None] * len(df)
    failures = 0
    failed_rows = []  # To store (idx, id, page) for rows that fail twice
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(call_llm, row["entry"], prompt_template): idx
            for idx, row in df.iterrows()
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result, token_info = future.result()
                if result == "LLM failed":
                    failures += 1
                    failed_rows.append((idx, df.iloc[idx]["id"], df.iloc[idx]["page"]))
                    logging.error(f"Row {idx+1}: LLM failed after all retries.")
                else:
                    results[idx] = result
                    logging.info(f"Row {idx+1}: LLM result = {result}")
            except Exception as e:
                results[idx] = "LLM failed"
                failures += 1
                failed_rows.append((idx, df.iloc[idx]["id"], df.iloc[idx]["page"]))
                logging.error(f"Row {idx+1}: Exception during LLM processing: {e}")
    return results, failures, failed_rows

def postprocess_and_save(df, xlsx_path, csv_path, failed_rows):
    # Save xlsx with complete_patent column
    df.to_excel(xlsx_path, index=False)
    logging.info(f"Saved intermediate xlsx to: {xlsx_path}")

    # Create logs directory if it doesn't exist
    logs_dir = CLEANED_XLSX_TEMP / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create summary file path
    filestem = xlsx_path.stem
    summary_path = logs_dir / f"{filestem}.txt"

    df_clean = df.copy().reset_index(drop=True)
    merged_isolated = 0
    pair_count = 0
    run_gt2_count = 0
    failed_count = (df_clean["complete_patent"] == "LLM failed").sum()

    # Track detailed information for logging
    merged_details = []  # (original_id, original_page, merged_with_id, merged_with_page, new_id)
    pair_details = []    # [(id, page), (id, page)]
    run_gt2_details = [] # [(id, page), (id, page), ...]
    failed_details = []  # (id, page)

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
                # Track details before merging
                original_id = df_clean.at[start, "id"]
                original_page = df_clean.at[start, "page"]
                merged_with_id = df_clean.at[end+1, "id"]
                merged_with_page = df_clean.at[end+1, "page"]
                
                # Merge entries
                df_clean.at[start, "entry"] = df_clean.at[start, "entry"] + " " + df_clean.at[end+1, "entry"]
                # For merged rows, set complete_patent to 1 (complete) since we're creating a complete entry
                df_clean.at[start, "complete_patent"] = "1"
                # Remove the row below
                to_remove.add(end+1)
                merged_isolated += 1
                
                # Store details for logging (new_id will be calculated after reassignment)
                merged_details.append((original_id, original_page, merged_with_id, merged_with_page))
        elif length == 2:
            # Pair: leave as is, count
            pair_details.append([
                (df_clean.at[start, "id"], df_clean.at[start, "page"]),
                (df_clean.at[end, "id"], df_clean.at[end, "page"])
            ])
            pair_count += 1
        elif length > 2:
            # Run >2: leave as is, count
            run_entries = []
            for i in range(start, end + 1):
                run_entries.append((df_clean.at[i, "id"], df_clean.at[i, "page"]))
            run_gt2_details.append(run_entries)
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

    # Write detailed summary file
    with open(summary_path, "w", encoding="utf-8") as f:
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Isolated incomplete rows merged with below: {merged_isolated}\n")
        f.write(f"Pairs of incomplete rows: {pair_count}\n")
        f.write(f"Runs of >2 incomplete rows: {run_gt2_count}\n")
        f.write(f"LLM failures: {failed_count}\n")
        f.write(f"Final row count: {len(df_clean)}\n\n")
        
        # Isolated incomplete rows merged with below
        if merged_details:
            f.write("ISOLATED INCOMPLETE ROWS MERGED WITH BELOW\n")
            f.write("-" * 50 + "\n")
            for i, (orig_id, orig_page, merged_id, merged_page) in enumerate(merged_details):
                # Find the new ID in the cleaned dataframe
                # Look for the row that contains the merged entry
                new_id = None
                for idx, row in df_clean.iterrows():
                    if orig_page in str(row["entry"]) and merged_page in str(row["entry"]):
                        new_id = row["id"]
                        break
                
                f.write(f"Merge {i+1}: Original (id: {orig_id}, page: {orig_page}) + (id: {merged_id}, page: {merged_page}) -> New (id: {new_id})\n")
            f.write("\n")
        
        # Pairs of incomplete rows
        if pair_details:
            f.write("PAIRS OF INCOMPLETE ROWS\n")
            f.write("-" * 50 + "\n")
            for i, pair in enumerate(pair_details):
                f.write(f"Pair {i+1}: (id: {pair[0][0]}, page: {pair[0][1]}) and (id: {pair[1][0]}, page: {pair[1][1]})\n")
            f.write("\n")
        
        # Runs of >2 incomplete rows
        if run_gt2_details:
            f.write("RUNS OF >2 INCOMPLETE ROWS\n")
            f.write("-" * 50 + "\n")
            for i, run in enumerate(run_gt2_details):
                f.write(f"Run {i+1}: ")
                for j, (id_val, page_val) in enumerate(run):
                    if j > 0:
                        f.write(", ")
                    f.write(f"(id: {id_val}, page: {page_val})")
                f.write("\n")
            f.write("\n")
        
        # LLM failures
        if failed_rows:
            f.write("LLM FAILURES\n")
            f.write("-" * 50 + "\n")
            for _, id_val, page_val in failed_rows:
                f.write(f"(id: {id_val}, page: {page_val})\n")
    
    logging.info(f"Saved detailed summary file to: {summary_path}")

    # Summary
    logging.info("")
    logging.info("Summary".center(60, "-"))
    logging.info(f"LLM failures: {failed_count}")
    logging.info(f"Isolated incomplete rows merged with below: {merged_isolated}")
    logging.info(f"Pairs of incomplete rows: {pair_count}")
    logging.info(f"Runs of >2 incomplete rows: {run_gt2_count}")
    logging.info(f"Final row count: {len(df_clean)}")
    logging.info("-"*60)

def main():
    parser = argparse.ArgumentParser(description="Check completeness of patent entries using LLM.")
    parser.add_argument("--csv", type=str, required=True, help="Name of the CSV file in data/complete_csvs/ to process.")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite", 
                       choices=["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"],
                       help="Model to use for LLM processing (default: gemini-2.5-flash-lite)")
    parser.add_argument("--max_workers", type=int, default=20, 
                       help="Max concurrent workers for API requests (default: 20)")
    args = parser.parse_args()

    # Update global model name and max workers based on command line arguments
    global FULL_MODEL_NAME, MAX_WORKERS
    FULL_MODEL_NAME = args.model
    MAX_WORKERS = args.max_workers

    input_csv = COMPLETE_CSVS / args.csv
    if not input_csv.exists():
        logging.error(f"Input file not found: {input_csv}")
        sys.exit(1)

    # Output paths
    filestem = input_csv.stem
    # Extract year from filename (e.g., "Patentamt_1889" -> "1889")
    year = filestem.split('_')[-1] if '_' in filestem else filestem
    xlsx_path = CLEANED_XLSX_TEMP / f"Patentamt_{year}_cleaned.xlsx"
    csv_path = CLEANED_CSVS / f"Patentamt_{year}_cleaned.csv"
    
    # Create output directories if they don't exist
    CLEANED_XLSX_TEMP.mkdir(parents=True, exist_ok=True)
    CLEANED_CSVS.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(input_csv)
    for col in ["id", "page", "entry", "category"]:
        if col not in df.columns:
            logging.error(f"Missing required column: {col}")
            sys.exit(1)

    prompt_template = load_prompt()
    logging.info(f"Processing {len(df)} rows from {input_csv.name}")
    logging.info(f"Model: {FULL_MODEL_NAME}")
    logging.info(f"Max Workers: {MAX_WORKERS}")
    start_time = time.time()

    # LLM processing
    results, failures, failed_rows = process_llm(df, prompt_template)
    df["complete_patent"] = results

    # Save xlsx and post-process for csv
    postprocess_and_save(df, xlsx_path, csv_path, failed_rows)
    elapsed = time.time() - start_time
    logging.info(f"Total script time: {elapsed:.1f} seconds")

    # Create processing log
    create_processing_log(logs_dir=CLEANED_XLSX_TEMP / "logs", filestem=filestem, csv_name=args.csv, 
                          row_count=len(df), processing_time=elapsed, max_workers=MAX_WORKERS)

    # Log token usage summary
    logging.info(f"Token usage summary: prompt={total_input_tokens:,}, candidate={total_candidate_tokens:,}, thoughts={total_thought_tokens:,}, total={total_input_tokens + total_thought_tokens + total_candidate_tokens:,}")
    logging.info(f"Failed API calls: {total_failed_calls}")

if __name__ == "__main__":
    main() 