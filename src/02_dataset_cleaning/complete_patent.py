import os
import sys
import argparse
import logging
import pandas as pd
import time
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

# LLM config
FULL_MODEL_NAME = "gemini-2.0-flash"
MAX_OUTPUT_TOKENS = 128
MAX_WORKERS = 8

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")

def call_llm(entry: str, prompt_template: str) -> str:
    client = genai.Client(api_key=API_KEY)
    prompt = f"{prompt_template}\n{entry.strip()}"
    try:
        response = client.models.generate_content(
            model=FULL_MODEL_NAME,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
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
                result = future.result()
                if result == "LLM failed":
                    # Retry once
                    logging.info(f"Row {idx+1}: LLM failed, retrying...")
                    result_retry = call_llm(df.iloc[idx]["entry"], prompt_template)
                    if result_retry == "LLM failed":
                        failures += 1
                        failed_rows.append((idx, df.iloc[idx]["id"], df.iloc[idx]["page"]))
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
            for _, id_val, _ in failed_rows:
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

def main():
    parser = argparse.ArgumentParser(description="Check completeness of patent entries using LLM.")
    parser.add_argument("--csv", type=str, required=True, help="Name of the CSV file in data/complete_csvs/ to process.")
    args = parser.parse_args()

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
    start_time = time.time()

    # LLM processing
    results, failures, failed_rows = process_llm(df, prompt_template)
    df["complete_patent"] = results

    # Save xlsx and post-process for csv
    postprocess_and_save(df, xlsx_path, csv_path, failed_rows)
    elapsed = time.time() - start_time
    logging.info(f"Total script time: {elapsed:.1f} seconds")

if __name__ == "__main__":
    main() 