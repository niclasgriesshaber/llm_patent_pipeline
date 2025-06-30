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
COMPLETE_CSVS = PROJECT_ROOT / "data" / "02_complete_csvs"
CLEANED_XLSX_TEMP = PROJECT_ROOT / "data" / "03_cleaned_xlsx_temp"
CLEANED_CSVS = PROJECT_ROOT / "data" / "04_cleaned_csvs"
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

def postprocess_and_save(df, xlsx_path, csv_path, summary_path, failed_rows):
    # Save xlsx with complete_patent column
    df.to_excel(xlsx_path, index=False)
    logging.info(f"Saved intermediate xlsx to: {xlsx_path}")

    # Post-processing for CSV
    df_clean = df.copy().reset_index(drop=True)
    removed_isolated = 0
    merged_pairs = 0
    deleted_runs = 0
    failed_count = (df_clean["complete_patent"] == "LLM failed").sum()

    # Helper: find runs of 0s (not LLM failed)
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

    # Process runs
    to_remove = set()
    to_merge = []
    for start, end in runs:
        length = end - start + 1
        if length == 1:
            # Isolated 0
            # Check neighbors (not out of bounds, not LLM failed)
            prev_ok = (start > 0 and df_clean.at[start-1, "complete_patent"] == "1")
            next_ok = (end < len(df_clean)-1 and df_clean.at[end+1, "complete_patent"] == "1")
            if prev_ok and next_ok:
                to_remove.add(start)
                removed_isolated += 1
        elif length == 2:
            # Merge these two
            to_merge.append((start, end))
            merged_pairs += 1
        else:
            # Delete all
            for i in range(start, end+1):
                to_remove.add(i)
            deleted_runs += 1

    # Merge pairs
    for start, end in to_merge:
        # Merge entry, keep category/page from first, id from first
        merged_entry = df_clean.at[start, "entry"] + " " + df_clean.at[end, "entry"]
        df_clean.at[start, "entry"] = merged_entry
        # Remove the second row
        to_remove.add(end)

    # Remove rows (sort descending so index stays valid)
    to_remove = sorted(to_remove, reverse=True)
    for idx in to_remove:
        df_clean = df_clean.drop(idx)
    df_clean = df_clean.reset_index(drop=True)

    # Reassign ids sequentially starting from 1
    df_clean["id"] = range(1, len(df_clean)+1)

    # Remove complete_patent column before saving final CSV
    if "complete_patent" in df_clean.columns:
        df_clean = df_clean.drop(columns=["complete_patent"])

    df_clean.to_csv(csv_path, index=False)
    logging.info(f"Saved cleaned csv to: {csv_path}")

    # Write summary file
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Isolated incomplete rows removed: {removed_isolated}\n")
        f.write(f"Pairs of incomplete rows merged: {merged_pairs}\n")
        f.write(f"Runs of >2 incomplete rows deleted: {deleted_runs}\n")
        f.write(f"LLM failures: {failed_count}\n")
        if failed_rows:
            f.write("\nFailed rows (id,page):\n")
            for _, id_val, page_val in failed_rows:
                f.write(f"{id_val},{page_val}\n")
    logging.info(f"Saved summary file to: {summary_path}")

    # Summary
    logging.info("")
    logging.info("Summary".center(60, "-"))
    logging.info(f"LLM failures: {failed_count}")
    logging.info(f"Isolated incomplete rows removed: {removed_isolated}")
    logging.info(f"Pairs of incomplete rows merged: {merged_pairs}")
    logging.info(f"Runs of >2 incomplete rows deleted: {deleted_runs}")
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
    xlsx_path = CLEANED_XLSX_TEMP / f"{filestem}.xlsx"
    csv_path = CLEANED_CSVS / f"{filestem}.csv"
    summary_path = CLEANED_CSVS / f"summary_{filestem}.txt"

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

    # Save xlsx and post-process for csv and summary
    postprocess_and_save(df, xlsx_path, csv_path, summary_path, failed_rows)
    elapsed = time.time() - start_time
    logging.info(f"Total script time: {elapsed:.1f} seconds")

if __name__ == "__main__":
    main() 