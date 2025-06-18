#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import time
import re
import json
import pandas as pd
import google.genai as genai
from google.genai import types
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dotenv import load_dotenv

###############################################################################
# Load Environment
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / "config" / ".env"

load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")

###############################################################################
# Model Configuration
###############################################################################
FULL_MODEL_NAME = "gemini-2.0-flash"
MAX_OUTPUT_TOKENS = 8192
MAX_RETRIES = 3

###############################################################################
# Utilities
###############################################################################
def format_duration(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def parse_response(text: str) -> dict:
    # Strip away any backticks that might wrap the text
    candidate = re.sub(r"^`+|`+$", "", text.strip())

    try:
        # Attempt to parse the cleaned text as JSON
        return json.loads(candidate)
    except json.JSONDecodeError:
        logging.warning(f"Failed to parse JSON response: {candidate}...") # Log partial response on error
        # Return default structure with "NaN" values if parsing fails
        return {
            "patent_id": "NaN",
            "name": "NaN",
            "address": "NaN",
            "description": "NaN",
            "date": "NaN"
        }

###############################################################################
# Prompt Builder
###############################################################################
def load_prompt_template() -> str:
    prompt_path = PROJECT_ROOT / "src" / "adding_variables" / "prompt.txt"
    if not prompt_path.exists():
        # Log a warning instead of raising an error, as per user request
        logging.warning(f"Prompt file 'prompt.txt' not found in working directory. Proceeding with empty prompt.")
        return "" # Return empty string if not found
    return prompt_path.read_text(encoding="utf-8")

def build_prompt(prompt_template: str, entry: str) -> str:
    return f"{prompt_template.strip()}\n{entry.strip()}"

###############################################################################
# Gemini API Call
###############################################################################
def gemini_api_call(prompt: str, temperature: float) -> Optional[dict]:
    client = genai.Client(api_key=API_KEY)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=FULL_MODEL_NAME,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                ),
            )
            if not response or not response.text:
                continue
            return {"text": response.text, "usage": response.usage_metadata}
        
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                # Extract retry-after value if available, otherwise use default
                retry_after = 10
                if "retry-after" in msg.lower():
                    try:
                        retry_after = int(re.search(r'retry-after: (\d+)', msg.lower()).group(1))
                    except:
                        pass
                logging.warning(f"Rate limit hit. Waiting {retry_after} seconds before retry {attempt+1}/{MAX_RETRIES}")
                time.sleep(retry_after)
            else:
                logging.warning(f"Error in attempt {attempt+1}/{MAX_RETRIES}: {msg}")
                time.sleep(2)
    
    logging.error(f"Failed to get response after {MAX_RETRIES} attempts")
    return None

###############################################################################
# Classify Single Entry with Retry Logic
###############################################################################
def classify_entry(entry: str, prompt_template: str, temperature: float) -> tuple:
    default_result = {
        "patent_id": "NaN",
        "name": "NaN",
        "address": "NaN",
        "description": "NaN",
        "date": "NaN"
    }
    for attempt in range(MAX_RETRIES):
        prompt = build_prompt(prompt_template, entry)
        result = gemini_api_call(prompt, temperature)
        if result and result["text"]:
            parsed_data = parse_response(result["text"])
            usage = result["usage"]
            ptk = getattr(usage, 'prompt_token_count', 0) or 0
            ctk = getattr(usage, 'candidates_token_count', 0) or 0
            ttk = getattr(usage, 'total_token_count', 0) or 0
            return parsed_data, ptk, ctk, ttk
        time.sleep(1)
    # Return default dictionary and zero tokens if all retries fail
    return default_result, 0, 0, 0

###############################################################################
# Main
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Classify entries using Gemini")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for Gemini API (default=0.0)")
    parser.add_argument("--max_workers", type=int, default=20, help="Max concurrent workers (default=20)")
    args = parser.parse_args()

    dataset_path = PROJECT_ROOT / "data" / "variable_dataset" / "imperial_patents.csv"
    df = pd.read_csv(dataset_path)

    if "entry" not in df.columns:
        raise KeyError("Column 'entry' not found in dataset.")

    prompt_template = load_prompt_template()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    # Define output columns based on expected JSON keys
    output_cols = ["patent_id", "name", "address", "description", "date"]
    for col in output_cols:
        df[col] = ""

    logging.info(f"Classifying {len(df)} entries using Gemini.")
    start_time = time.time()

    # Initialize global token counters
    global_tokens = defaultdict(int)
    processed_count = 0
    total_tasks = len(df)

    # Create a thread-safe dictionary to store results (dictionaries now)
    results_dict = {}

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(classify_entry, row["entry"], prompt_template, args.temperature): idx
                   for idx, row in df.iterrows()}

        for future in as_completed(futures):
            idx = futures[future]
            processed_count += 1
            progress = f"({processed_count}/{total_tasks})"
            try:
                result_dict, ptk, ctk, ttk = future.result()
                # Store result dictionary in thread-safe dictionary
                results_dict[idx] = result_dict

                # Update global token counters
                global_tokens['prompt'] += ptk
                global_tokens['candidate'] += ctk
                global_tokens['total'] += ttk

                # Log a summary of the result (e.g., patent_id or name if available)
                log_summary = result_dict.get("patent_id", result_dict.get("name", "N/A"))
                logging.info(f"[{idx}] {progress} → Processed: {log_summary} (Tokens: input={ptk}, candidate={ctk}, total={ttk})")
            except Exception as e:
                logging.error(f"[Row {idx}] {progress} Exception during processing: {e}")
                # Use default dictionary for errors
                results_dict[idx] = {col: "NaN" for col in output_cols}

            # Introduce a small delay to throttle the request rate
            time.sleep(0.1)

    # Update DataFrame after all processing is complete
    for idx, result_data in results_dict.items():
        for col in output_cols:
            # Ensure the key exists in the result dictionary, otherwise use NaN
            df.at[idx, col] = result_data.get(col, "NaN")

    # Save to a generic output file
    out_path = PROJECT_ROOT / "data" / "variable_dataset" / "classified_output.csv"
    df.to_csv(out_path, index=False)
    logging.info(f"Saved to: {out_path}")

    script_duration = time.time() - start_time
    logging.info(f"Finished in {format_duration(script_duration)}")

    # Log global token usage summary
    logging.info("")
    logging.info(" Global Usage Summary ".center(80, "="))
    logging.info(f"  Prompt Tokens:     {global_tokens['prompt']:,}")
    logging.info(f"  Candidate Tokens:  {global_tokens['candidate']:,}")
    logging.info(f"  Total Tokens:      {global_tokens['total']:,}")
    logging.info(f"  Total Script Time: {format_duration(script_duration)}")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()