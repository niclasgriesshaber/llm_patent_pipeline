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
FULL_MODEL_NAME = "gemini-2.5-flash-lite"  # Updated to match cleaning script
MAX_OUTPUT_TOKENS = 1000  # Updated as requested
MAX_WORKERS = 20

# Rate limiting configuration
MAX_REQUESTS_PER_MINUTE = 4000
MAX_TOKENS_PER_MINUTE = 4000000
REQUEST_WINDOW = 60  # seconds

# Conservative rate limiting (use 80% of limits to be safe)
SAFE_REQUESTS_PER_MINUTE = int(MAX_REQUESTS_PER_MINUTE * 0.8)  # 3200 requests/min
SAFE_TOKENS_PER_MINUTE = int(MAX_TOKENS_PER_MINUTE * 0.8)      # 3,200,000 tokens/min

# Enhanced retry configuration
MAX_RETRIES = 10
MAX_RATE_LIMIT_RETRIES = 10  # Separate retry limit for rate limit errors
BASE_DELAY = 5  # Increased from 1 to 5 seconds
MAX_DELAY = 30  # Maximum delay of 1 minute (60 seconds)
RATE_LIMIT_DELAY_MULTIPLIER = 5  # More aggressive multiplier for rate limit errors
RATE_LIMIT_BASE_DELAY = 30  # Start with 30 seconds for rate limit errors

# Global token tracking
total_input_tokens = 0
total_thought_tokens = 0
total_candidate_tokens = 0
total_failed_calls = 0  # Track failed API calls for cost monitoring
processing_completed = False  # Track if processing completed successfully

# Rate limiting tracking
request_times = []
token_usage_times = []
token_usage_amounts = []  # Track actual token amounts used
rate_limit_hits = 0  # Track rate limit hits for dynamic worker adjustment
consecutive_rate_limit_hits = 0  # Track consecutive rate limit hits

# Rate limit event tracking for better logging
rate_limit_events = []  # Store (timestamp, error_msg, tokens_used) for analysis

###############################################################################
# Utilities
###############################################################################
def format_duration(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def adjust_workers_if_needed():
    """Dynamically adjust MAX_WORKERS based on rate limit hits"""
    global MAX_WORKERS, rate_limit_hits, consecutive_rate_limit_hits
    
    # If we've hit rate limits multiple times, reduce workers more aggressively
    if rate_limit_hits >= 1 and MAX_WORKERS > 5:
        # More aggressive reduction: reduce by 50% or at least 5 workers
        reduction = max(5, MAX_WORKERS // 2)
        new_workers = max(5, MAX_WORKERS - reduction)
        if new_workers != MAX_WORKERS:
            logging.warning(f"Rate limits detected {rate_limit_hits} times. Reducing workers from {MAX_WORKERS} to {new_workers}")
            MAX_WORKERS = new_workers
            rate_limit_hits = 0  # Reset counter after adjustment
    
    # If we have consecutive rate limit hits, be even more aggressive
    if consecutive_rate_limit_hits >= 2 and MAX_WORKERS > 3:
        # Very aggressive reduction for consecutive hits
        new_workers = max(3, MAX_WORKERS // 3)
        if new_workers != MAX_WORKERS:
            logging.error(f"Consecutive rate limits detected {consecutive_rate_limit_hits} times. Aggressively reducing workers from {MAX_WORKERS} to {new_workers}")
            MAX_WORKERS = new_workers
            consecutive_rate_limit_hits = 0  # Reset counter after adjustment

def wait_for_rate_limit():
    """Wait if we're approaching rate limits"""
    current_time = time.time()
    
    # Clean old requests (older than 1 minute)
    global request_times, token_usage_times, token_usage_amounts
    request_times = [t for t in request_times if current_time - t < REQUEST_WINDOW]
    token_usage_times = [t for t in token_usage_times if current_time - t < REQUEST_WINDOW]
    token_usage_amounts = token_usage_amounts[-len(token_usage_times):]  # Keep only recent amounts
    
    # Check request rate limit (use conservative limit)
    if len(request_times) >= SAFE_REQUESTS_PER_MINUTE:
        sleep_time = REQUEST_WINDOW - (current_time - request_times[0])
        if sleep_time > 0:
            logging.warning(f"Request rate limit approaching ({len(request_times)}/{SAFE_REQUESTS_PER_MINUTE}). Waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
    
    # Check token rate limit (use actual token amounts)
    if len(token_usage_amounts) > 0:
        total_tokens_in_window = sum(token_usage_amounts)
        
        if total_tokens_in_window >= SAFE_TOKENS_PER_MINUTE:
            sleep_time = REQUEST_WINDOW - (current_time - token_usage_times[0])
            if sleep_time > 0:
                logging.warning(f"Token rate limit approaching ({total_tokens_in_window:,}/{SAFE_TOKENS_PER_MINUTE:,} tokens). Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
    
    # Log current usage for monitoring
    if len(request_times) > 0 or len(token_usage_amounts) > 0:
        requests_in_window = len(request_times)
        tokens_in_window = sum(token_usage_amounts) if token_usage_amounts else 0
        logging.debug(f"Current usage: {requests_in_window}/{SAFE_REQUESTS_PER_MINUTE} requests, {tokens_in_window:,}/{SAFE_TOKENS_PER_MINUTE:,} tokens")

def calculate_backoff_delay(attempt: int, is_rate_limit: bool = False) -> float:
    """
    Calculate backoff delay with exponential backoff and jitter.
    
    Args:
        attempt: Current attempt number (0-based)
        is_rate_limit: Whether this is a rate limit error (uses longer delays)
    
    Returns:
        Delay in seconds
    """
    import random
    
    # Base exponential backoff
    if is_rate_limit:
        # For rate limit errors, use much longer delays starting from higher base
        delay = RATE_LIMIT_BASE_DELAY * (RATE_LIMIT_DELAY_MULTIPLIER ** attempt)
    else:
        # For other errors, use standard exponential backoff
        delay = BASE_DELAY * (2 ** attempt)
    
    # Cap the delay at maximum
    delay = min(delay, MAX_DELAY)
    
    # Add jitter (±25% random variation) to prevent thundering herd
    jitter = delay * 0.25 * random.uniform(-1, 1)
    delay += jitter
    
    # Ensure minimum delay
    delay = max(delay, 1.0)
    
    return delay

def create_processing_log(logs_dir: Path, filestem: str, csv_name: str, row_count: int, 
                         processing_time: float, max_workers: int, api_failures: list) -> None:
    """Create a JSON log file with processing information."""
    log_file = logs_dir / f"{filestem}_variable_extraction_log.json"
    
    # Count API failures
    api_fail_count = sum(api_failures)
    
    # Calculate rate limit statistics
    rate_limit_event_count = len(rate_limit_events)
    max_consecutive_rate_limit_hits = max([event['consecutive_hits'] for event in rate_limit_events]) if rate_limit_events else 0
    
    log_data = {
        "file_name": csv_name,
        "model": FULL_MODEL_NAME,
        "number_of_rows": row_count,
        "total_input_tokens": total_input_tokens,
        "total_thought_tokens": total_thought_tokens,
        "total_candidate_tokens": total_candidate_tokens,
        "total_tokens": total_input_tokens + total_thought_tokens + total_candidate_tokens,
        "total_failed_calls": total_failed_calls,
        "api_failures_count": api_fail_count,
        "processing_time_seconds": round(processing_time, 2),
        "max_workers": max_workers,
        "processing_completed": processing_completed,
        "rate_limit_events_count": rate_limit_event_count,
        "max_consecutive_rate_limit_hits": max_consecutive_rate_limit_hits,
        "final_worker_count": MAX_WORKERS,
        "rate_limit_events": rate_limit_events
    }
    
    try:
        with log_file.open("w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Processing log saved to: {log_file}")
    except Exception as e:
        logging.error(f"Failed to write processing log {log_file}: {e}")
    
    # Also create a separate rate limit analysis file if there were rate limit events
    if rate_limit_events:
        rate_limit_analysis_file = logs_dir / f"{filestem}_rate_limit_analysis.json"
        try:
            with rate_limit_analysis_file.open("w", encoding="utf-8") as f:
                json.dump({
                    "rate_limit_events": rate_limit_events,
                    "summary": {
                        "total_rate_limit_events": rate_limit_event_count,
                        "max_consecutive_hits": max_consecutive_rate_limit_hits,
                        "final_worker_count": MAX_WORKERS,
                        "processing_time_seconds": round(processing_time, 2)
                    }
                }, f, indent=2, ensure_ascii=False)
            logging.info(f"Rate limit analysis saved to: {rate_limit_analysis_file}")
        except Exception as e:
            logging.error(f"Failed to write rate limit analysis {rate_limit_analysis_file}: {e}")

def parse_response(text: str) -> dict:
    # Strip away any backticks that might wrap the text
    candidate = re.sub(r"^`+|`+$", "", text.strip())

    try:
        # Attempt to parse the cleaned text as JSON
        parsed_data = json.loads(candidate)
        
        # Clean up patent_id to remove float residuals
        if "patent_id" in parsed_data:
            patent_id = parsed_data["patent_id"]
            if isinstance(patent_id, (int, float)):
                # Convert to string and remove .0 if it's a float
                patent_id_str = str(int(patent_id)) if patent_id == int(patent_id) else str(patent_id)
                parsed_data["patent_id"] = patent_id_str
            elif isinstance(patent_id, str):
                # Remove .0 from string if present
                parsed_data["patent_id"] = re.sub(r'\.0$', '', patent_id)
        
        return parsed_data
    except json.JSONDecodeError:
        logging.warning(f"Failed to parse JSON response: {candidate}...") # Log partial response on error
        # Return default structure with "NaN" values if parsing fails
        return {
            "patent_id": "NaN",
            "name": "NaN",
            "location": "NaN",
            "description": "NaN",
            "date": "NaN"
        }

###############################################################################
# Prompt Builder
###############################################################################
def load_prompt_template() -> str:
    prompt_path = PROJECT_ROOT / "src" / "03_variable_extraction" / "prompt.txt"
    if not prompt_path.exists():
        # Log a warning instead of raising an error, as per user request
        logging.warning(f"Prompt file 'prompt.txt' not found in src/03_variable_extraction/. Proceeding with empty prompt.")
        return "" # Return empty string if not found
    return prompt_path.read_text(encoding="utf-8")

def build_prompt(prompt_template: str, entry: str) -> str:
    return f"{prompt_template.strip()}\n{entry.strip()}"

###############################################################################
# Enhanced Gemini API Call with Rate Limiting and Retry Logic
###############################################################################
def call_llm(entry: str, prompt_template: str) -> tuple[dict, dict, bool]:
    global total_input_tokens, total_thought_tokens, total_candidate_tokens, total_failed_calls
    global request_times, token_usage_times, token_usage_amounts, rate_limit_events, consecutive_rate_limit_hits, rate_limit_hits
    
    client = genai.Client(api_key=API_KEY)
    prompt = f"{prompt_template}\n{entry.strip()}"
    
    # Configure model-specific settings
    config_args = {
        "temperature": 0.0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_mime_type": "application/json",
    }
    
    # For gemini-2.5 models, set thinking_config with minimum thinking_budget
    if "2.5" in FULL_MODEL_NAME:
        if "lite" in FULL_MODEL_NAME:
            # For lite model: no thinking, minimal output tokens
            config_args["max_output_tokens"] = 1000
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
    
    # Separate retry counters
    regular_attempts = 0
    rate_limit_attempts = 0
    
    # Reset consecutive rate limit hits for this specific request
    request_consecutive_hits = 0
    
    while True:
        try:
            # Check rate limits before making request
            wait_for_rate_limit()
            
            # Adjust workers if we've hit rate limits too often
            adjust_workers_if_needed()
            
            # Record request time
            request_times.append(time.time())
            
            response = client.models.generate_content(
                model=FULL_MODEL_NAME,
                contents=[prompt],
                config=config,
            )
            
            # Reset consecutive rate limit hits on successful request
            consecutive_rate_limit_hits = 0
            
            if not response or not response.text:
                logging.warning(f"Empty response from {FULL_MODEL_NAME}")
                regular_attempts += 1
                if regular_attempts < MAX_RETRIES:
                    delay = calculate_backoff_delay(regular_attempts - 1, is_rate_limit=False)
                    logging.warning(f"Empty response retry... (attempt {regular_attempts}/{MAX_RETRIES}) in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                else:
                    total_failed_calls += 1
                    return {"patent_id": "NaN", "name": "NaN", "location": "NaN", "description": "NaN", "date": "NaN"}, {}, True
            
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
                
                # Record token usage time and amount for rate limiting
                token_usage_times.append(time.time())
                token_usage_amounts.append(ptk + ctk + ttk)  # Append total tokens for rate limiting
                
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
            
            # Parse the response
            parsed_data = parse_response(text)
            
            # Check if parsing was successful (not all fields are NaN)
            if not all(parsed_data.get(col, "NaN") == "NaN" for col in ["patent_id", "name", "location", "description", "date"]):
                return parsed_data, token_info, False  # Third parameter indicates no API failure
            else:
                logging.warning(f"All fields are NaN after parsing response from {FULL_MODEL_NAME}: '{text}'")
                regular_attempts += 1
                if regular_attempts < MAX_RETRIES:
                    delay = calculate_backoff_delay(regular_attempts - 1, is_rate_limit=False)
                    logging.warning(f"All NaN fields retry... (attempt {regular_attempts}/{MAX_RETRIES}) in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                else:
                    total_failed_calls += 1
                    return parsed_data, token_info, True  # Third parameter indicates API failure after max attempts
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if this is a rate limit error
            is_rate_limit_error = (
                "429" in error_msg or 
                "rate limit" in error_msg.lower() or 
                "resource exhausted" in error_msg.lower()
            )
            
            # Check if this is any API failure that should be retried
            is_api_failure = (
                is_rate_limit_error or
                "timeout" in error_msg.lower() or
                "connection" in error_msg.lower() or
                "network" in error_msg.lower()
            )
            
            if is_rate_limit_error:
                # Handle rate limit errors with separate counter
                rate_limit_attempts += 1
                request_consecutive_hits += 1
                consecutive_rate_limit_hits += 1  # Increment global consecutive rate limit hits
                rate_limit_hits += 1  # Increment global rate limit hits
                
                # Log rate limit event for analysis
                rate_limit_events.append({
                    'timestamp': time.time(),
                    'error_msg': error_msg,
                    'attempt': rate_limit_attempts,
                    'consecutive_hits': request_consecutive_hits,
                    'current_workers': MAX_WORKERS
                })
                
                if rate_limit_attempts < MAX_RATE_LIMIT_RETRIES:
                    delay = calculate_backoff_delay(rate_limit_attempts - 1, is_rate_limit=True)
                    logging.error(f"RATE LIMIT ERROR for {FULL_MODEL_NAME} (rate limit attempt {rate_limit_attempts}/{MAX_RATE_LIMIT_RETRIES}): {error_msg}")
                    logging.error(f"Request consecutive hits: {request_consecutive_hits}, Global consecutive hits: {consecutive_rate_limit_hits}")
                    logging.error(f"Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                    continue
                else:
                    logging.error(f"MAX RATE LIMIT RETRIES REACHED for {FULL_MODEL_NAME} after {rate_limit_attempts} attempts")
                    logging.error(f"Final error: {error_msg}")
                    total_failed_calls += 1
                    return {"patent_id": "NaN", "name": "NaN", "location": "NaN", "description": "NaN", "date": "NaN"}, {}, True
                    
            elif is_api_failure:
                # Handle other API failures with regular counter
                regular_attempts += 1
                if regular_attempts < MAX_RETRIES:
                    delay = calculate_backoff_delay(regular_attempts - 1, is_rate_limit=False)
                    logging.warning(f"API failure for {FULL_MODEL_NAME} (attempt {regular_attempts}/{MAX_RETRIES}): {error_msg}")
                    logging.warning(f"Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                    continue
                else:
                    logging.error(f"MAX API RETRIES REACHED for {FULL_MODEL_NAME} after {regular_attempts} attempts")
                    logging.error(f"Final error: {error_msg}")
                    total_failed_calls += 1
                    return {"patent_id": "NaN", "name": "NaN", "location": "NaN", "description": "NaN", "date": "NaN"}, {}, True
            else:
                # Non-API failure - log and return immediately
                logging.error(f"Non-API error for {FULL_MODEL_NAME}: {error_msg}")
                
                # Check for specific error types
                if "thinking" in error_msg.lower():
                    logging.error(f"Thinking budget configuration issue for {FULL_MODEL_NAME}")
                elif "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                    logging.error(f"Model {FULL_MODEL_NAME} not found or not available")
                
                total_failed_calls += 1
                return {"patent_id": "NaN", "name": "NaN", "location": "NaN", "description": "NaN", "date": "NaN"}, {}, True
    
    # This should never be reached, but just in case
    total_failed_calls += 1
    return {"patent_id": "NaN", "name": "NaN", "location": "NaN", "description": "NaN", "date": "NaN"}, {}, True

###############################################################################
# Classify Single Entry with Enhanced Retry Logic
###############################################################################
def classify_entry(entry: str, prompt_template: str, temperature: float) -> tuple:
    global total_input_tokens, total_thought_tokens, total_candidate_tokens
    
    prompt = build_prompt(prompt_template, entry)
    
    # Use the enhanced call_llm function
    result_dict, token_info, is_api_failure = call_llm(entry, prompt_template)
    
    # Extract token counts from token_info
    ptk = token_info.get('prompt_tokens', 0)
    ctk = token_info.get('candidate_tokens', 0)
    ttk = token_info.get('thoughts_tokens', 0)
    
    return result_dict, ptk, ctk, ttk, is_api_failure

###############################################################################
# Main
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Classify entries using Gemini")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV filename located in data/cleaned_csvs (e.g., Patentamt_1889_cleaned.csv)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for Gemini API (default=0.0)")
    parser.add_argument("--max_workers", type=int, default=20, help="Max concurrent workers (default=20)")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite", 
                       choices=["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"],
                       help="Model to use for LLM processing (default: gemini-2.5-flash-lite)")
    args = parser.parse_args()

    # Update global model name and max workers based on command line arguments
    global FULL_MODEL_NAME, MAX_WORKERS
    FULL_MODEL_NAME = args.model
    MAX_WORKERS = args.max_workers

    input_filename = args.csv
    input_path = PROJECT_ROOT / "data" / "02_dataset_cleaning" / "cleaned_csvs" / input_filename
    output_dir = PROJECT_ROOT / "data" / "03_variable_extraction" / "cleaned_with_variables_csvs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle both old and new naming conventions
    # If input is "Patentamt_XXXX_cleaned.csv", output should be "Patentamt_XXXX_cleaned_with_variables.csv"
    # If input is "Patentamt_XXXX.csv", output should be "Patentamt_XXXX_with_variables.csv"
    if input_filename.endswith("_cleaned.csv"):
        # New naming convention: Patentamt_XXXX_cleaned.csv -> Patentamt_XXXX_cleaned_with_variables.csv
        output_filename = input_filename.replace("_cleaned.csv", "_cleaned_with_variables.csv")
    else:
        # Old naming convention: Patentamt_XXXX.csv -> Patentamt_XXXX_with_variables.csv
        output_filename = input_filename.replace(".csv", "_with_variables.csv")
    
    output_path = output_dir / output_filename
    error_path = output_dir / f"error_{input_filename.replace('.csv', '')}.txt"
    summary_stem = os.path.splitext(input_filename)[0]
    summary_path = output_dir / f"summary_{summary_stem}.txt"

    df = pd.read_csv(input_path)

    if "entry" not in df.columns:
        raise KeyError("Column 'entry' not found in dataset.")
    if "id" not in df.columns:
        raise KeyError("Column 'id' not found in dataset. The input CSV must have an 'id' column.")
    has_page = "page" in df.columns

    prompt_template = load_prompt_template()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    # Define output columns based on expected JSON keys
    output_cols = ["patent_id", "name", "location", "description", "date"]
    for col in output_cols:
        df[col] = ""

    logging.info(f"Classifying {len(df)} entries using Gemini.")
    logging.info(f"Model: {FULL_MODEL_NAME}")
    logging.info(f"Max Workers: {MAX_WORKERS}")
    logging.info(f"Conservative rate limits: {SAFE_REQUESTS_PER_MINUTE}/{MAX_REQUESTS_PER_MINUTE} requests/min, {SAFE_TOKENS_PER_MINUTE:,}/{MAX_TOKENS_PER_MINUTE:,} tokens/min")
    start_time = time.time()

    processed_count = 0
    total_tasks = len(df)

    # Create a thread-safe dictionary to store results (dictionaries now)
    results_dict = {}
    api_failures = [False] * len(df)  # Track API failures per row
    llm_api_failures = 0  # When the LLM API call itself fails
    successful_calls_with_missing_data = 0  # When API succeeds but some fields are NaN
    api_failure_rows = []  # Rows where LLM API call failed
    missing_data_rows = []  # Rows where API succeeded but some variables are NaN

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(classify_entry, row["entry"], prompt_template, args.temperature): idx
                       for idx, row in df.iterrows()}

            for future in as_completed(futures):
                idx = futures[future]
                processed_count += 1
                progress = f"({processed_count}/{total_tasks})"
                try:
                    result_dict, ptk, ctk, ttk, is_api_failure = future.result()
                    results_dict[idx] = result_dict
                    api_failures[idx] = is_api_failure
                    
                    log_summary = result_dict.get("patent_id", result_dict.get("name", "N/A"))
                    logging.info(f"[{idx}] {progress} → Processed: {log_summary} (Tokens: input={ptk}, candidate={ctk}, thoughts={ttk})")
                    
                    # Check if all variables are NaN (LLM API call failed)
                    if all(result_dict.get(col, "NaN") == "NaN" for col in output_cols):
                        failed_id = df.at[idx, "id"]
                        failed_page = df.at[idx, "page"] if has_page else "N/A"
                        api_failure_rows.append((failed_id, failed_page))
                        llm_api_failures += 1
                        logging.error(f"LLM API call failed for id: {failed_id}, page: {failed_page}")
                    # Check if some variables are NaN (API succeeded but missing data)
                    elif any(result_dict.get(col, "NaN") == "NaN" for col in output_cols):
                        successful_calls_with_missing_data += 1
                        failed_id = df.at[idx, "id"]
                        failed_page = df.at[idx, "page"] if has_page else "N/A"
                        missing_data_rows.append((failed_id, failed_page))
                        logging.warning(f"Successful LLM call with missing data for id: {failed_id}, page: {failed_page} (some variables NaN)")
                except Exception as e:
                    logging.error(f"[Row {idx}] {progress} Exception during processing: {e}")
                    results_dict[idx] = {col: "NaN" for col in output_cols}
                    api_failures[idx] = True
                    llm_api_failures += 1
                    failed_id = df.at[idx, "id"]
                    failed_page = df.at[idx, "page"] if has_page else "N/A"
                    api_failure_rows.append((failed_id, failed_page))
                    logging.error(f"LLM API call failed for id: {failed_id}, page: {failed_page}")
                time.sleep(0.1)

        # Mark processing as completed
        global processing_completed
        processing_completed = True
        
    except Exception as e:
        logging.error(f"Processing failed with error: {e}")
        # Mark processing as incomplete
        processing_completed = False
        
        # Write warning to log file
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        warning_path = logs_dir / f"{summary_stem}_WARNING.txt"
        
        with open(warning_path, "w", encoding="utf-8") as f:
            f.write("WARNING: PROCESSING INCOMPLETE\n")
            f.write("=" * 50 + "\n")
            f.write("THE SCRIPT FAILED BEFORE COMPLETING ALL ROWS.\n")
            f.write("PLEASE CHECK THE LOGS AND RESTART THE PROCESS.\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Failed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logging.error(f"WARNING: Processing incomplete. Check {warning_path} for details.")
        raise  # Re-raise the exception to exit with error code

    # Update DataFrame after all processing is complete
    for idx, result_data in results_dict.items():
        for col in output_cols:
            # Ensure the key exists in the result dictionary, otherwise use NaN
            df.at[idx, col] = result_data.get(col, "NaN")

    # Add successful_variable_extraction column
    # 1 if all new variables are non-NaN, 0 if any are NaN
    df['successful_variable_extraction'] = 0  # Default to 0
    for idx in df.index:
        # Check if all the newly created variables are non-NaN
        all_successful = all(df.at[idx, col] != "NaN" for col in output_cols)
        df.at[idx, 'successful_variable_extraction'] = 1 if all_successful else 0

    # Add variable_API_fail column
    df['variable_API_fail'] = "0"  # Default to 0
    for idx, is_api_failure in enumerate(api_failures):
        if is_api_failure:
            df.at[idx, 'variable_API_fail'] = "1"

    # Save to the output file in cleaned_with_variables_csvs
    df.to_csv(output_path, index=False)
    logging.info(f"Saved CSV to: {output_path}")
    
    # Save to XLSX format in check_variable_extraction_xlsx
    xlsx_output_dir = PROJECT_ROOT / "data" / "03_variable_extraction" / "check_variable_extraction_xlsx"
    xlsx_output_dir.mkdir(parents=True, exist_ok=True)
    xlsx_output_path = xlsx_output_dir / f"{output_filename.replace('.csv', '.xlsx')}"
    df.to_excel(xlsx_output_path, index=False)
    logging.info(f"Saved XLSX to: {xlsx_output_path}")

    # Create logs directory if it doesn't exist
    logs_dir = xlsx_output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Save runtime summary to logs folder with improved terminology
    summary_filename = f"{output_filename.replace('.csv', '')}.txt"
    summary_path = logs_dir / summary_filename
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Total rows processed: {len(df)}\n")
        f.write(f"LLM API call failures: {llm_api_failures}\n")
        f.write(f"Successful LLM calls with missing data: {successful_calls_with_missing_data}\n")
        f.write(f"Total entries with incomplete data: {llm_api_failures + successful_calls_with_missing_data}\n\n")
        
        if api_failure_rows:
            f.write("LLM API call failures - rows where API call failed (id, page):\n")
            for fid, fpage in api_failure_rows:
                f.write(f"id: {fid}, page: {fpage}\n")
            f.write("\n")
        
        if missing_data_rows:
            f.write("Successful LLM calls with missing data - rows where API succeeded but some variables are NaN (id, page):\n")
            for fid, fpage in missing_data_rows:
                f.write(f"id: {fid}, page: {fpage}\n")
            f.write("\n")
    
    logging.info(f"Runtime summary saved to: {summary_path}")

    script_duration = time.time() - start_time
    logging.info(f"Finished in {format_duration(script_duration)}")

    # Log global token usage summary with improved terminology
    logging.info("")
    logging.info(" Global Usage Summary ".center(80, "="))
    logging.info(f"  Prompt Tokens:     {total_input_tokens:,}")
    logging.info(f"  Candidate Tokens:  {total_candidate_tokens:,}")
    logging.info(f"  Total Tokens:      {total_input_tokens + total_candidate_tokens:,}")
    logging.info(f"  Total Script Time: {format_duration(script_duration)}")
    logging.info(f"  LLM API Call Failures: {llm_api_failures}")
    logging.info(f"  Successful Calls with Missing Data: {successful_calls_with_missing_data}")
    logging.info(f"  Total Entries with Incomplete Data: {llm_api_failures + successful_calls_with_missing_data}")
    logging.info("=" * 80)

    # Create and save processing log
    create_processing_log(logs_dir, summary_stem, input_filename, len(df), script_duration, args.max_workers, api_failures)

    # Log token usage summary
    logging.info(f"Token usage summary: prompt={total_input_tokens:,}, candidate={total_candidate_tokens:,}, thoughts={total_thought_tokens:,}, total={total_input_tokens + total_thought_tokens + total_candidate_tokens:,}")
    logging.info(f"Failed API calls: {total_failed_calls}")
    
    # Log API failure summary
    api_fail_count = sum(api_failures)
    logging.info(f"Rows with API failures (10+ attempts): {api_fail_count}")
    if api_fail_count > 0:
        logging.warning(f"Note: {api_fail_count} rows had API failures after 10 attempts and were set to 'NaN'")
    
    # Log rate limit statistics
    rate_limit_event_count = len(rate_limit_events)
    if rate_limit_event_count > 0:
        max_consecutive_hits = max([event['consecutive_hits'] for event in rate_limit_events])
        logging.warning(f"Rate limit events: {rate_limit_event_count} total events")
        logging.warning(f"Max consecutive rate limit hits: {max_consecutive_hits}")
        logging.warning(f"Final worker count: {MAX_WORKERS} (started with {args.max_workers})")
    else:
        logging.info(f"No rate limit events occurred - processing completed smoothly")
    
    # Log conservative rate limiting info
    logging.info(f"Used conservative rate limits: {SAFE_REQUESTS_PER_MINUTE}/{MAX_REQUESTS_PER_MINUTE} requests/min, {SAFE_TOKENS_PER_MINUTE:,}/{MAX_TOKENS_PER_MINUTE:,} tokens/min")

if __name__ == "__main__":
    main()