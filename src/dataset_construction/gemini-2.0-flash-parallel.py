#!/usr/bin/env python3
import os
import sys
import re
import json
import time
import argparse
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Any, Union, Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd

import google.genai as genai
from google.genai import types
from google.api_core import exceptions as google_exceptions

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CSVS_DIR = DATA_DIR / "csvs"
PROMPT_SRC_DIR = PROJECT_ROOT / "src" / "dataset_construction"
PDF_SRC_DIR = DATA_DIR / "pdfs" / "patent_pdfs"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

# --- Hardcoded Prompt File ---
PROMPT_FILE_PATH = PROMPT_SRC_DIR / "prompt.txt"

# --- Load Environment Variables ---
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    sys.exit("Error: GOOGLE_API_KEY not found.")

# --- Configuration based on original working script & user request ---
MODEL_NAME = "gemini-2.0-flash" #"gemini-2.5-pro-exp-03-25" #"gemini-2.0-flash" # gemini-2.5-pro-exp-03-25 # gemini-2.5-pro-preview-03-25
MAX_OUTPUT_TOKENS = 8192
MAX_RETRIES = 3
BACKOFF_SLEEP_SECONDS = 30

# --- Utilities ---
def format_duration(seconds: float) -> str:
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    return f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"

def parse_json_str(response_text: str) -> Any:
    # Using the exact parsing logic from the original script
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
        if candidate.startswith('\ufeff'): candidate = candidate[1:]
    else:
        candidate = response_text.strip().strip("`")
        if candidate.startswith('\ufeff'): candidate = candidate[1:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON content: {e}\nContent snippet: {candidate[:200]}...") from e

# --- Core Logic: API Call ---
def gemini_api_call(prompt: str, pil_image: Image.Image, temperature: float) -> Tuple[Optional[dict], str, bool]:
    """Call Gemini with up to 3 retries; if 429 occurs, sleep briefly and retry."""
    try:
        # Using original client initialization
        client = genai.Client(api_key=API_KEY)
    except AttributeError:
         logging.critical("FATAL: 'genai.Client' not found. Library version/installation issue?", exc_info=True)
         return (None, "FATAL: genai.Client not found in library.", False)
    except Exception as client_e:
         logging.critical(f"FATAL: Failed to initialize Gemini Client: {client_e}", exc_info=True)
         return (None, f"FATAL: Client init failed: {client_e}", False)

    error_msg = ""
    is_rate_limit = False
    tmp_file_path = None
    file_upload = None # Declare file_upload here to handle potential assignment in try

    for attempt in range(MAX_RETRIES):
        # file_upload = None # Reset inside loop if needed, but better outside with check
        try:
            # Using original tempfile handling
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_file_path_str = tmp.name
                tmp_file_path = Path(tmp_file_path_str)
                pil_image.save(tmp_file_path_str, "PNG")

            try:
                # Using original client.files.upload
                file_upload = client.files.upload(path=tmp_file_path_str)
                logging.debug(f"File uploaded via client.files.upload: {file_upload.uri}")
            except Exception as upload_err:
                error_msg = f"File upload failed: {upload_err}"
                logging.warning(f"Attempt {attempt+1}/{MAX_RETRIES}: {error_msg}")
                if tmp_file_path and tmp_file_path.exists():
                    try: tmp_file_path.unlink()
                    except OSError: pass
                tmp_file_path = None
                if attempt < MAX_RETRIES - 1: time.sleep(2); continue
                else: break

            # *** API Call - Using original structure ***
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    types.Part.from_uri(
                        file_uri=file_upload.uri,
                        mime_type=file_upload.mime_type,
                    ),
                    prompt # The prompt string is passed here
                ],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                )
            )

            if not response or not response.text:
                error_msg = "API returned empty response or no text"
                logging.warning(f"Attempt {attempt+1}/{MAX_RETRIES}: {error_msg}")
                if response and response.prompt_feedback and response.prompt_feedback.block_reason:
                     error_msg += f" (Block Reason: {response.prompt_feedback.block_reason})"
                     logging.warning(f"Block Reason: {response.prompt_feedback.block_reason}")
                if attempt < MAX_RETRIES - 1: time.sleep(1); continue
                else: break

            usage = response.usage_metadata
            # Returning dict structure matching original process_page expectation
            return ({"text": response.text, "usage": usage}, "", False)

        except google_exceptions.ResourceExhausted as e:
            is_rate_limit = True
            error_msg = f"API Error (Rate Limit): {e}"
            logging.warning(f"Attempt {attempt+1}/{MAX_RETRIES}: Rate limit error! Sleeping {BACKOFF_SLEEP_SECONDS}s...")
            if attempt < MAX_RETRIES - 1: time.sleep(BACKOFF_SLEEP_SECONDS)

        except Exception as e:
            msg = str(e)
            error_msg = f"API Error (General): {type(e).__name__} - {msg}"
            # Using original rate limit detection logic
            if "429" in msg or "rate limit" in msg.lower() or "resource has been exhausted" in msg.lower():
                is_rate_limit = True
                logging.warning(f"Attempt {attempt+1}/{MAX_RETRIES}: Rate limit detected! Sleeping {BACKOFF_SLEEP_SECONDS}s...")
                if attempt < MAX_RETRIES - 1: time.sleep(BACKOFF_SLEEP_SECONDS)
            else:
                # Log full exception for unexpected errors in original style
                logging.warning(f"Attempt {attempt+1}/{MAX_RETRIES}: {error_msg}", exc_info=True if not is_rate_limit else False)
                if attempt < MAX_RETRIES - 1: time.sleep(2)

        finally:
            # Using original finally block cleanup
            if tmp_file_path and tmp_file_path.exists():
                try:
                    tmp_file_path.unlink()
                    logging.debug(f"Deleted temp file: {tmp_file_path}")
                except OSError as del_err:
                    logging.warning(f"Failed to delete temp file {tmp_file_path}: {del_err}")
            tmp_file_path = None

    final_error_msg = f"API failed after {MAX_RETRIES} attempts: {error_msg}"
    logging.error(final_error_msg)
    return (None, final_error_msg, is_rate_limit)


# --- Page Processing ---
def process_page(page_idx: int,
                 png_path: Path,
                 prompt_text: str, # Renamed parameter for clarity
                 temperature: float,
                 json_dir: Path) -> dict:
    """Processes a single page using the original script's logic."""
    result_info = {
        "page_idx": page_idx, "prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0,
        "success": False, "error_msg": "", "error_type": None, "api_failures": 0,
        "rate_limit_failures": 0, "parse_failures": 0, "last_parse_error": ""
    }
    logging.info(f"[Worker p.{page_idx:04d}] Processing {png_path.name}...")
    pil_image = None

    try:
        pil_image = Image.open(png_path)
        w, h = pil_image.size
        logging.info(f"[Worker p.{page_idx:04d}] PNG info: {w}x{h}") # DPI removed

        # Use the passed prompt_text
        initial_result, error, is_rate_limit = gemini_api_call(prompt_text, pil_image, temperature)

        # Track API failures (original logic didn't explicitly track per attempt inside process_page)
        # We'll rely on the return value indicating final failure
        if not initial_result:
            result_info["error_msg"] = error
            result_info["error_type"] = "api"
            result_info["api_failures"] = MAX_RETRIES # Assume max if final failure
            if is_rate_limit: result_info["rate_limit_failures"] = MAX_RETRIES # Assume max if final failure
            return result_info

        # Access usage metadata (original structure)
        usage = initial_result["usage"]
        ptk = getattr(usage, 'prompt_token_count', 0) or 0
        ctk = getattr(usage, 'candidates_token_count', 0) or 0
        ttk = getattr(usage, 'total_token_count', 0) or 0
        result_info.update({"prompt_tokens": ptk, "candidate_tokens": ctk, "total_tokens": ttk})
        resp_text = initial_result["text"]
        logging.info(f"[Worker p.{page_idx:04d}] Initial API usage -> input={ptk}, candidate={ctk}, total={ttk}")

        parse_attempts = 0
        parsed = None
        while parse_attempts < MAX_RETRIES:
            try:
                parsed = parse_json_str(resp_text)
                logging.info(f"[Worker p.{page_idx:04d}] Parsed JSON successfully (attempt {parse_attempts+1}).")
                break
            except ValueError as ve:
                parse_attempts += 1
                result_info["parse_failures"] += 1
                parse_error_msg = str(ve)
                result_info["last_parse_error"] = parse_error_msg
                logging.error(f"[Worker p.{page_idx:04d}] JSON parse error ({parse_attempts}/{MAX_RETRIES}): {parse_error_msg}")
                # Original logging of raw response on error
                logging.error(f"[Worker p.{page_idx:04d}] RAW RESPONSE:\n{'-'*60}\n{resp_text}\n{'-'*60}")

                if parse_attempts >= MAX_RETRIES:
                    logging.error(f"[Worker p.{page_idx:04d}] JSON parse failed after {MAX_RETRIES} attempts.")
                    result_info["error_msg"] = f"JSON parse failed after {MAX_RETRIES} attempts: {parse_error_msg}"
                    result_info["error_type"] = "parse"
                    break # Exit loop

                logging.info(f"[Worker p.{page_idx:04d}] Re-calling API for parse retry ({parse_attempts+1}/{MAX_RETRIES})...")
                # Use the same prompt_text for recall
                recall_res, recall_error, recall_rate_limit = gemini_api_call(prompt_text, pil_image, temperature)
                # Track total API attempts across retries (approximate)
                result_info["api_failures"] += 1 # Increment for the recall attempt
                if recall_rate_limit: result_info["rate_limit_failures"] += 1 # Increment if recall hit rate limit

                if not recall_res:
                    result_info["error_msg"] = f"API failed during parse retry: {recall_error}"
                    result_info["error_type"] = "api"
                    # Don't update api_failures to MAX_RETRIES here, just note the failure
                    logging.error(f"[Worker p.{page_idx:04d}] {result_info['error_msg']}")
                    break # Exit loop

                usage2 = recall_res["usage"]
                ptk2 = getattr(usage2, 'prompt_token_count', 0) or 0
                ctk2 = getattr(usage2, 'candidates_token_count', 0) or 0
                ttk2 = getattr(usage2, 'total_token_count', 0) or 0
                # Accumulate tokens (original logic)
                result_info["prompt_tokens"] += ptk2
                result_info["candidate_tokens"] += ctk2
                result_info["total_tokens"] += ttk2
                resp_text = recall_res["text"] # Use the new response text
                logging.info(f"[Worker p.{page_idx:04d}] Recall API usage -> input={ptk2}, candidate={ctk2}, total={ttk2}")
                # Loop continues to try parsing the new resp_text

        if not parsed:
            # If parsing failed after all attempts
             if not result_info["error_msg"]: # Ensure error message is set
                 result_info["error_msg"] = f"Parsing failed after {MAX_RETRIES} attempts."
                 result_info["error_type"] = "parse"
             # Close image before returning on failure
             if pil_image:
                 try: pil_image.close()
                 except Exception: pass # Ignore close errors on failure path
             return result_info

        # --- Save Successfully Parsed JSON ---
        json_dir.mkdir(parents=True, exist_ok=True)
        json_out = json_dir / f"{png_path.stem}.json"
        try:
            # Save JSON (original logic)
            with json_out.open("w", encoding="utf-8") as jf:
                # The original script didn't reorder keys *before* saving JSON
                json.dump(parsed, jf, indent=2, ensure_ascii=False)
            result_info["success"] = True
            # No specific log message for saving in original, adding a debug one
            logging.debug(f"[Worker p.{page_idx:04d}] Successfully saved JSON to {json_out.relative_to(PROJECT_ROOT)}")
        except Exception as e_save:
            logging.error(f"[Worker p.{page_idx:04d}] Failed to save JSON {json_out}: {e_save}")
            result_info.update({"error_msg": f"Failed to save JSON: {e_save}", "error_type": "save", "success": False})

    except FileNotFoundError:
        logging.error(f"[Worker p.{page_idx:04d}] File not found: {png_path}")
        result_info.update({"error_msg": f"File not found: {png_path.name}", "error_type": "file"})
    except Exception as e_outer:
        # Use original logging.exception for unexpected errors
        logging.exception(f"[Worker p.{page_idx:04d}] Unexpected error: {e_outer}")
        result_info.update({"error_msg": f"Unexpected error: {e_outer}", "error_type": "other"})
    finally:
        # Original finally block for closing image
        if pil_image:
            try: pil_image.close()
            except Exception as close_err: logging.warning(f"[Worker p.{page_idx:04d}] Error closing image: {close_err}")

    return result_info


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Gemini PDF->PNG->JSON->CSV Pipeline (Original API Style)")
    parser.add_argument("--pdfs", required=True, nargs="+", help="PDF filenames in data/pdfs/patent_pdfs/")
    parser.add_argument("--temperature", type=float, default=0.5, help="LLM temperature (default=0.5)")
    # Added max_workers argument for consistency, defaulting to original script's implicit behavior (high concurrency)
    parser.add_argument("--max_workers", type=int, default=500, help="Max concurrent workers for page processing (default=500, matching original effective concurrency)")

    args = parser.parse_args()

    # Logging setup (original settings)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logging.info("") # Separator line
    logging.info(f"Using Model: {MODEL_NAME}")

    # --- Load the hardcoded prompt (original logic) ---
    logging.info(f"Loading hardcoded prompt from: {PROMPT_FILE_PATH.relative_to(PROJECT_ROOT)}")
    try:
        task_prompt = PROMPT_FILE_PATH.read_text(encoding="utf-8").strip()
        if not task_prompt:
            logging.critical(f"FATAL: Prompt file is empty: {PROMPT_FILE_PATH}")
            sys.exit(1)
        logging.info("Prompt loaded successfully.")
    except FileNotFoundError:
        logging.critical(f"FATAL: Prompt file not found: {PROMPT_FILE_PATH}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"FATAL: Failed to read prompt file {PROMPT_FILE_PATH}: {e}", exc_info=True)
        sys.exit(1)
    # --- End prompt loading ---

    logging.info(f"Processing {len(args.pdfs)} PDF(s) | Temperature={args.temperature}")
    logging.info("") # Separator line

    global_tokens = defaultdict(int)
    script_start_time = time.time()

    for i, pdf_name in enumerate(args.pdfs, 1):
        pdf_path = PDF_SRC_DIR / pdf_name
        if not pdf_path.is_file():
            logging.error(f"PDF not found: {pdf_path}. Skipping.")
            continue

        pdf_stem = pdf_path.stem
        pdf_start = time.time()
        logging.info(f"Starting PDF {i}/{len(args.pdfs)}: {pdf_name}")
        logging.info("") # Separator line

        pdf_base_out_dir = CSVS_DIR / pdf_stem
        page_by_page_dir = pdf_base_out_dir / "page_by_page"
        png_dir = page_by_page_dir / "PNG"
        json_dir = page_by_page_dir / "JSON" # Define JSON dir path
        png_files = []

        # PDF to PNG conversion (original logic)
        try:
            existing = sorted([p for p in png_dir.glob("page_*.png") if p.is_file()])
            if not existing:
                logging.info(f"Converting PDF->PNG for {pdf_name}...")
                png_dir.mkdir(parents=True, exist_ok=True)
                try:
                    pages = convert_from_path(str(pdf_path)) # Original simple call
                except Exception as convert_e:
                    logging.error(f"PDF conversion failed for {pdf_name}: {convert_e}. Skipping PDF.", exc_info=True)
                    continue # Skip this PDF if conversion fails
                for page_num, img in enumerate(pages, 1):
                    out_png = png_dir / f"page_{page_num:04d}.png"
                    img.save(out_png, "PNG")
                    png_files.append(out_png)
                logging.info(f"Successfully created {len(pages)} PNGs in {png_dir.relative_to(PROJECT_ROOT)}")
            else:
                logging.info(f"Found {len(existing)} existing PNGs in {png_dir.relative_to(PROJECT_ROOT)}")
                png_files = existing
            if not png_files:
                logging.error(f"No PNG files available for {pdf_name} after check/conversion. Skipping PDF.")
                continue
        except Exception as e:
            logging.error(f"Error during PNG preparation for {pdf_name}: {e}. Skipping PDF.", exc_info=True)
            if png_dir.exists(): shutil.rmtree(png_dir, ignore_errors=True) # Clean up partial conversion
            continue

        # --- Process pages for the single, loaded prompt (original logic) ---
        prompt_start_time = time.time() # Track time for processing this PDF with the prompt
        failed_pages_for_pdf, errors_for_pdf = [], {}
        page_results_list = []

        logging.info(f"Launching {len(png_files)} page tasks (max_workers={args.max_workers})...")
        logging.info("") # Separator line

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Ensure only valid PNG files are submitted
            valid_png_files = [png for png in png_files if png.is_file()]
            if len(valid_png_files) != len(png_files):
                logging.warning(f"{len(png_files) - len(valid_png_files)} PNG file(s) were missing or invalid.")

            # Call process_page with the loaded task_prompt and json_dir
            futures = {executor.submit(process_page, k, png, task_prompt, args.temperature, json_dir): (k, png.name)
                       for k, png in enumerate(valid_png_files, 1)}

            processed_count = 0
            total_tasks = len(futures)
            for fut in as_completed(futures):
                page_idx, png_name = futures[fut]
                processed_count += 1
                progress = f"({processed_count}/{total_tasks})"
                try:
                    res = fut.result()
                    page_results_list.append(res)
                    # Accumulate tokens for this PDF (original accumulation logic)
                    pdf_tokens = defaultdict(int) # Recalculate PDF tokens based on results list below
                    for r in page_results_list:
                        pdf_tokens['prompt'] += r.get("prompt_tokens", 0)
                        pdf_tokens['candidate'] += r.get("candidate_tokens", 0)
                        pdf_tokens['total'] += r.get("total_tokens", 0)

                    if not res["success"]:
                        failed_pages_for_pdf.append(page_idx)
                        error_type = res.get('error_type', '?')
                        error_msg = res.get('error_msg', 'N/A')
                        errors_for_pdf[page_idx] = f"Type={error_type}, Msg={error_msg}"
                        logging.warning(f"Page {page_idx:04d} {progress} FAILED ({png_name}): {errors_for_pdf[page_idx]}")
                        # Original error aggregation logic was slightly different, simplified here based on result_info
                    else:
                        # Use DEBUG for successful pages to reduce noise (as in previous version)
                        logging.debug(f"Page {page_idx:04d} {progress} OK ({png_name}).")
                except Exception as e:
                    logging.error(f"[FATAL] Error retrieving result for page {page_idx:04d} ({png_name}): {e}", exc_info=True)
                    failed_pages_for_pdf.append(page_idx)
                    errors_for_pdf[page_idx] = f"Future error: {e}"
                    page_results_list.append({"page_idx": page_idx, "success": False, "error_type": "future", "error_msg": str(e)})

        logging.info("") # Separator line
        logging.info(f"Finished processing {total_tasks} pages for PDF {pdf_name}.")

        # Update global tokens after processing all pages for the PDF
        # Note: pdf_tokens was recalculated above based on the full results list
        global_tokens['prompt'] += pdf_tokens['prompt']
        global_tokens['candidate'] += pdf_tokens['candidate']
        global_tokens['total'] += pdf_tokens['total']

        # Aggregate final failure counts for reporting (matching original reporting style)
        total_pdf_api_failures = sum(r.get('api_failures', 0) for r in page_results_list if not r['success'] and r.get('error_type') == 'api')
        total_pdf_parse_failures = sum(r.get('parse_failures', 0) for r in page_results_list if not r['success'] and r.get('error_type') == 'parse')
        total_pdf_rate_limit_hits = sum(r.get('rate_limit_failures', 0) for r in page_results_list) # Sum hits regardless of success/fail

        if failed_pages_for_pdf:
             # Original error file writing logic
             err_file = pdf_base_out_dir / f"errors_{pdf_stem}.txt"
             logging.warning(f"{len(failed_pages_for_pdf)} page(s) failed. Writing details to: {err_file.relative_to(PROJECT_ROOT)}")
             pdf_base_out_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists
             with err_file.open("w", encoding="utf-8") as ef:
                 ef.write(f"Errors for PDF: {pdf_name}\n")
                 ef.write(f"Prompt Used: {PROMPT_FILE_PATH.name}\n") # Assuming single prompt use
                 ef.write(f"Total Failed Pages: {len(failed_pages_for_pdf)}\n\n")
                 # Use aggregated counts calculated above
                 api_fails_count = sum(1 for r in page_results_list if not r['success'] and r.get('error_type') == 'api')
                 parse_fails_count = sum(1 for r in page_results_list if not r['success'] and r.get('error_type') == 'parse')
                 other_fails = sum(1 for r in page_results_list if not r['success'] and r.get('error_type') not in ['api', 'parse'])
                 ef.write(f"API Call Final Failures (pages): {api_fails_count}\n") # Count pages where API was final error
                 ef.write(f"JSON Parse Final Failures (pages): {parse_fails_count}\n") # Count pages where Parse was final error
                 ef.write(f"Other/Future Failures: {other_fails}\n")
                 ef.write(f"Rate Limit Hits (occurrences): {total_pdf_rate_limit_hits}\n\nDetails:\n")
                 for p_idx in sorted(failed_pages_for_pdf):
                     ef.write(f"Page {p_idx:04d}: {errors_for_pdf.get(p_idx)}\n")

        # <<< START: MODIFIED CSV CONSOLIDATION LOGIC WITH FFILL, FILTER, ID, RENAME, REORDER >>>
        logging.info(f"Starting consolidation for {pdf_stem} (ffill, filter, id, rename, reorder)")
        all_json_files = sorted(json_dir.glob("page_*.json"))
        consolidated_data_with_page = []
        read_json_count = 0
        read_errors = 0
        total_items_read = 0

        if not all_json_files:
            try:
                log_dir_path = json_dir.relative_to(PROJECT_ROOT)
            except (ValueError, NameError):
                log_dir_path = json_dir
            logging.warning(f"No JSON files found in {log_dir_path} to consolidate for {pdf_stem}")
        else:
            logging.info(f"Reading content from {len(all_json_files)} JSON files for {pdf_stem}...")
            for fpath in all_json_files:
                page_num = -1
                try:
                    match = re.search(r'page_(\d+)\.json$', fpath.name)
                    if match:
                        page_num = int(match.group(1))
                    else:
                        logging.warning(f"Could not extract page number from filename: {fpath.name}. Skipping file.")
                        continue

                    if page_num in failed_pages_for_pdf:
                        logging.warning(f"Skipping reading failed page's JSON: {fpath.name} (Page {page_num})")
                        continue

                    with fpath.open("r", encoding="utf-8") as jf:
                        content = json.load(jf)
                    read_json_count += 1

                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                consolidated_data_with_page.append({'data': item, 'page': page_num})
                                total_items_read += 1
                            else:
                                logging.debug(f"Skipping non-dict item within list in {fpath.name}: {type(item)}") # Debug level
                    elif isinstance(content, dict):
                        consolidated_data_with_page.append({'data': content, 'page': page_num})
                        total_items_read += 1
                    else:
                        logging.warning(f"Unexpected or empty content type ({type(content)}) in {fpath.name}. Skipping content.")

                except json.JSONDecodeError as e:
                    try: log_fpath = fpath.relative_to(PROJECT_ROOT)
                    except (ValueError, NameError): log_fpath = fpath
                    logging.error(f"Error decoding JSON {log_fpath}: {e}. Skipping.")
                    read_errors += 1
                except ValueError as e:
                    try: log_fpath = fpath.relative_to(PROJECT_ROOT)
                    except (ValueError, NameError): log_fpath = fpath
                    logging.error(f"Error processing page number for {log_fpath}: {e}", exc_info=True)
                    read_errors += 1
                except Exception as e:
                    try: log_fpath = fpath.relative_to(PROJECT_ROOT)
                    except (ValueError, NameError): log_fpath = fpath
                    logging.error(f"Error reading JSON {log_fpath}: {e}", exc_info=True)
                    read_errors += 1

            logging.info(f"Read {read_json_count} JSON files, found {total_items_read} processable items.")
            if read_errors > 0:
                logging.warning(f"Encountered {read_errors} errors during JSON reading/processing.")

        # --- Process the consolidated data ---
        initial_csv_data = []
        logging.info(f"Generating initial data rows from {len(consolidated_data_with_page)} items...")
        for record in consolidated_data_with_page:
            item = record.get('data', {})
            page_num = record.get('page', None)
            if isinstance(item, dict):
                entry_value = item.get("entry", None)
                category_value = item.get("category", None)
                initial_csv_data.append({
                    "entry": entry_value,
                    "category": category_value,
                    "page_number": page_num # Keep original name for now
                })

        logging.info(f"Generated {len(initial_csv_data)} initial rows.")

        # --- Manipulate data using Pandas ---
        if initial_csv_data:
            final_csv_path = pdf_base_out_dir / f"{pdf_stem}.csv"
            try:
                logging.info("Creating DataFrame...")
                df = pd.DataFrame(initial_csv_data)

                # 1. Forward fill 'category'
                logging.info("Performing forward fill on 'category' column...")
                # Replace potential empty strings with None before ffill if necessary
                # df['category'] = df['category'].replace('', None) # Optional: uncomment if JSON might contain "" for category
                df['category'] = df['category'].fillna(method='ffill')
                logging.info("Forward fill complete.")

                # 2. Filter out rows where 'entry' is null/empty
                logging.info("Filtering rows with empty 'entry'...")
                original_row_count = len(df)
                # df = df[df['entry'].notna()] # Filter only for None/NaN
                # More robust filter: handles None, NaN, and empty strings ''
                df = df[df['entry'].notna() & (df['entry'] != '')]
                rows_removed = original_row_count - len(df)
                logging.info(f"Filtering complete. Removed {rows_removed} rows.")

                if not df.empty:
                    # 3. Rename 'page_number' to 'page'
                    logging.info("Renaming 'page_number' column to 'page'...")
                    df.rename(columns={'page_number': 'page'}, inplace=True)

                    # 4. Add sequential 'id' column (starting from 1)
                    logging.info("Adding sequential 'id' column...")
                    df['id'] = range(1, len(df) + 1)

                    # 5. Reorder columns
                    logging.info("Reordering columns to 'id', 'page', 'entry', 'category'...")
                    df = df[['id', 'page', 'entry', 'category']]

                    # --- Save the final DataFrame ---
                    logging.info(f"Saving final DataFrame with {len(df)} rows to CSV...")
                    final_csv_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(final_csv_path, index=False, encoding='utf-8', quoting=1) # quoting=1 for quoted fields

                    try:
                        log_path = final_csv_path.relative_to(PROJECT_ROOT)
                    except (ValueError, NameError):
                        log_path = final_csv_path
                    logging.info(f"Final CSV saved: {log_path} ({len(df)} rows)")
                else:
                    logging.warning(f"DataFrame became empty after filtering for PDF '{pdf_stem}'. Final CSV file will not be created.")

            except Exception as e_pandas:
                logging.error(f"Error during Pandas manipulation or CSV writing for {final_csv_path}: {e_pandas}", exc_info=True)
        else:
            # Log if no initial data was generated
            if total_items_read > 0:
                logging.warning(f"No dictionary items found or processed from JSON files for PDF '{pdf_stem}' after initial processing step. Final CSV file will not be created.")
            elif read_json_count > 0:
                logging.warning(f"JSON files were read for PDF '{pdf_stem}', but contained no processable dictionary items. Final CSV file will not be created.")
            else:
                logging.warning(f"No JSON files were successfully read or no data found for PDF '{pdf_stem}'. Final CSV file will not be created.")

        # <<< END: MODIFIED CSV CONSOLIDATION LOGIC >>>

        pdf_duration = time.time() - prompt_start_time # Use the PDF processing start time
        logging.info(f"Tokens (PDF: {pdf_name}): Prompt Tokens={pdf_tokens['prompt']}, Candidate Tokens={pdf_tokens['candidate']}, Total Tokens={pdf_tokens['total']}")
        logging.info("") # Separator line
        logging.info(f"PDF {pdf_name} Processing Finished ({format_duration(pdf_duration)})")
        logging.info(f"Global Tokens Running Total: Prompt Tokens={global_tokens['prompt']}, Candidate Tokens={global_tokens['candidate']}, Total Tokens={global_tokens['total']}")
        logging.info("-" * 80)

        # --- PDF Cleanup (original logic) ---
        if png_dir.exists():
            logging.info(f"Cleaning up PNG directory: {png_dir.relative_to(PROJECT_ROOT)}")
            shutil.rmtree(png_dir, ignore_errors=True)
        else:
            # Original script didn't log this case, adding for consistency
            logging.info(f"PNG directory not found for cleanup or already removed: {png_dir.relative_to(PROJECT_ROOT)}")

        pdf_total_duration = time.time() - pdf_start
        logging.info(f"Finished PDF {i}/{len(args.pdfs)}: {pdf_name} (Total time: {format_duration(pdf_total_duration)})")
        # Use failure counts calculated earlier for summary
        if total_pdf_api_failures > 0 or total_pdf_parse_failures > 0:
             api_fails_count = sum(1 for r in page_results_list if not r['success'] and r.get('error_type') == 'api')
             parse_fails_count = sum(1 for r in page_results_list if not r['success'] and r.get('error_type') == 'parse')
             # Original summary format
             logging.warning(f"PDF Summary: API Failures (pages)={api_fails_count}, Parse Failures (pages)={parse_fails_count}, Rate Limit Hits={total_pdf_rate_limit_hits}")

    # --- Final Summary (original format) ---
    logging.info("") # Separator before summary
    logging.info(" SCRIPT COMPLETE ".center(80, "="))
    logging.info(" Global Usage Summary ".center(80, "="))
    logging.info(f"  Prompt Tokens:     {global_tokens['prompt']:,}")
    logging.info(f"  Candidate Tokens:  {global_tokens['candidate']:,}")
    logging.info(f"  Total Tokens:      {global_tokens['total']:,}")
    script_duration = time.time() - script_start_time
    logging.info(f"  Total Script Time: {format_duration(script_duration)}")
    logging.info("=" * 80)
    logging.info("Exiting.")

if __name__ == "__main__":
    main()