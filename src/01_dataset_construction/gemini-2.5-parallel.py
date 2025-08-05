#!/usr/bin/env python3

# Import modules
import os
import sys
import re
import json
import time
import argparse
import logging
import tempfile
import resource
import threading
import pandas as pd
import google.genai as genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum, auto
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CSVS_DIR = DATA_DIR / "01_dataset_construction" / "csvs"
PDF_SRC_DIR = DATA_DIR / "pdfs" / "patent_pdfs"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

# Environment setup
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")

# Model configuration
MODEL_NAME = "gemini-2.5-pro" #flash
MAX_OUTPUT_TOKENS = 65536 # max output window is 65536
DEFAULT_THINKING_BUDGET = 32768 #24576 #-1  # Default thinking budget, -1 for dynamic
MAX_THINKING_BUDGET = 32768 #24576 # Maximum thinking budget
MAX_FILE_DESCRIPTORS = 10000

# Error type enumeration
class ErrorType(Enum):
    API = auto()           # API call failures
    PARSE = auto()         # JSON parsing failures
    FILE_LIMIT = auto()    # Too many open files
    RATE_LIMIT = auto()    # Rate limit hits
    FILE_NOT_FOUND = auto()  # File not found errors
    SAVE = auto()          # JSON save failures
    FUTURE = auto()        # Future execution errors
    OTHER = auto()         # Other unexpected errors

# Error tracking implementation
class ErrorTracker:
    """Thread-safe error tracking system for parallel processing."""
    def __init__(self):
        self.errors = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.rate_limit_hits = 0
        self._lock = threading.Lock()
    def add_error(self, page_idx: int, error_type: ErrorType, error_msg: str, is_rate_limit: bool = False):
        with self._lock:
            self.errors[page_idx].append((error_type, error_msg))
            self.error_counts[error_type] += 1
            if is_rate_limit:
                self.rate_limit_hits += 1
    def get_error_summary(self) -> Dict[str, int]:
        with self._lock:
            summary = {error_type.name: count for error_type, count in self.error_counts.items()}
            summary["RATE_LIMIT_HITS"] = self.rate_limit_hits
            return summary
    def get_failed_pages(self) -> Set[int]:
        with self._lock:
            return set(self.errors.keys())
    def get_page_errors(self, page_idx: int) -> List[Tuple[ErrorType, str]]:
        with self._lock:
            return self.errors.get(page_idx, [])

# Utility functions
def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS format."""
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    return f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"

def parse_json_str(response_text: str) -> Any:
    """Parse JSON from response text with robust error handling."""
    # Extract JSON from code blocks if present
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
        if candidate.startswith('\ufeff'): candidate = candidate[1:]
    else:
        # Try the entire response if no code blocks
        candidate = response_text.strip().strip("`")
        if candidate.startswith('\ufeff'): candidate = candidate[1:]
    
    # Parse the candidate as JSON
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        # Provide detailed error context
        error_context = f"Failed to parse JSON content: {e}\n"
        error_context += f"Content snippet: {candidate[:200]}...\n"
        error_context += f"Content length: {len(candidate)} characters\n"
        
        # Identify common JSON formatting issues
        if "Expecting property name" in str(e):
            error_context += "Possible issue: Missing quotes around property names or trailing commas\n"
        elif "Expecting ',' delimiter" in str(e):
            error_context += "Possible issue: Missing commas between array/object elements\n"
        elif "Expecting value" in str(e):
            error_context += "Possible issue: Empty values or missing values\n"
        
        raise ValueError(error_context) from e

def increase_file_descriptor_limit():
    """Increase the file descriptor limit to prevent 'Too many open files' errors."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(MAX_FILE_DESCRIPTORS, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        
        # Verify the change was successful
        current_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if current_soft < new_soft:
            logging.warning(f"Failed to increase file descriptor limit to {new_soft}. Current limit: {current_soft}")
            return False
            
        logging.info(f"Increased file descriptor limit from {soft} to {new_soft}")
        return True
    except Exception as e:
        logging.warning(f"Failed to increase file descriptor limit: {e}")
        return False

def get_relative_path(path: Path) -> Path:
    """Get a path relative to the project root, or return the original path if not possible."""
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        # Only catch ValueError, which occurs when paths are not relative
        return path

# API call
def gemini_api_call(prompt: str, pil_image: Image.Image, temperature: float, thinking_budget: int) -> Tuple[Optional[dict], str, bool, dict]:
    """Execute a Gemini API call with no retry logic and direct error handling. Returns extra_info dict for error context."""
    try:
        client = genai.Client(api_key=API_KEY)
    except AttributeError:
         logging.critical("FATAL: 'genai.Client' not found. Library version/installation issue?", exc_info=True)
         return (None, "FATAL: genai.Client not found in library.", False, {})
    except Exception as client_e:
         logging.critical(f"FATAL: Failed to initialize Gemini Client: {client_e}", exc_info=True)
         return (None, f"FATAL: Client init failed: {client_e}", False, {})

    error_msg = ""
    is_rate_limit = False
    tmp_file_path = None
    file_upload = None
    extra_info = {}

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_file_path_str = tmp.name
            tmp_file_path = Path(tmp_file_path_str)
            pil_image.save(tmp_file_path_str, "PNG")

        try:
            file_upload = client.files.upload(file=tmp_file_path_str)
            logging.debug(f"File uploaded via client.files.upload: {file_upload.uri}")
        except Exception as upload_err:
            error_msg = f"File upload failed: {upload_err}"
            logging.warning(error_msg)
            if tmp_file_path and tmp_file_path.exists():
                try: 
                    tmp_file_path.unlink()
                    logging.debug(f"Deleted temp file after upload failure: {tmp_file_path}")
                except OSError as e: 
                    logging.warning(f"Failed to delete temp file {tmp_file_path}: {e}")
            tmp_file_path = None
            return (None, error_msg, False, {})

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_uri(
                    file_uri=file_upload.uri,
                    mime_type=file_upload.mime_type,
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                    include_thoughts=True
                )
            )
        )

        # --- Enhanced error info for empty response ---
        if not response or not response.text:
            error_msg = "API returned empty response or no text"
            block_reason = None
            prompt_feedback = None
            # Try to extract token usage if available
            usage = getattr(response, 'usage_metadata', None)
            ptk = getattr(usage, 'prompt_token_count', None) if usage else None
            ttk = getattr(usage, 'thoughts_token_count', None) if usage else None
            ctk = getattr(usage, 'candidates_token_count', None) if usage else None
            totk = getattr(usage, 'total_token_count', None) if usage else None
            extra_info['prompt_tokens'] = ptk if ptk is not None else 'N/A'
            extra_info['thoughts_tokens'] = ttk if ttk is not None else 'N/A'
            extra_info['candidate_tokens'] = ctk if ctk is not None else 'N/A'
            extra_info['total_tokens'] = totk if totk is not None else 'N/A'
            if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                prompt_feedback = response.prompt_feedback
                block_reason = getattr(prompt_feedback, 'block_reason', None)
                extra_info['prompt_feedback'] = str(prompt_feedback)
                if block_reason:
                    error_msg += f" (Block Reason: {block_reason})"
                    logging.warning(f"Block Reason: {block_reason}")
                else:
                    logging.warning(f"Prompt feedback: {prompt_feedback}")
            else:
                logging.warning("No prompt_feedback available in response.")
            extra_info['block_reason'] = block_reason
            # Log token usage in terminal
            logging.warning(f"Token usage for failed page: prompt={extra_info['prompt_tokens']}, candidate={extra_info['candidate_tokens']}, thoughts={extra_info['thoughts_tokens']}, total={extra_info['total_tokens']}")
            return (None, error_msg, False, extra_info)

        usage = response.usage_metadata
        # Extract all token counts, including thoughts
        ptk = getattr(usage, 'prompt_token_count', 0) or 0
        ttk = getattr(usage, 'thoughts_token_count', 0) or 0
        ctk = getattr(usage, 'candidates_token_count', 0) or 0
        totk = getattr(usage, 'total_token_count', 0) or 0

        return ({"text": response.text, "usage": usage, "prompt_tokens": ptk, "thoughts_tokens": ttk, "candidate_tokens": ctk, "total_tokens": totk}, "", False, {})

    except google_exceptions.ResourceExhausted as e:
        is_rate_limit = True
        error_msg = f"API Error (Rate Limit): {e}"
        logging.warning("Rate limit error! Sleeping 30s...")
        return (None, error_msg, is_rate_limit, {})

    except Exception as e:
        msg = str(e)
        error_msg = f"API Error (General): {type(e).__name__} - {msg}"
        if "429" in msg or "rate limit" in msg.lower() or "resource has been exhausted" in msg.lower():
            is_rate_limit = True
            logging.warning("Rate limit detected! Sleeping 30s...")
        else:
            logging.warning(error_msg, exc_info=True)
        return (None, error_msg, is_rate_limit, {})

    finally:
        # Clean up temporary file
        if tmp_file_path and tmp_file_path.exists():
            try:
                tmp_file_path.unlink()
                logging.debug(f"Deleted temp file: {tmp_file_path}")
            except OSError as del_err:
                logging.warning(f"Failed to delete temp file {tmp_file_path}: {del_err}")
        tmp_file_path = None
        # Clean up file upload
        if file_upload:
            client.files.delete(name=file_upload.name)

# Page processing
def process_page(page_idx: int,
                 png_path: Path,
                 prompt_text: str,
                 temperature: float,
                 thinking_budget: int,
                 json_dir: Path,
                 error_tracker: ErrorTracker) -> dict:
    """Process a single page image through the Gemini API and save the results."""
    result_info = {
        "page_idx": page_idx, "prompt_tokens": 0, "candidate_tokens": 0, "thoughts_tokens": 0, "total_tokens": 0,
        "success": False, "error_msg": "", "error_type": None, "api_failures": 0,
        "rate_limit_failures": 0, "parse_failures": 0, "last_parse_error": "",
        "extra_info": {}  # <-- new field for extra error info
    }
    logging.info(f"[Worker p.{page_idx:04d}] Processing {png_path.name}...")
    pil_image = None

    try:
        pil_image = Image.open(png_path)
        w, h = pil_image.size
        #logging.info(f"[Worker p.{page_idx:04d}] PNG info: {w}x{h}")

        initial_result, error, is_rate_limit, extra_info = gemini_api_call(prompt_text, pil_image, temperature, thinking_budget)

        if not initial_result:
            result_info["error_msg"] = error
            result_info["error_type"] = "api"
            result_info["api_failures"] = 1
            result_info["extra_info"] = extra_info or {}
            # Enhanced terminal logging for extra_info
            if extra_info:
                for k, v in extra_info.items():
                    logging.warning(f"[Worker p.{page_idx:04d}] Extra error info: {k}: {v}")
                # Log token usage summary for failed page
                logging.warning(f"[Worker p.{page_idx:04d}] Token usage (failed): prompt={extra_info.get('prompt_tokens','N/A')}, candidate={extra_info.get('candidate_tokens','N/A')}, thoughts={extra_info.get('thoughts_tokens','N/A')}, total={extra_info.get('total_tokens','N/A')}")
                # Extract token usage from extra_info and add to result_info for global tracking
                result_info["prompt_tokens"] = extra_info.get('prompt_tokens', 0) or 0
                result_info["candidate_tokens"] = extra_info.get('candidate_tokens', 0) or 0
                result_info["thoughts_tokens"] = extra_info.get('thoughts_tokens', 0) or 0
                result_info["total_tokens"] = extra_info.get('total_tokens', 0) or 0
            if is_rate_limit: 
                result_info["rate_limit_failures"] = 1
                error_tracker.add_error(page_idx, ErrorType.RATE_LIMIT, error, True)
            else:
                error_tracker.add_error(page_idx, ErrorType.API, error)
            return result_info

        usage = initial_result["usage"]
        ptk = getattr(usage, 'prompt_token_count', 0) or 0
        ctk = getattr(usage, 'candidates_token_count', 0) or 0
        thtk = getattr(usage, 'thoughts_token_count', 0) or 0
        ttk = getattr(usage, 'total_token_count', 0) or 0
        result_info.update({"prompt_tokens": ptk, "candidate_tokens": ctk, "thoughts_tokens": thtk, "total_tokens": ttk})
        resp_text = initial_result["text"]
        logging.info(f"[Worker p.{page_idx:04d}] Initial API usage -> prompt={ptk}, candidate={ctk}, thoughts={thtk}, total={ttk}")

        # Only one parse attempt
        try:
            parsed = parse_json_str(resp_text)
            logging.info(f"[Worker p.{page_idx:04d}] Parsed JSON successfully.")
        except ValueError as ve:
            result_info["parse_failures"] = 1
            parse_error_msg = str(ve)
            result_info["last_parse_error"] = parse_error_msg
            logging.error(f"[Worker p.{page_idx:04d}] JSON parse error: {parse_error_msg}")
            logging.error(f"[Worker p.{page_idx:04d}] RAW RESPONSE:\n{'-'*60}\n{resp_text}\n{'-'*60}")
            result_info["error_msg"] = f"JSON parse failed: {parse_error_msg}"
            result_info["error_type"] = "parse"
            error_tracker.add_error(page_idx, ErrorType.PARSE, parse_error_msg)
            if pil_image:
                try: pil_image.close()
                except Exception: pass
            return result_info

        json_dir.mkdir(parents=True, exist_ok=True)
        json_out = json_dir / f"{png_path.stem}.json"
        try:
            with json_out.open("w", encoding="utf-8") as jf:
                json.dump(parsed, jf, indent=2, ensure_ascii=False)
            result_info["success"] = True
            logging.debug(f"[Worker p.{page_idx:04d}] Successfully saved JSON to {get_relative_path(json_out)}")
        except Exception as e_save:
            logging.error(f"[Worker p.{page_idx:04d}] Failed to save JSON {get_relative_path(json_out)}: {e_save}")
            result_info.update({"error_msg": f"Failed to save JSON: {e_save}", "error_type": "save", "success": False})
            error_tracker.add_error(page_idx, ErrorType.SAVE, str(e_save))

    except FileNotFoundError:
        logging.error(f"[Worker p.{page_idx:04d}] File not found: {get_relative_path(png_path)}")
        result_info.update({"error_msg": f"File not found: {png_path.name}", "error_type": "file", "success": False})
        error_tracker.add_error(page_idx, ErrorType.FILE_NOT_FOUND, f"File not found: {png_path.name}")
    except OSError as e_os:
        if "Too many open files" in str(e_os):
            logging.error(f"[Worker p.{page_idx:04d}] Too many open files: {e_os}")
            result_info.update({"error_msg": f"Too many open files: {e_os}", "error_type": "file_limit", "success": False})
            error_tracker.add_error(page_idx, ErrorType.FILE_LIMIT, str(e_os))
        else:
            logging.exception(f"[Worker p.{page_idx:04d}] OS error: {e_os}")
            result_info.update({"error_msg": f"OS error: {e_os}", "error_type": "other", "success": False})
            error_tracker.add_error(page_idx, ErrorType.OTHER, str(e_os))
    except Exception as e_outer:
        logging.exception(f"[Worker p.{page_idx:04d}] Unexpected error: {e_outer}")
        result_info.update({"error_msg": f"Unexpected error: {e_outer}", "error_type": "other", "success": False})
        error_tracker.add_error(page_idx, ErrorType.OTHER, str(e_outer))
    finally:
        if pil_image:
            try: pil_image.close()
            except Exception as close_err: logging.warning(f"[Worker p.{page_idx:04d}] Error closing image: {close_err}")

    return result_info

# Page processing
def process_specific_pages(pages: List[int], png_dir: Path, json_dir: Path, task_prompt: str, 
                         temperature: float, thinking_budget: int, error_tracker: ErrorTracker, max_workers: int = 5) -> Tuple[List[dict], Set[int]]:
    """Process a specific list of pages in parallel. Returns (results, successfully_processed_pages)."""
    results = []
    successful_pages = set()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {}
        for page_num in pages:
            if not validate_page_exists(png_dir, page_num):
                continue
            if check_json_exists(json_dir, page_num):
                logging.info(f"Page {page_num:04d} skipped: JSON already exists.")
                successful_pages.add(page_num)
                # Add a dummy result for skipped pages to maintain consistency
                results.append({
                    "page_idx": page_num,
                    "prompt_tokens": 0,
                    "candidate_tokens": 0,
                    "thoughts_tokens": 0,
                    "total_tokens": 0,
                    "success": True,
                    "error_msg": "",
                    "error_type": None,
                    "api_failures": 0,
                    "rate_limit_failures": 0,
                    "parse_failures": 0,
                    "last_parse_error": "",
                    "extra_info": {}
                })
                continue
            png_file = png_dir / f"page_{page_num:04d}.png"
            future = executor.submit(process_page, page_num, png_file, task_prompt, temperature, thinking_budget, json_dir, error_tracker)
            future_to_page[future] = page_num
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                result = future.result()
                results.append(result)
                if result["success"]:
                    successful_pages.add(page_num)
            except Exception as e:
                logging.error(f"Error processing page {page_num}: {e}")
    return results, successful_pages

# After the ErrorTracker class, before main()

def check_json_exists(json_dir: Path, page_num: int) -> bool:
    """Check if JSON file exists for a specific page."""
    json_file = json_dir / f"page_{page_num:04d}.json"
    exists = json_file.exists()
    if exists:
        logging.info(f"JSON file already exists for page {page_num:04d}: {get_relative_path(json_file)}")
    else:
        logging.info(f"No existing JSON file found for page {page_num:04d}")
    return exists

def validate_page_exists(png_dir: Path, page_num: int) -> bool:
    """Validate that PNG file exists for a specific page."""
    png_file = png_dir / f"page_{page_num:04d}.png"
    if not png_file.exists():
        logging.error(f"PNG file not found for page {page_num:04d}: {get_relative_path(png_file)}")
        return False
    return True

def create_consolidated_csv(json_dir: Path, pdf_base_out_dir: Path, pdf_stem: str) -> Optional[Path]:
    """Create consolidated CSV from all available JSON files."""
    all_json_files = sorted(json_dir.glob("page_*.json"))
    if not all_json_files:
        logging.warning(f"No JSON files found in {get_relative_path(json_dir)} to consolidate")
        return None

    consolidated_data_with_page = []
    read_json_count = 0
    read_errors = 0
    total_items_read = 0

    logging.info(f"Reading content from {len(all_json_files)} JSON files...")
    for fpath in all_json_files:
        page_num = -1
        try:
            match = re.search(r'page_(\d+)\.json$', fpath.name)
            if not match:
                logging.warning(f"Could not extract page number from filename: {fpath.name}. Skipping file.")
                continue
            page_num = int(match.group(1))

            with fpath.open("r", encoding="utf-8") as jf:
                content = json.load(jf)
            read_json_count += 1

            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        consolidated_data_with_page.append({'data': item, 'page': page_num})
                        total_items_read += 1
            elif isinstance(content, dict):
                consolidated_data_with_page.append({'data': content, 'page': page_num})
                total_items_read += 1

        except Exception as e:
            logging.error(f"Error processing {get_relative_path(fpath)}: {e}")
            read_errors += 1

    if not consolidated_data_with_page:
        logging.warning("No valid data found in JSON files")
        return None

    # Generate CSV data
    initial_csv_data = []
    for record in consolidated_data_with_page:
        item = record.get('data', {})
        page_num = record.get('page', None)
        if isinstance(item, dict):
            entry_value = item.get("entry", None)
            category_value = item.get("category", None)
            initial_csv_data.append({
                "entry": entry_value,
                "category": category_value,
                "page_number": page_num
            })

    if not initial_csv_data:
        logging.warning("No valid rows generated for CSV")
        return None

    # Create DataFrame and process
    df = pd.DataFrame(initial_csv_data)
    df['category'] = df['category'].ffill()
    df = df[df['entry'].notna() & (df['entry'] != '')]
    
    if df.empty:
        logging.warning("DataFrame is empty after filtering")
        return None

    # Finalize DataFrame
    df.rename(columns={'page_number': 'page'}, inplace=True)
    df['id'] = range(1, len(df) + 1)
    df = df[['id', 'page', 'entry', 'category']]

    # Save CSV
    final_csv_path = pdf_base_out_dir / f"{pdf_stem}.csv"
    pdf_base_out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(final_csv_path, index=False, encoding='utf-8', quoting=1)
    
    logging.info(f"Created consolidated CSV with {len(df)} rows: {get_relative_path(final_csv_path)}")
    return final_csv_path

def overwrite_error_file(pdf_base_out_dir: Path, pdf_stem: str, pdf_name: str, 
                        failed_pages: Set[int], error_tracker: ErrorTracker, prompt_file_path: Path) -> None:
    """Overwrite the error file with current error information (minimal, world-class)."""
    err_file = pdf_base_out_dir / f"errors_{pdf_stem}.txt"
    logging.warning(f"{len(failed_pages)} page(s) failed. Writing details to: {get_relative_path(err_file)}")
    pdf_base_out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with err_file.open("w", encoding="utf-8") as ef:
            ef.write(f"Failed Pages: {sorted(list(failed_pages))}\n\n")
            ef.write(f"Errors for PDF: {pdf_name}\n")
            ef.write(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            ef.write(f"Prompt Used: {prompt_file_path.name}\n")
            ef.write(f"Total Failed Pages: {len(failed_pages)}\n\n")
            ef.write("Detailed Error Information:\n")
            ef.write("=" * 40 + "\n")
            for p_idx in sorted(failed_pages):
                page_errors = error_tracker.get_page_errors(p_idx)
                # Try to get extra_info from the result_info for this page
                # We'll look for the JSON file for this page and see if extra_info is present
                # But since we don't save result_info for failed pages, let's recommend the user to check the terminal logs for full details
                for error_type, error_msg in page_errors:
                    ef.write(f"\nPage {p_idx:04d}:\n")
                    ef.write(f"  Error Type: {error_type.name}\n")
                    ef.write(f"  Error: {error_msg}\n")
                # Write token usage if available from extra_info in result_info (if present in error_msg, print it)
                # For now, recommend user to check terminal logs for full token usage details
                # But let's try to extract token usage from error_msg if present
                # If error_msg contains 'Token usage', print it as a separate line
                # Instead, let's add a placeholder for token usage
                    ef.write(f"  Token usage: See terminal logs for prompt/candidate/thoughts/total tokens.\n")
                    if 'Block Reason:' in error_msg:
                        ef.write(f"  Block Reason: {error_msg.split('Block Reason:')[1].strip()}\n")
    except Exception as e:
        logging.error(f"Failed to write error file {get_relative_path(err_file)}: {e}", exc_info=True)

# After the overwrite_error_file function, before main()

def update_processing_log(pdf_base_out_dir: Path, pdf_stem: str, pdf_name: str, 
                         page_count: int, pdf_tokens: dict, processing_time: float, max_workers: int) -> None:
    """Create or update a JSON log file with minimal processing information."""
    log_file = pdf_base_out_dir / f"{pdf_stem}_log.json"
    
    # Load existing log data if it exists
    existing_data = {}
    if log_file.exists():
        try:
            with log_file.open("r", encoding="utf-8") as f:
                existing_data = json.load(f)
            logging.info(f"Loaded existing log data from: {get_relative_path(log_file)}")
        except Exception as e:
            logging.warning(f"Failed to load existing log data: {e}")
    
    # Accumulate tokens with existing data
    total_input_tokens = existing_data.get('total_input_tokens', 0) + pdf_tokens.get('prompt', 0)
    total_thought_tokens = existing_data.get('total_thought_tokens', 0) + pdf_tokens.get('thoughts', 0)
    total_candidate_tokens = existing_data.get('total_candidate_tokens', 0) + pdf_tokens.get('candidate', 0)
    
    # For page count, use the maximum of existing and current (in case of single page processing)
    total_pages = max(existing_data.get('number_of_pages', 0), page_count)
    
    # For processing time, accumulate (add new time to existing)
    total_processing_time = existing_data.get('processing_time_seconds', 0) + round(processing_time, 2)
    
    # Use the most recent max_workers setting
    current_max_workers = max_workers
    
    log_data = {
        "file_name": pdf_name,
        "model": MODEL_NAME,
        "number_of_pages": total_pages,
        "total_input_tokens": total_input_tokens,
        "total_thought_tokens": total_thought_tokens,
        "total_candidate_tokens": total_candidate_tokens,
        "processing_time_seconds": total_processing_time,
        "max_workers": current_max_workers
    }
    
    try:
        with log_file.open("w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Processing log updated to: {get_relative_path(log_file)}")
        logging.info(f"Accumulated tokens: input={total_input_tokens:,}, thoughts={total_thought_tokens:,}, candidate={total_candidate_tokens:,}")
    except Exception as e:
        logging.error(f"Failed to write processing log {get_relative_path(log_file)}: {e}")

# Main execution
def main():
    parser = argparse.ArgumentParser(description="Gemini PDF->PNG->JSON->CSV Pipeline")
    parser.add_argument("--pdf", required=True, help="PDF filename in data/pdfs/patent_pdfs/")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default=0.0)")
    parser.add_argument("--thinking_budget", type=int, default=DEFAULT_THINKING_BUDGET, help=f"Thinking budget for the model (default={DEFAULT_THINKING_BUDGET}, max={MAX_THINKING_BUDGET})")
    parser.add_argument("--max_workers", type=int, default=20, help="Max concurrent workers for page processing (default=20)")
    parser.add_argument("--page", type=int, help="Process a single specific page number")
    args = parser.parse_args()
    PROMPT_FILE_PATH = Path(__file__).parent / "prompt.txt"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logging.info("")
    logging.info(f"Using Model: {MODEL_NAME}")
    logging.info(f"Thinking Budget: {args.thinking_budget}")
    increase_file_descriptor_limit()
    logging.info(f"Loading prompt from: {get_relative_path(PROMPT_FILE_PATH)}")
    try:
        task_prompt = PROMPT_FILE_PATH.read_text(encoding="utf-8").strip()
        if not task_prompt:
            logging.critical(f"FATAL: Prompt file is empty: {get_relative_path(PROMPT_FILE_PATH)}")
            sys.exit(1)
        logging.info("Prompt loaded successfully.")
    except FileNotFoundError:
        logging.critical(f"FATAL: Prompt file not found: {get_relative_path(PROMPT_FILE_PATH)}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"FATAL: Failed to read prompt file {get_relative_path(PROMPT_FILE_PATH)}: {e}", exc_info=True)
        sys.exit(1)
    logging.info(f"Processing PDF: {args.pdf} | Temperature={args.temperature}")
    logging.info("")
    global_tokens = defaultdict(int)
    script_start_time = time.time()
    pdf_name = args.pdf
    pdf_path = PDF_SRC_DIR / pdf_name
    if not pdf_path.is_file():
        logging.error(f"PDF not found: {get_relative_path(pdf_path)}. Exiting.")
        sys.exit(1)
    pdf_stem = pdf_path.stem
    pdf_start = time.time()
    logging.info(f"Starting PDF: {pdf_name}")
    logging.info("")
    pdf_base_out_dir = CSVS_DIR / pdf_stem
    page_by_page_dir = pdf_base_out_dir / "page_by_page"
    png_dir = page_by_page_dir / "PNG"
    json_dir = page_by_page_dir / "JSON"
    png_files = []
    try:
        existing = sorted([p for p in png_dir.glob("page_*.png") if p.is_file()])
        if not existing:
            logging.info(f"Converting PDF->PNG for {pdf_name}...")
            png_dir.mkdir(parents=True, exist_ok=True)
            try:
                pages = convert_from_path(str(pdf_path))
            except Exception as convert_e:
                logging.error(f"PDF conversion failed for {pdf_name}: {convert_e}. Exiting.", exc_info=True)
                sys.exit(1)
            for page_num, img in enumerate(pages, 1):
                out_png = png_dir / f"page_{page_num:04d}.png"
                img.save(out_png, "PNG")
                png_files.append(out_png)
            logging.info(f"Successfully created {len(pages)} PNGs in {get_relative_path(png_dir)}")
        else:
            logging.info(f"Found {len(existing)} existing PNGs in {get_relative_path(png_dir)}")
            png_files = existing
        if not png_files:
            logging.error(f"No PNG files available for {pdf_name} after check/conversion. Exiting.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Error during PNG preparation for {pdf_name}: {e}. Exiting.", exc_info=True)
        sys.exit(1)
    prompt_start_time = time.time()
    page_results_list = []
    error_tracker = ErrorTracker()
    if args.page:
        logging.info(f"Processing single page mode: page {args.page}")
        pages_to_process = [args.page]
        results, successful_pages = process_specific_pages(pages_to_process, png_dir, json_dir, 
                                                        task_prompt, args.temperature, args.thinking_budget, error_tracker, args.max_workers)
        page_results_list.extend(results)
        if successful_pages:
            logging.info(f"Successfully processed page {args.page}")
            # Always recreate CSV after successful single page processing
            create_consolidated_csv(json_dir, pdf_base_out_dir, pdf_stem)
        else:
            logging.error(f"Failed to process page {args.page}")
            return  # Only return if processing failed
    else:
        # Parallel processing for full PDF
        logging.info(f"Launching {len(png_files)} page tasks (max_workers={args.max_workers})...")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            valid_png_files = [png for png in png_files if png.is_file()]
            if len(valid_png_files) != len(png_files):
                logging.warning(f"{len(png_files) - len(valid_png_files)} PNG file(s) were missing or invalid.")
            futures = {executor.submit(process_page, k, png, task_prompt, args.temperature, args.thinking_budget, json_dir, error_tracker): (k, png.name)
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
                    if not res["success"]:
                        error_type = res.get('error_type', '?')
                        error_msg = res.get('error_msg', 'N/A')
                        logging.warning(f"Page {page_idx:04d} {progress} FAILED ({png_name}): Type={error_type}, Msg={error_msg}")
                    else:
                        logging.debug(f"Page {page_idx:04d} {progress} OK ({png_name}).")
                except Exception as e:
                    logging.error(f"[FATAL] Error retrieving result for page {page_idx:04d} ({png_name}): {e}", exc_info=True)
                    error_tracker.add_error(page_idx, ErrorType.FUTURE, str(e))
                    page_results_list.append({"page_idx": page_idx, "success": False, "error_type": "future", "error_msg": str(e)})
        logging.info("")
        logging.info(f"Finished processing {total_tasks} pages for PDF {pdf_name}.")
        # Write error file if any failures
        failed_pages = error_tracker.get_failed_pages()
        if failed_pages:
            overwrite_error_file(pdf_base_out_dir, pdf_stem, pdf_name, failed_pages, error_tracker, PROMPT_FILE_PATH)
        # Create CSV after all processing
        create_consolidated_csv(json_dir, pdf_base_out_dir, pdf_stem)
    pdf_duration = time.time() - prompt_start_time
    # After all processing (single-page or multi-page), accumulate tokens
    pdf_tokens = defaultdict(int)
    for r in page_results_list:
        pdf_tokens['prompt'] += r.get("prompt_tokens", 0)
        pdf_tokens['candidate'] += r.get("candidate_tokens", 0)
        pdf_tokens['thoughts'] += r.get("thoughts_tokens", 0)
        pdf_tokens['total'] += r.get("total_tokens", 0)

    global_tokens['prompt'] += pdf_tokens['prompt']
    global_tokens['candidate'] += pdf_tokens['candidate']
    global_tokens['thoughts'] += pdf_tokens['thoughts']
    global_tokens['total'] += pdf_tokens['total']

    # Create processing log
    update_processing_log(pdf_base_out_dir, pdf_stem, pdf_name, len(png_files), pdf_tokens, pdf_duration, args.max_workers)

    logging.info(f"Tokens (PDF: {pdf_name}): prompt={pdf_tokens['prompt']:,}, candidate={pdf_tokens['candidate']:,}, thoughts={pdf_tokens['thoughts']:,}, total={pdf_tokens['total']:,}")
    logging.info("")
    logging.info(f"PDF {pdf_name} Processing Finished ({format_duration(pdf_duration)})")
    logging.info(f"Global Tokens Running Total: prompt={global_tokens['prompt']:,}, candidate={global_tokens['candidate']:,}, thoughts={global_tokens['thoughts']:,}, total={global_tokens['total']:,}")
    logging.info("-" * 80)

    # --- Copy CSV to data/01_dataset_construction/complete_csvs if all pages succeeded, not in single-page mode, and file does not exist ---
    if not args.page:
        failed_pages = error_tracker.get_failed_pages()
        if not failed_pages:
            complete_csvs_dir = PROJECT_ROOT / "data" / "01_dataset_construction" / "complete_csvs"
            complete_csvs_dir.mkdir(parents=True, exist_ok=True)
            src_csv = pdf_base_out_dir / f"{pdf_stem}.csv"
            dst_csv = complete_csvs_dir / f"{pdf_stem}.csv"
            if not dst_csv.exists():
                try:
                    import shutil
                    shutil.copy2(src_csv, dst_csv)
                    logging.info(f"Copied CSV to {get_relative_path(dst_csv)} (all pages succeeded, first time)")
                except Exception as e:
                    logging.error(f"Failed to copy CSV to {get_relative_path(dst_csv)}: {e}")
            else:
                logging.info(f"CSV already exists in {get_relative_path(dst_csv)}; not overwriting.")
    pdf_total_duration = time.time() - pdf_start
    logging.info(f"Finished PDF: {pdf_name} (Total time: {format_duration(pdf_total_duration)})")
    logging.info("")
    logging.info(" SCRIPT COMPLETE ".center(80, "="))
    logging.info(" Global Usage Summary ".center(80, "="))
    logging.info(f"  Prompt Tokens:     {global_tokens['prompt']:,}")
    logging.info(f"  Candidate Tokens:  {global_tokens['candidate']:,}")
    logging.info(f"  Thoughts Tokens:   {global_tokens['thoughts']:,}")
    logging.info(f"  Total Tokens:      {global_tokens['total']:,}")
    script_duration = time.time() - script_start_time
    logging.info(f"  Total Script Time: {format_duration(script_duration)}")
    logging.info("=" * 80)
    logging.info("Exiting.")

if __name__ == "__main__":
    main()