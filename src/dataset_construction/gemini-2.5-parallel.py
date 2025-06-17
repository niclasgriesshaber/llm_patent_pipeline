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
CSVS_DIR = DATA_DIR / "csvs"
PROMPT_SRC_DIR = PROJECT_ROOT / "src" / "dataset_construction"
PDF_SRC_DIR = DATA_DIR / "pdfs" / "patent_pdfs"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

# Prompt file path
PROMPT_FILE_PATH = PROMPT_SRC_DIR / "prompt.txt"

# Environment setup
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")

# Model configuration
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
MAX_OUTPUT_TOKENS = 8192 # for gemini-2.0-flash
DEFAULT_THINKING_BUDGET = 10000  # Default thinking budget
MAX_THINKING_BUDGET = 24576  # Maximum thinking budget
MAX_RETRIES = 3
BACKOFF_SLEEP_SECONDS = 30
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
        self.retry_counts = defaultdict(int)
        self.max_retries = MAX_RETRIES
        self._lock = threading.Lock()
    
    def add_error(self, page_idx: int, error_type: ErrorType, error_msg: str, is_rate_limit: bool = False):
        """Record an error with thread-safe access."""
        with self._lock:
            self.errors[page_idx].append((error_type, error_msg))
            self.error_counts[error_type] += 1
            if is_rate_limit:
                self.rate_limit_hits += 1
    
    def add_error_and_get_summary(self, page_idx: int, error_type: ErrorType, error_msg: str, is_rate_limit: bool = False) -> Dict[str, int]:
        """Record an error and return the current error summary atomically."""
        with self._lock:
            self.errors[page_idx].append((error_type, error_msg))
            self.error_counts[error_type] += 1
            if is_rate_limit:
                self.rate_limit_hits += 1
            
            summary = {error_type.name: count for error_type, count in self.error_counts.items()}
            summary["RATE_LIMIT_HITS"] = self.rate_limit_hits
            return summary
    
    def increment_retry(self, page_idx: int) -> bool:
        """Increment retry counter and check if more retries are allowed."""
        with self._lock:
            self.retry_counts[page_idx] += 1
            return self.retry_counts[page_idx] <= self.max_retries
    
    def get_retry_count(self, page_idx: int) -> int:
        """Get the current retry count for a page."""
        with self._lock:
            return self.retry_counts[page_idx]
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get a summary of error counts by type."""
        with self._lock:
            summary = {error_type.name: count for error_type, count in self.error_counts.items()}
            summary["RATE_LIMIT_HITS"] = self.rate_limit_hits
            return summary
    
    def get_failed_pages(self) -> Set[int]:
        """Get the set of pages that have failed."""
        with self._lock:
            return set(self.errors.keys())
    
    def get_page_errors(self, page_idx: int) -> List[Tuple[ErrorType, str]]:
        """Get all errors for a specific page."""
        with self._lock:
            return self.errors.get(page_idx, [])
    
    def clear_page_errors(self, page_idx: int):
        """Clear errors for a specific page (used when retrying)."""
        with self._lock:
            if page_idx in self.errors:
                del self.errors[page_idx]
            if page_idx in self.retry_counts:
                del self.retry_counts[page_idx]
    
    def clear_page_errors_and_get_failed_pages(self, page_idx: int) -> Set[int]:
        """Clear errors for a page and get the current set of failed pages atomically."""
        with self._lock:
            if page_idx in self.errors:
                del self.errors[page_idx]
            if page_idx in self.retry_counts:
                del self.retry_counts[page_idx]
            return set(self.errors.keys())

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
def gemini_api_call(prompt: str, pil_image: Image.Image, temperature: float, thinking_budget: int) -> Tuple[Optional[dict], str, bool]:
    """Execute a Gemini API call with retry logic and rate limit handling."""
    try:
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
    file_upload = None

    for attempt in range(MAX_RETRIES):
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_file_path_str = tmp.name
                tmp_file_path = Path(tmp_file_path_str)
                pil_image.save(tmp_file_path_str, "PNG")

            try:
                file_upload = client.files.upload(path=tmp_file_path_str)
                logging.debug(f"File uploaded via client.files.upload: {file_upload.uri}")
            except Exception as upload_err:
                error_msg = f"File upload failed: {upload_err}"
                logging.warning(f"Attempt {attempt+1}/{MAX_RETRIES}: {error_msg}")
                if tmp_file_path and tmp_file_path.exists():
                    try: 
                        tmp_file_path.unlink()
                        logging.debug(f"Deleted temp file after upload failure: {tmp_file_path}")
                    except OSError as e: 
                        logging.warning(f"Failed to delete temp file {tmp_file_path}: {e}")
                tmp_file_path = None
                if attempt < MAX_RETRIES - 1: time.sleep(2); continue
                else: break

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
                        thinking_budget=thinking_budget
                    )
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
            # Extract all token counts, including thoughts
            ptk = getattr(usage, 'prompt_token_count', 0) or 0
            ttk = getattr(usage, 'thoughts_token_count', 0) or 0
            ctk = getattr(usage, 'candidates_token_count', 0) or 0
            totk = getattr(usage, 'total_token_count', 0) or 0

            # Log the raw response and token information
            logging.info("-" * 80)
            logging.info("Raw LLM Response:")
            logging.info(response.text)
            logging.info("-" * 80)
            logging.info(f"Token Usage:")
            logging.info(f"Prompt tokens: {ptk}")
            logging.info(f"Thoughts tokens: {ttk}/{thinking_budget}")
            logging.info(f"Output tokens: {ctk}")
            logging.info(f"Total tokens: {totk}")
            logging.info("-" * 80)

            return ({"text": response.text, "usage": usage, "prompt_tokens": ptk, "thoughts_tokens": ttk, "candidate_tokens": ctk, "total_tokens": totk}, "", False)

        except google_exceptions.ResourceExhausted as e:
            is_rate_limit = True
            error_msg = f"API Error (Rate Limit): {e}"
            logging.warning(f"Attempt {attempt+1}/{MAX_RETRIES}: Rate limit error! Sleeping {BACKOFF_SLEEP_SECONDS}s...")
            if attempt < MAX_RETRIES - 1: time.sleep(BACKOFF_SLEEP_SECONDS)

        except Exception as e:
            msg = str(e)
            error_msg = f"API Error (General): {type(e).__name__} - {msg}"
            if "429" in msg or "rate limit" in msg.lower() or "resource has been exhausted" in msg.lower():
                is_rate_limit = True
                logging.warning(f"Attempt {attempt+1}/{MAX_RETRIES}: Rate limit detected! Sleeping {BACKOFF_SLEEP_SECONDS}s...")
                if attempt < MAX_RETRIES - 1: time.sleep(BACKOFF_SLEEP_SECONDS)
            else:
                logging.warning(f"Attempt {attempt+1}/{MAX_RETRIES}: {error_msg}", exc_info=True)
                if attempt < MAX_RETRIES - 1: time.sleep(2)

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

    final_error_msg = f"API failed after {MAX_RETRIES} attempts: {error_msg}"
    logging.error(final_error_msg, exc_info=True)
    return (None, final_error_msg, is_rate_limit)

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
        "page_idx": page_idx, "prompt_tokens": 0, "thoughts_tokens": 0, "candidate_tokens": 0, "total_tokens": 0,
        "success": False, "error_msg": "", "error_type": None, "api_failures": 0,
        "rate_limit_failures": 0, "parse_failures": 0, "last_parse_error": ""
    }
    logging.info(f"[Worker p.{page_idx:04d}] Processing {png_path.name}...")
    pil_image = None

    try:
        pil_image = Image.open(png_path)
        w, h = pil_image.size
        #logging.info(f"[Worker p.{page_idx:04d}] PNG info: {w}x{h}")

        initial_result, error, is_rate_limit = gemini_api_call(prompt_text, pil_image, temperature, thinking_budget)

        if not initial_result:
            result_info["error_msg"] = error
            result_info["error_type"] = "api"
            result_info["api_failures"] = MAX_RETRIES
            if is_rate_limit: 
                result_info["rate_limit_failures"] = MAX_RETRIES
                error_tracker.add_error(page_idx, ErrorType.RATE_LIMIT, error, True)
            else:
                error_tracker.add_error(page_idx, ErrorType.API, error)
            return result_info

        usage = initial_result["usage"]
        ptk = initial_result.get("prompt_tokens", getattr(usage, 'prompt_token_count', 0) or 0)
        ttk = initial_result.get("thoughts_tokens", getattr(usage, 'thoughts_token_count', 0) or 0)
        ctk = initial_result.get("candidate_tokens", getattr(usage, 'candidates_token_count', 0) or 0)
        totk = initial_result.get("total_tokens", getattr(usage, 'total_token_count', 0) or 0)
        result_info.update({"prompt_tokens": ptk, "thoughts_tokens": ttk, "candidate_tokens": ctk, "total_tokens": totk})
        resp_text = initial_result["text"]
        logging.info(f"[Worker p.{page_idx:04d}] Initial API usage -> input={ptk}, thoughts={ttk}, candidate={ctk}, total={totk}")

        parse_attempts = 0
        parsed = None
        max_parse_retries = MAX_RETRIES  # Limit parse retries
        
        while parse_attempts < max_parse_retries:
            try:
                parsed = parse_json_str(resp_text)
                logging.info(f"[Worker p.{page_idx:04d}] Parsed JSON successfully (attempt {parse_attempts+1}).")
                break
            except ValueError as ve:
                parse_attempts += 1
                result_info["parse_failures"] += 1
                parse_error_msg = str(ve)
                result_info["last_parse_error"] = parse_error_msg
                logging.error(f"[Worker p.{page_idx:04d}] JSON parse error ({parse_attempts}/{max_parse_retries}): {parse_error_msg}")
                logging.error(f"[Worker p.{page_idx:04d}] RAW RESPONSE:\n{'-'*60}\n{resp_text}\n{'-'*60}")

                if parse_attempts >= max_parse_retries:
                    logging.error(f"[Worker p.{page_idx:04d}] JSON parse failed after {max_parse_retries} attempts.")
                    result_info["error_msg"] = f"JSON parse failed after {max_parse_retries} attempts: {parse_error_msg}"
                    result_info["error_type"] = "parse"
                    error_tracker.add_error(page_idx, ErrorType.PARSE, parse_error_msg)
                    break

                # Retry API call if we haven't reached max parse retries
                logging.info(f"[Worker p.{page_idx:04d}] Re-calling API for parse retry ({parse_attempts+1}/{max_parse_retries})...")
                recall_res, recall_error, recall_rate_limit = gemini_api_call(prompt_text, pil_image, temperature, thinking_budget)
                result_info["api_failures"] += 1
                if recall_rate_limit: 
                    result_info["rate_limit_failures"] += 1
                    error_tracker.add_error(page_idx, ErrorType.RATE_LIMIT, recall_error, True)

                if not recall_res:
                    result_info["error_msg"] = f"API failed during parse retry: {recall_error}"
                    result_info["error_type"] = "api"
                    error_tracker.add_error(page_idx, ErrorType.API, recall_error)
                    logging.error(f"[Worker p.{page_idx:04d}] {result_info['error_msg']}")
                    break

                usage2 = recall_res["usage"]
                ptk2 = recall_res.get("prompt_tokens", getattr(usage2, 'prompt_token_count', 0) or 0)
                ttk2 = recall_res.get("thoughts_tokens", getattr(usage2, 'thoughts_token_count', 0) or 0)
                ctk2 = recall_res.get("candidate_tokens", getattr(usage2, 'candidates_token_count', 0) or 0)
                totk2 = recall_res.get("total_tokens", getattr(usage2, 'total_token_count', 0) or 0)
                result_info["prompt_tokens"] += ptk2
                result_info["thoughts_tokens"] += ttk2
                result_info["candidate_tokens"] += ctk2
                result_info["total_tokens"] += totk2
                resp_text = recall_res["text"]
                logging.info(f"[Worker p.{page_idx:04d}] Recall API usage -> input={ptk2}, thoughts={ttk2}, candidate={ctk2}, total={totk2}")

        if not parsed:
            if not result_info["error_msg"]:
                result_info["error_msg"] = f"Parsing failed after {max_parse_retries} attempts."
                result_info["error_type"] = "parse"
                error_tracker.add_error(page_idx, ErrorType.PARSE, result_info["error_msg"])
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

# Retry mechanism
def retry_failed_pages(failed_pages: Set[int], png_files: List[Path], prompt_text: str, 
                      temperature: float, thinking_budget: int, json_dir: Path, error_tracker: ErrorTracker) -> List[dict]:
    """Retry processing of failed pages with exponential backoff."""
    if not failed_pages:
        return []
    
    logging.info(f"Retrying {len(failed_pages)} failed pages...")
    retry_results = []
    
    for page_idx in sorted(failed_pages):
        if not error_tracker.increment_retry(page_idx):
            logging.warning(f"Page {page_idx:04d} has already been retried {MAX_RETRIES} times. Skipping.")
            continue
        
        retry_count = error_tracker.get_retry_count(page_idx)
        logging.info(f"Retry attempt {retry_count}/{MAX_RETRIES} for page {page_idx:04d}")
        
        png_file = next((f for f in png_files if f.name.startswith(f"page_{page_idx:04d}")), None)
        if not png_file:
            logging.error(f"Could not find PNG file for page {page_idx:04d} during retry")
            continue
        
        result = process_page(page_idx, png_file, prompt_text, temperature, thinking_budget, json_dir, error_tracker)
        retry_results.append(result)
        
        if result["success"]:
            logging.info(f"Retry successful for page {page_idx:04d}")
            # Clear errors and get updated failed pages atomically
            error_tracker.clear_page_errors_and_get_failed_pages(page_idx)
        else:
            logging.warning(f"Retry failed for page {page_idx:04d}: {result['error_msg']}")
    
    return retry_results

# After the ErrorTracker class, before main()

def extract_failed_pages_from_error_file(pdf_base_out_dir: Path, pdf_stem: str) -> List[int]:
    """Extract failed page numbers from error file."""
    err_file = pdf_base_out_dir / f"errors_{pdf_stem}.txt"
    if not err_file.exists():
        logging.error(f"Error file not found: {get_relative_path(err_file)}")
        return []
    
    try:
        content = err_file.read_text(encoding='utf-8')
        # Look for the Failed Pages Summary section
        match = re.search(r'Failed Pages Summary: \[([\d, ]+)\]', content)
        if not match:
            logging.error(f"Failed pages summary not found in error file: {get_relative_path(err_file)}")
            return []
        
        # Parse page numbers
        pages = [int(p.strip()) for p in match.group(1).split(',') if p.strip()]
        logging.info(f"Found {len(pages)} failed pages in error file: {pages}")
        return pages
    except Exception as e:
        logging.error(f"Error reading/parsing error file {get_relative_path(err_file)}: {e}")
        return []

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
                        failed_pages: Set[int], error_tracker: ErrorTracker) -> None:
    """Overwrite the error file with current error information."""
    err_file = pdf_base_out_dir / f"errors_{pdf_stem}.txt"
    logging.warning(f"{len(failed_pages)} page(s) failed. Writing details to: {get_relative_path(err_file)}")
    pdf_base_out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        total_pages = len(failed_pages)  # We only care about remaining failed pages
        error_summary = error_tracker.get_error_summary()
        
        with err_file.open("w", encoding="utf-8") as ef:
            # Basic Information
            ef.write(f"Errors for PDF: {pdf_name}\n")
            ef.write(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            ef.write(f"Failed Pages Summary: {sorted(list(failed_pages))}\n")
            ef.write(f"Prompt Used: {PROMPT_FILE_PATH.name}\n")
            ef.write(f"Total Failed Pages: {total_pages}\n\n")
            
            # Error Summary Section
            ef.write("Error Summary:\n")
            ef.write("-" * 40 + "\n")
            ef.write(f"API Call Final Failures (pages): {error_summary.get('API', 0)}\n")
            ef.write(f"JSON Parse Final Failures (pages): {error_summary.get('PARSE', 0)}\n")
            ef.write(f"Too Many Open Files (pages): {error_summary.get('FILE_LIMIT', 0)}\n")
            ef.write(f"Rate Limit Hits (occurrences): {error_summary.get('RATE_LIMIT_HITS', 0)}\n")
            ef.write(f"Other/Future Failures: {error_summary.get('OTHER', 0) + error_summary.get('FUTURE', 0)}\n\n")
            
            # Detailed Error Information:
            ef.write("Detailed Error Information:\n")
            ef.write("=" * 40 + "\n")
            
            # Group errors by type for better organization
            errors_by_type = defaultdict(list)
            for p_idx in sorted(failed_pages):
                page_errors = error_tracker.get_page_errors(p_idx)
                for error_type, error_msg in page_errors:
                    errors_by_type[error_type].append((p_idx, error_msg))
            
            # Write errors grouped by type
            for error_type in ErrorType:
                if errors_by_type[error_type]:
                    ef.write(f"\n{error_type.name} Errors:\n")
                    ef.write("-" * 20 + "\n")
                    for page_idx, error_msg in errors_by_type[error_type]:
                        retry_count = error_tracker.get_retry_count(page_idx)
                        ef.write(f"Page {page_idx:04d} (Retries: {retry_count+1}/{MAX_RETRIES}):\n")
                        ef.write(f"  Error: {error_msg}\n")
                        if "Content snippet" in error_msg:
                            ef.write("  " + "-" * 18 + "\n")
    except Exception as e:
        logging.error(f"Failed to write error file {get_relative_path(err_file)}: {e}", exc_info=True)

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
                successful_pages.add(page_num)
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

# Main execution
def main():
    parser = argparse.ArgumentParser(description="Gemini PDF->PNG->JSON->CSV Pipeline (Original API Style)")
    parser.add_argument("--pdf", required=True, help="PDF filename in data/pdfs/patent_pdfs/")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default=0.0)")
    parser.add_argument("--thinking_budget", type=int, default=DEFAULT_THINKING_BUDGET, 
                       help=f"Thinking budget for the model (default={DEFAULT_THINKING_BUDGET}, max={MAX_THINKING_BUDGET})")
    parser.add_argument("--max_workers", type=int, default=20, help="Max concurrent workers for page processing (default=20)")
    parser.add_argument("--retry_from_error_file", choices=['yes', 'no'], default='no',
                       help="Retry failed pages listed in the error file")
    parser.add_argument("--page", type=int, help="Process a single specific page number")

    args = parser.parse_args()

    # Validate thinking budget
    if args.thinking_budget < 0 or args.thinking_budget > MAX_THINKING_BUDGET:
        logging.error(f"Thinking budget must be between 0 and {MAX_THINKING_BUDGET}")
        sys.exit(1)

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

    # Determine pages to process
    if args.page:
        logging.info(f"Processing single page mode: page {args.page}")
        pages_to_process = [args.page]
        results, successful_pages = process_specific_pages(pages_to_process, png_dir, json_dir, 
                                                        task_prompt, args.temperature, args.thinking_budget, error_tracker, args.max_workers)
        page_results_list.extend(results)
        
        if successful_pages:
            logging.info(f"Successfully processed page {args.page}")
            # Create/update CSV after successful single page processing
            create_consolidated_csv(json_dir, pdf_base_out_dir, pdf_stem)
        else:
            logging.error(f"Failed to process page {args.page}")
    
    elif args.retry_from_error_file == 'yes':
        logging.info("Retrying failed pages from error file...")
        pages_to_process = extract_failed_pages_from_error_file(pdf_base_out_dir, pdf_stem)
        if pages_to_process:
            results, successful_pages = process_specific_pages(pages_to_process, png_dir, json_dir,
                                                            task_prompt, args.temperature, args.thinking_budget, error_tracker, args.max_workers)
            page_results_list.extend(results)
            
            if successful_pages:
                logging.info(f"Successfully processed pages: {sorted(list(successful_pages))}")
                # Create/update CSV after successful retry from error file
                create_consolidated_csv(json_dir, pdf_base_out_dir, pdf_stem)
                
                # Update error file if there are still failures
                failed_pages = set(pages_to_process) - successful_pages
                if failed_pages:
                    error_tracker.errors = {p: error_tracker.errors[p] for p in failed_pages}
                    overwrite_error_file(pdf_base_out_dir, pdf_stem, pdf_name, failed_pages, error_tracker)
            else:
                logging.warning("No pages were successfully processed")
            # If error file exists, delete it since there are no failed pages
            err_file = pdf_base_out_dir / f"errors_{pdf_stem}.txt"
            if err_file.exists():
                try:
                    err_file.unlink()
                    logging.info(f"Deleted error file as there are no failed pages: {get_relative_path(err_file)}")
                except Exception as e:
                    logging.error(f"Failed to delete error file {get_relative_path(err_file)}: {e}")
            return
        else:
            logging.warning("No failed pages found in error file or error file not found.")
            # If error file exists, delete it since there are no failed pages
            err_file = pdf_base_out_dir / f"errors_{pdf_stem}.txt"
            if err_file.exists():
                try:
                    err_file.unlink()
                    logging.info(f"Deleted error file as there are no failed pages: {get_relative_path(err_file)}")
                except Exception as e:
                    logging.error(f"Failed to delete error file {get_relative_path(err_file)}: {e}")
            return
    
    else:
        # Original parallel processing logic for full PDF
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

        # Handle retries for the full run
        failed_pages = error_tracker.get_failed_pages()
        if failed_pages:
            logging.info(f"Retrying {len(failed_pages)} failed pages...")
            retry_results = retry_failed_pages(failed_pages, valid_png_files, task_prompt, args.temperature, args.thinking_budget, json_dir, error_tracker)
            page_results_list.extend(retry_results)

        # Create CSV after all processing and retries are done
        create_consolidated_csv(json_dir, pdf_base_out_dir, pdf_stem)
        
        # Create error file if there are still failures
        failed_pages = error_tracker.get_failed_pages()
        if failed_pages:
            overwrite_error_file(pdf_base_out_dir, pdf_stem, pdf_name, failed_pages, error_tracker)

    pdf_tokens = defaultdict(int)
    for r in page_results_list:
        pdf_tokens['prompt'] += r.get("prompt_tokens", 0)
        pdf_tokens['candidate'] += r.get("candidate_tokens", 0)
        pdf_tokens['total'] += r.get("total_tokens", 0)

    global_tokens['prompt'] += pdf_tokens['prompt']
    global_tokens['candidate'] += pdf_tokens['candidate']
    global_tokens['total'] += pdf_tokens['total']

    error_summary = error_tracker.get_error_summary()
    failed_pages = error_tracker.get_failed_pages()

    if failed_pages:
         err_file = pdf_base_out_dir / f"errors_{pdf_stem}.txt"
         logging.warning(f"{len(failed_pages)} page(s) failed. Writing details to: {get_relative_path(err_file)}")
         pdf_base_out_dir.mkdir(parents=True, exist_ok=True)
         try:
             total_pages = len(png_files)
             success_rate = ((total_pages - len(failed_pages)) / total_pages) * 100 if total_pages > 0 else 0
             
             with err_file.open("w", encoding="utf-8") as ef:
                 # Basic Information
                 ef.write(f"Errors for PDF: {pdf_name}\n")
                 ef.write(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                 ef.write(f"Failed Pages Summary: {sorted(list(failed_pages))}\n")  # Add failed pages list
                 ef.write(f"Prompt Used: {PROMPT_FILE_PATH.name}\n")
                 ef.write(f"Total Pages: {total_pages}\n")
                 ef.write(f"Total Failed Pages: {len(failed_pages)}\n")
                 ef.write(f"Success Rate: {success_rate:.1f}%\n\n")
                 
                 # Error Summary Section
                 ef.write("Error Summary:\n")
                 ef.write("-" * 40 + "\n")
                 ef.write(f"API Call Final Failures (pages): {error_summary.get('API', 0)}\n")
                 ef.write(f"JSON Parse Final Failures (pages): {error_summary.get('PARSE', 0)}\n")
                 ef.write(f"Too Many Open Files (pages): {error_summary.get('FILE_LIMIT', 0)}\n")
                 ef.write(f"Rate Limit Hits (occurrences): {error_summary.get('RATE_LIMIT_HITS', 0)}\n")
                 ef.write(f"Other/Future Failures: {error_summary.get('OTHER', 0) + error_summary.get('FUTURE', 0)}\n\n")
                 
                 # Detailed Error Information - Organized by Error Type
                 ef.write("Detailed Error Information:\n")
                 ef.write("=" * 40 + "\n")
                 
                 # Group errors by type for better organization
                 errors_by_type = defaultdict(list)
                 for p_idx in sorted(failed_pages):
                     page_errors = error_tracker.get_page_errors(p_idx)
                     for error_type, error_msg in page_errors:
                         errors_by_type[error_type].append((p_idx, error_msg))
                 
                 # Write errors grouped by type
                 for error_type in ErrorType:
                     if errors_by_type[error_type]:
                         ef.write(f"\n{error_type.name} Errors:\n")
                         ef.write("-" * 20 + "\n")
                         for page_idx, error_msg in errors_by_type[error_type]:
                             retry_count = error_tracker.get_retry_count(page_idx)
                             ef.write(f"Page {page_idx:04d} (Retries: {retry_count+1}/{MAX_RETRIES}):\n")
                             ef.write(f"  Error: {error_msg}\n")
                             if "Content snippet" in error_msg:
                                 ef.write("  " + "-" * 18 + "\n")
         except Exception as e:
             logging.error(f"Failed to write error file {get_relative_path(err_file)}: {e}", exc_info=True)

    logging.info(f"Starting consolidation for {pdf_stem} (ffill, filter, id, rename, reorder)")
    all_json_files = sorted(json_dir.glob("page_*.json"))
    consolidated_data_with_page = []
    read_json_count = 0
    read_errors = 0
    total_items_read = 0

    if not all_json_files:
        logging.warning(f"No JSON files found in {get_relative_path(json_dir)} to consolidate for {pdf_stem}")
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

                if page_num in failed_pages:
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
                            logging.debug(f"Skipping non-dict item within list in {fpath.name}: {type(item)}")
                elif isinstance(content, dict):
                    consolidated_data_with_page.append({'data': content, 'page': page_num})
                    total_items_read += 1
                else:
                    logging.warning(f"Unexpected or empty content type ({type(content)}) in {fpath.name}. Skipping content.")

            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON {get_relative_path(fpath)}: {e}. Skipping.")
                read_errors += 1
            except ValueError as e:
                logging.error(f"Error processing page number for {get_relative_path(fpath)}: {e}", exc_info=True)
                read_errors += 1
            except Exception as e:
                logging.error(f"Error reading JSON {get_relative_path(fpath)}: {e}", exc_info=True)
                read_errors += 1

        logging.info(f"Read {read_json_count} JSON files, found {total_items_read} processable items.")
        if read_errors > 0:
            logging.warning(f"Encountered {read_errors} errors during JSON reading/processing.")

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
                "page_number": page_num
            })

    logging.info(f"Generated {len(initial_csv_data)} initial rows.")

    if initial_csv_data:
        final_csv_path = pdf_base_out_dir / f"{pdf_stem}.csv"
        try:
            logging.info("Creating DataFrame...")
            df = pd.DataFrame(initial_csv_data)

            logging.info("Performing forward fill on 'category' column...")
            df['category'] = df['category'].ffill()
            logging.info("Forward fill complete.")

            logging.info("Filtering rows with empty 'entry'...")
            original_row_count = len(df)
            df = df[df['entry'].notna() & (df['entry'] != '')]
            rows_removed = original_row_count - len(df)
            logging.info(f"Filtering complete. Removed {rows_removed} rows.")

            if not df.empty:
                logging.info("Renaming 'page_number' column to 'page'...")
                df.rename(columns={'page_number': 'page'}, inplace=True)

                logging.info("Adding sequential 'id' column...")
                df['id'] = range(1, len(df) + 1)

                logging.info("Reordering columns to 'id', 'page', 'entry', 'category'...")
                df = df[['id', 'page', 'entry', 'category']]

                logging.info(f"Saving final DataFrame with {len(df)} rows to CSV...")
                final_csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(final_csv_path, index=False, encoding='utf-8', quoting=1)

                logging.info(f"Final CSV saved: {get_relative_path(final_csv_path)} ({len(df)} rows)")
            else:
                logging.warning(f"DataFrame became empty after filtering for PDF '{pdf_stem}'. Final CSV file will not be created.")

        except Exception as e_pandas:
            logging.error(f"Error during Pandas manipulation or CSV writing for {get_relative_path(final_csv_path)}: {e_pandas}", exc_info=True)
    else:
        if total_items_read > 0:
            logging.warning(f"No dictionary items found or processed from JSON files for PDF '{pdf_stem}' after initial processing step. Final CSV file will not be created.")
        elif read_json_count > 0:
            logging.warning(f"JSON files were read for PDF '{pdf_stem}', but contained no processable dictionary items. Final CSV file will not be created.")
        else:
            logging.warning(f"No JSON files were successfully read or no data found for PDF '{pdf_stem}'. Final CSV file will not be created.")

    pdf_duration = time.time() - prompt_start_time
    logging.info(f"Tokens (PDF: {pdf_name}): Prompt Tokens={pdf_tokens['prompt']}, Candidate Tokens={pdf_tokens['candidate']}, Total Tokens={pdf_tokens['total']}")
    logging.info("")
    logging.info(f"PDF {pdf_name} Processing Finished ({format_duration(pdf_duration)})")
    logging.info(f"Global Tokens Running Total: Prompt Tokens={global_tokens['prompt']}, Candidate Tokens={global_tokens['candidate']}, Total Tokens={global_tokens['total']}")
    logging.info("-" * 80)

    pdf_total_duration = time.time() - pdf_start
    logging.info(f"Finished PDF: {pdf_name} (Total time: {format_duration(pdf_total_duration)})")
    
    if failed_pages:
        logging.warning(f"PDF Summary: API Failures={error_summary.get('API', 0)}, Parse Failures={error_summary.get('PARSE', 0)}, "
                       f"File Limit Errors={error_summary.get('FILE_LIMIT', 0)}, Rate Limit Hits={error_summary.get('RATE_LIMIT_HITS', 0)}, "
                       f"Other Failures={error_summary.get('OTHER', 0) + error_summary.get('FUTURE', 0)}")

    logging.info("")
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