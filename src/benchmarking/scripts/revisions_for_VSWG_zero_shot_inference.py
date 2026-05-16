#!/usr/bin/env python3
"""
=============================================================================
REVISIONS FOR VSWG — Zero-Shot Inference Script
=============================================================================

Purpose:
    This script implements a zero-shot digitization approach for the VSWG paper
    revisions. Instead of the multi-stage pipeline (extraction → cleaning →
    variable extraction), it extracts patent entries AND structured variables
    in a single LLM call per page.

Model: gemini-3.1-pro-preview
Approach: One API call per sampled PDF page (41 pages total)
Output: CSV files with columns [id, entry, patent_id, name, location, description, date]

Design choices:
    - NO retry logic: if a page fails, we log it and move on
    - Parallel execution: all 41 pages processed concurrently
    - Re-runnable: skips pages that already have a CSV output file
    - Temperature 0.0, dynamic thinking (budget=-1), max_output_tokens=20000

Usage:
    python revisions_for_VSWG_zero_shot_inference.py

    To re-run only failed pages, simply run the script again — it will skip
    pages that already have CSV output.
=============================================================================
"""

import re
import json
import logging
import tempfile
import pandas as pd
import fitz  # PyMuPDF — used instead of pdf2image+poppler for PDF→PNG conversion
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from dotenv import load_dotenv
import os

import google.genai as genai
from google.genai import types

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load API key from config/.env (same approach as existing pipeline scripts)
PROJECT_ROOT_FOR_ENV = Path(__file__).resolve().parents[3]
load_dotenv(dotenv_path=PROJECT_ROOT_FOR_ENV / "config" / ".env")
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found. Set it in config/.env or as an environment variable.")

# Model configuration
MODEL_NAME = "gemini-3.1-pro-preview"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 20000
THINKING_BUDGET = -1  # Dynamic thinking

# Paths — all relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLED_PDFS_DIR = PROJECT_ROOT / "data" / "benchmarking" / "input_data" / "sampled_pdfs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmarking" / "results" / "revisions_for_VSWG_zero_shot" / "llm_csv"
PROMPTS_DIR = PROJECT_ROOT / "src" / "benchmarking" / "prompts" / "revisions_for_VSWG_zero_shot"

# Parallel workers — all 41 pages processed concurrently
MAX_WORKERS = 41

# =============================================================================
# PROMPT LOADING
# =============================================================================

def is_special_volume(pdf_path: Path) -> bool:
    """
    Check if the PDF is from 1878 or 1879 (special volumes with different format).
    These volumes use "P. R. XXXX." registration numbers instead of leading patent IDs.
    """
    filename = pdf_path.name
    return '1878' in filename or '1879' in filename


def load_prompt(pdf_path: Path) -> str:
    """
    Load the appropriate zero-shot prompt for a given PDF.
    Uses special_volumes_prompt.txt for 1878/1879, zero_shot_prompt.txt otherwise.
    """
    if is_special_volume(pdf_path):
        prompt_file = PROMPTS_DIR / "special_volumes_prompt.txt"
    else:
        prompt_file = PROMPTS_DIR / "zero_shot_prompt.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    return prompt_file.read_text(encoding="utf-8")


# =============================================================================
# JSON PARSING
# =============================================================================

def parse_json_from_response(response_text: str) -> list:
    """
    Extract and parse a JSON array from the model's response text.
    Handles code block wrappers and BOM markers robustly.
    """
    if not isinstance(response_text, str):
        raise ValueError(f"Model response is not a string (type={type(response_text)})")

    # Try to extract JSON from markdown code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
    else:
        # Fallback: use entire response, stripping backticks
        json_str = response_text.strip().strip("`")

    # Remove BOM if present
    if json_str.startswith('\ufeff'):
        json_str = json_str[1:]

    try:
        result = json.loads(json_str)
        if not isinstance(result, list):
            raise ValueError(f"Expected JSON array, got {type(result)}")
        return result
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}\nSnippet: {json_str[:300]}...")


# =============================================================================
# API CALL (single attempt, no retry)
# =============================================================================

def call_gemini(prompt_text: str, pil_image: Image.Image) -> list:
    """
    Make a single API call to Gemini-3.1-Pro-Preview with an image and prompt.

    NO retry logic — if this fails, the page is marked as failed.
    Returns a list of extracted JSON objects on success, raises on failure.
    """
    client = genai.Client(api_key=API_KEY)

    # Configuration: temperature=0.0, dynamic thinking, high output token limit
    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(
            thinking_budget=THINKING_BUDGET,
            include_thoughts=False
        )
    )

    file_upload = None
    try:
        # Save image to temporary file for upload
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
            pil_image.save(tmp_file.name, "PNG")
            file_upload = client.files.upload(file=tmp_file.name)

            # Make the API call with image + prompt
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    types.Part.from_uri(
                        file_uri=file_upload.uri,
                        mime_type=file_upload.mime_type,
                    ),
                    prompt_text
                ],
                config=config
            )

        # Validate response
        if not response or not getattr(response, 'text', None):
            error_msg = "API returned empty response or no text"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                block_reason = getattr(response.prompt_feedback, 'block_reason', None)
                if block_reason:
                    error_msg += f" (Block Reason: {block_reason})"
            raise RuntimeError(error_msg)

        return parse_json_from_response(response.text)

    finally:
        # Clean up uploaded file from API server
        if file_upload:
            try:
                client.files.delete(name=file_upload.name)
            except Exception:
                pass  # Non-critical cleanup failure


# =============================================================================
# SINGLE PAGE PROCESSING
# =============================================================================

def process_single_pdf(pdf_path: Path) -> dict:
    """
    Process a single sampled PDF page:
    1. Convert PDF to PNG image
    2. Call Gemini with zero-shot prompt
    3. Parse response into structured entries
    4. Save as CSV

    Returns a dict with 'success', 'filename', and optionally 'error'.
    """
    pdf_stem = pdf_path.stem
    output_csv = OUTPUT_DIR / f"{pdf_stem}.csv"

    # Skip if CSV already exists (enables re-running for failures only)
    if output_csv.exists():
        logging.info(f"[SKIP] CSV already exists: {pdf_stem}.csv")
        return {"success": True, "filename": pdf_path.name, "skipped": True}

    try:
        # Step 1: Load the appropriate prompt
        prompt_text = load_prompt(pdf_path)
        prompt_type = "special volumes" if is_special_volume(pdf_path) else "regular"
        logging.info(f"[START] Processing {pdf_path.name} ({prompt_type} prompt)")

        # Step 2: Convert PDF to PNG image using PyMuPDF (each sampled PDF is a single page)
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            raise RuntimeError(f"PDF has no pages: {pdf_path.name}")
        # Render first page at 300 DPI for high quality
        page = doc[0]
        mat = fitz.Matrix(300 / 72, 300 / 72)  # Scale from 72 DPI to 300 DPI
        pix = page.get_pixmap(matrix=mat)
        pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        # Step 3: Call the Gemini API (single attempt, no retry)
        page_objects = call_gemini(prompt_text, pil_image)

        # Step 4: Parse results into rows for the CSV
        rows = []
        entry_id = 1
        for obj in page_objects:
            if not isinstance(obj, dict):
                logging.warning(f"[{pdf_stem}] Non-dict object in response, skipping: {type(obj)}")
                continue

            # Skip category-only objects (we only want patent entries)
            if "category" in obj and "entry" not in obj:
                continue

            # Extract entry and variables
            entry_text = obj.get("entry", None)
            if not entry_text or str(entry_text).strip() == "":
                continue

            rows.append({
                "id": entry_id,
                "entry": str(entry_text).strip(),
                "patent_id": str(obj.get("patent_id", "NaN")).strip(),
                "name": str(obj.get("name", "NaN")).strip(),
                "location": str(obj.get("location", "NaN")).strip(),
                "description": str(obj.get("description", "NaN")).strip(),
                "date": str(obj.get("date", "NaN")).strip(),
            })
            entry_id += 1

        # Step 5: Save to CSV
        if not rows:
            logging.warning(f"[{pdf_stem}] No valid entries extracted — creating empty CSV")

        df = pd.DataFrame(rows, columns=["id", "entry", "patent_id", "name", "location", "description", "date"])
        df.to_csv(output_csv, index=False)
        logging.info(f"[DONE] {pdf_stem}: {len(rows)} entries saved to CSV")

        return {"success": True, "filename": pdf_path.name, "entries": len(rows)}

    except Exception as e:
        logging.error(f"[FAIL] {pdf_path.name}: {e}")
        return {"success": False, "filename": pdf_path.name, "error": str(e)}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function: discovers PDFs, runs inference in parallel, reports results.
    """
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover all sampled PDFs
    pdf_files = sorted(SAMPLED_PDFS_DIR.glob("*.pdf"))
    if not pdf_files:
        logging.error(f"No PDFs found in {SAMPLED_PDFS_DIR}")
        return

    logging.info(f"Found {len(pdf_files)} sampled PDFs to process")
    logging.info(f"Model: {MODEL_NAME} | Temperature: {TEMPERATURE} | "
                 f"Thinking: dynamic | Max tokens: {MAX_OUTPUT_TOKENS}")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    logging.info("=" * 70)

    # Process all PDFs in parallel (no retry — single attempt per page)
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pdf = {
            executor.submit(process_single_pdf, pdf_path): pdf_path
            for pdf_path in pdf_files
        }
        for future in as_completed(future_to_pdf):
            result = future.result()
            results.append(result)

    # ==========================================================================
    # SUMMARY REPORT
    # ==========================================================================
    succeeded = [r for r in results if r["success"] and not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    failed = [r for r in results if not r["success"]]

    logging.info("=" * 70)
    logging.info("INFERENCE COMPLETE — SUMMARY")
    logging.info("=" * 70)
    logging.info(f"  Total PDFs:    {len(pdf_files)}")
    logging.info(f"  Succeeded:     {len(succeeded)}")
    logging.info(f"  Skipped:       {len(skipped)} (CSV already existed)")
    logging.info(f"  Failed:        {len(failed)}")

    if failed:
        logging.info("")
        logging.info("FAILED PAGES (re-run the script to retry these):")
        for r in sorted(failed, key=lambda x: x["filename"]):
            logging.info(f"  - {r['filename']}: {r['error']}")

    if succeeded:
        total_entries = sum(r.get("entries", 0) for r in succeeded)
        logging.info(f"\n  Total entries extracted this run: {total_entries}")


if __name__ == "__main__":
    main()
