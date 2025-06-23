import os
import re
import json
import time
import logging
import tempfile
import pandas as pd
import google.genai as genai
from google.genai import types
from pathlib import Path
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

# --- Configuration ---
# Load environment variables
try:
    # Adjust path to be relative to this script's location
    # src/benchmarking/core/llm_processing.py -> project_root
    project_root = Path(__file__).resolve().parents[3]
    env_path = project_root / "config" / ".env"
    load_dotenv(dotenv_path=env_path)
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
except Exception as e:
    logging.error(f"Error loading environment variables: {e}")
    API_KEY = None

# NOTE: The genai.configure call has been removed as it's not compatible with the user's environment.
# The API key will be passed directly to the client in the gemini_api_call function.

MAX_RETRIES = 3
BACKOFF_FACTOR = 2
MAX_OUTPUT_TOKENS = 8192

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_json_from_response(response_text: str) -> dict:
    """Extracts and parses a JSON object from a model's text response."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # Fallback if no markdown code block is found
        json_str = response_text.strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {e}\nResponse snippet: {json_str[:200]}...")
        raise ValueError("Invalid JSON response from model.") from e

def gemini_api_call(model_name: str, prompt: str, pil_image: Image.Image) -> dict:
    """
    Makes a call to the Gemini API with a given image and prompt.
    Includes retry logic and model-specific configurations.
    """
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        logging.error(f"Failed to initialize Gemini Client: {e}")
        raise

    # --- Model-specific configuration ---
    config_args = {
        "temperature": 0.0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_mime_type": "application/json",
    }

    if "2.5" in model_name:
        logging.info("Using Gemini 2.5-specific config with dynamic thinking budget.")
        config_args["thinking_config"] = types.ThinkingConfig(
            thinking_budget=-1,
            include_thoughts=True
        )

    config = types.GenerateContentConfig(**config_args)
    file_upload = None  # Ensure it exists for the finally block

    for attempt in range(MAX_RETRIES):
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
                pil_image.save(tmp_file.name, "PNG")

                logging.info(f"Uploading image for API call (Attempt {attempt + 1})...")
                file_upload = client.files.upload(file=tmp_file.name)

                logging.info(f"Making generate_content call for model {model_name}...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_uri(
                            file_uri=file_upload.uri,
                            mime_type=file_upload.mime_type,
                        ),
                        prompt
                    ],
                    config=config
                )

                return parse_json_from_response(response.text)

        except Exception as e:
            logging.warning(f"API call attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = BACKOFF_FACTOR ** attempt
                logging.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logging.error("All API call retries failed.")
                raise
        finally:
            if file_upload:
                try:
                    client.files.delete(name=file_upload.name)
                    logging.debug(f"Deleted temp file on API server: {file_upload.name}")
                except Exception as e:
                    logging.warning(f"Failed to delete uploaded file {file_upload.name}: {e}")

    raise RuntimeError("Should not be reached; indicates a problem in retry logic.")

def process_pdf(model_name: str, prompt_text: str, pdf_path: Path, output_dir: Path):
    """
    Processes a single PDF, converting it to images, calling the LLM for each page,
    and saving the consolidated results to a CSV file.
    """
    if not API_KEY:
        logging.error("API_KEY is not configured. Aborting PDF processing.")
        return

    logging.info(f"Starting processing for PDF: {pdf_path.name}")
    pdf_stem = pdf_path.stem
    output_csv_path = output_dir / f"{pdf_stem}.csv"
    
    if output_csv_path.exists():
        logging.info(f"CSV for {pdf_path.name} already exists. Skipping.")
        return

    all_rows = []

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            logging.info("Converting PDF to images...")
            images = convert_from_path(pdf_path, output_folder=temp_dir)
            
            for i, image in enumerate(images):
                page_num = i + 1
                logging.info(f"Processing page {page_num}/{len(images)} of {pdf_path.name}")
                try:
                    page_objs = gemini_api_call(model_name, prompt_text, image)
                    if not isinstance(page_objs, list):
                        logging.warning(f"Expected a list from LLM output, got {type(page_objs)}. Wrapping in list.")
                        page_objs = [page_objs]
                    for obj in page_objs:
                        if not isinstance(obj, dict):
                            logging.warning(f"Expected dict in LLM output array, got {type(obj)}. Skipping.")
                            continue
                        # Only extract 'entry' and 'category' keys
                        entry_value = obj.get("entry", None)
                        category_value = obj.get("category", None)
                        all_rows.append({
                            "entry": entry_value,
                            "category": category_value
                        })
                except Exception as e:
                    logging.error(f"Failed to process page {page_num} for {pdf_path.name}: {e}")
                    # Optionally, add a placeholder for failed pages
                    all_rows.append({
                        "pdf_filename": pdf_path.name,
                        "page_number": page_num,
                        "error": str(e)
                    })
    except Exception as e:
        logging.error(f"An unexpected error occurred during processing of {pdf_path.name}: {e}")
        return

    if not all_rows:
        logging.warning(f"No data was extracted from {pdf_path.name}. CSV will not be created.")
        return

    # Robust DataFrame creation: forward-fill category, filter, and keep only 'entry' and 'category'
    try:
        df = pd.DataFrame(all_rows)
        # Forward-fill category
        if "category" in df.columns:
            df["category"] = df["category"].ffill()
        # Filter out rows with empty or missing 'entry'
        df = df[df.get("entry").notna() & (df["entry"] != "")]
        # Add sequential 'id' column as the first column
        df = df.reset_index(drop=True)
        df.insert(0, 'id', range(1, len(df) + 1))
        # Only keep 'id', 'entry', and 'category' columns (in that order)
        keep_cols = [col for col in ['id', 'entry', 'category'] if col in df.columns]
        df = df[keep_cols]
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        logging.info(f"Successfully created CSV: {output_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save DataFrame to CSV for {pdf_path.name}: {e}") 