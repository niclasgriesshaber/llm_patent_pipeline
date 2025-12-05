import argparse
import logging
import json
import pandas as pd
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Core modules are now in the same directory
from core.llm_processing import process_pdf
from core.benchmarking import run_comparison, run_unified_comparison

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define project root (scripts directory is 3 levels down from project root)
project_root = Path(__file__).resolve().parents[3]
BENCHMARKING_ROOT = project_root / 'data' / 'benchmarking'
PROMPTS_DIR = project_root / 'src' / 'benchmarking' / 'prompts' / '01_dataset_construction'
GT_XLSX_DIR = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'perfect_transcriptions_xlsx'
SAMPLED_PDFS_DIR = BENCHMARKING_ROOT / 'input_data' / 'sampled_pdfs'

MODELS = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-3-pro-preview']

# --- Main Functions ---

def is_special_volume(pdf_path: Path) -> bool:
    """Check if the PDF is from 1878 or 1879 special volumes."""
    filename = pdf_path.name
    return '1878' in filename or '1879' in filename

def get_prompt_for_pdf(pdf_path: Path, prompt_name: str) -> str:
    """Get the appropriate prompt text for a PDF file."""
    if is_special_volume(pdf_path):
        # Use special volumes prompt for 1878/1879
        special_prompt_file = PROMPTS_DIR / 'special_volumes_prompt.txt'
        if not special_prompt_file.exists():
            logging.error(f"Special volumes prompt file not found: {special_prompt_file}. Aborting.")
            raise FileNotFoundError(f"Special volumes prompt file not found: {special_prompt_file}")
        return special_prompt_file.read_text(encoding='utf-8')
    else:
        # Use regular prompt for other years
        prompt_file = PROMPTS_DIR / prompt_name
        if not prompt_file.exists():
            logging.error(f"Prompt file not found: {prompt_file}. Aborting.")
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        return prompt_file.read_text(encoding='utf-8')

def run_single_benchmark(model_name: str, prompt_name: str, max_workers: int = 20, threshold: float = 0.85):
    """
    Executes the full benchmarking pipeline for a single model and prompt combination.
    """
    logging.info(f"--- Starting benchmark for model: [{model_name}] with prompt: [{prompt_name}] ---")

    prompt_file = PROMPTS_DIR / prompt_name
    if not prompt_file.exists():
        logging.error(f"Prompt file not found: {prompt_file}. Aborting.")
        return

    try:
        prompt_text = prompt_file.read_text(encoding='utf-8')
    except Exception as e:
        logging.error(f"Failed to read prompt file {prompt_file}: {e}")
        return

    # Define output directories
    prompt_stem = prompt_file.stem
    run_output_dir = BENCHMARKING_ROOT / 'results' / '01_dataset_construction' / model_name / prompt_stem
    llm_csv_output_dir = run_output_dir / 'llm_csv'
    
    run_output_dir.mkdir(parents=True, exist_ok=True)
    llm_csv_output_dir.mkdir(exist_ok=True)
    
    logging.info(f"Output will be saved in: {run_output_dir}")

    # 1. Process PDFs to generate LLM CSVs (skip if already exist)
    pdf_files = list(SAMPLED_PDFS_DIR.glob('*.pdf'))
    if not pdf_files:
        logging.error(f"No PDFs found in {SAMPLED_PDFS_DIR}. Cannot proceed.")
        return

    logging.info(f"Found {len(pdf_files)} PDFs to process.")
    processed_count = 0
    skipped_count = 0
    
    # Filter out PDFs that already have corresponding CSV files
    pdfs_to_process = []
    for pdf_path in pdf_files:
        csv_filename = f"{pdf_path.stem}.csv"
        csv_path = llm_csv_output_dir / csv_filename
        
        if csv_path.exists():
            logging.info(f"CSV already exists for {pdf_path.name}, skipping API call.")
            skipped_count += 1
        else:
            pdfs_to_process.append(pdf_path)
    
    # Process remaining PDFs in parallel
    if pdfs_to_process:
        logging.info(f"Processing {len(pdfs_to_process)} PDFs in parallel with {max_workers} workers...")
        
        def process_single_pdf(pdf_path):
            """Process a single PDF and return success status."""
            try:
                # Get the appropriate prompt for this PDF
                pdf_prompt_text = get_prompt_for_pdf(pdf_path, prompt_name)
                prompt_type = "special volumes" if is_special_volume(pdf_path) else "regular"
                logging.info(f"Processing PDF: {pdf_path.name} with {prompt_type} prompt")
                process_pdf(
                    model_name=model_name,
                    prompt_text=pdf_prompt_text,
                    pdf_path=pdf_path,
                    output_dir=llm_csv_output_dir
                )
                return True
            except Exception as e:
                logging.error(f"Error processing {pdf_path.name}: {e}")
                return False
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {executor.submit(process_single_pdf, pdf_path): pdf_path for pdf_path in pdfs_to_process}
            
            # Process completed tasks
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    success = future.result()
                    if success:
                        processed_count += 1
                        logging.info(f"Successfully processed: {pdf_path.name}")
                    else:
                        logging.error(f"Failed to process: {pdf_path.name}")
                except Exception as e:
                    logging.error(f"Exception occurred while processing {pdf_path.name}: {e}")
    
    logging.info(f"PDF processing complete. Processed: {processed_count}, Skipped: {skipped_count}")
    logging.info("--- Starting unified comparison phase. ---")

    # 2. Run unified comparison with three-table format
    student_xlsx_dir = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'student_transcriptions_xlsx'
    perfect_xlsx_dir = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'perfect_transcriptions_xlsx'
    
    if student_xlsx_dir.exists() and perfect_xlsx_dir.exists():
        logging.info("Running unified three-table comparison...")
        unified_results = run_unified_comparison(
            llm_csv_dir=llm_csv_output_dir,
            student_xlsx_dir=student_xlsx_dir,
            perfect_xlsx_dir=perfect_xlsx_dir,
            sampled_pdfs_dir=SAMPLED_PDFS_DIR,
            output_dir=run_output_dir,
            fuzzy_threshold=threshold
        )
        
        if unified_results:
            logging.info("Unified comparison completed successfully.")
            logging.info(f"Total files processed: {unified_results.get('total_files', 0)}")
            logging.info(f"Comparison results generated: {unified_results.get('comparison_results', 0)}")
        else:
            logging.warning("No unified comparison results generated.")
    else:
        logging.warning(f"Required directories not found:")
        if not student_xlsx_dir.exists():
            logging.warning(f"  - Student transcriptions: {student_xlsx_dir}")
        if not perfect_xlsx_dir.exists():
            logging.warning(f"  - Perfect transcriptions: {perfect_xlsx_dir}")
    
    logging.info(f"--- Benchmark finished for model: [{model_name}] with prompt: [{prompt_name}] ---")

def main():
    """
    Main function to parse arguments and orchestrate the benchmarking runs.
    """
    parser = argparse.ArgumentParser(description="Run the benchmarking pipeline.")
    parser.add_argument(
        '--model',
        type=str,
        choices=MODELS,
        help='The name of the model to benchmark.'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='The filename of the prompt to use (e.g., "construction_v0.0_prompt.txt").'
    )
    parser.add_argument(
        '--run-all',
        action='store_true',
        help='Run the pipeline for all available models and prompts.'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=20,
        help='Maximum number of worker threads for parallel PDF processing. Default: 20'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Fuzzy matching threshold for patent entry matching (0.0-1.0). Default: 0.85'
    )
    
    args = parser.parse_args()
    
    # Validate threshold parameter
    if not (0.0 <= args.threshold <= 1.0):
        logging.error(f"Threshold must be between 0.0 and 1.0, got: {args.threshold}")
        sys.exit(1)

    if args.run_all:
        logging.info("--- Starting full benchmark run for all models and prompts. ---")
        prompt_files = [f.name for f in PROMPTS_DIR.glob('*.txt')]
        
        for model_name in MODELS:
            for prompt_name in prompt_files:
                run_single_benchmark(model_name, prompt_name, args.max_workers, args.threshold)
        
        logging.info("--- All benchmark runs complete. ---")

    elif args.model and args.prompt:
        run_single_benchmark(args.model, args.prompt, args.max_workers, args.threshold)
        logging.info("--- Single benchmark run complete. ---")

    else:
        parser.print_help()
        logging.warning("Please specify --model and --prompt, or use --run-all.")

if __name__ == "__main__":
    main() 