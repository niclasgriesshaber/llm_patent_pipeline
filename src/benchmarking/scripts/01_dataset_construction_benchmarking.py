import argparse
import logging
import json
import pandas as pd
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure the 'core' module can be found
# The script is in src/benchmarking/scripts, so we add src to the path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root / 'src'))

from benchmarking.core.llm_processing import process_pdf
from benchmarking.core.benchmarking import run_comparison
from create_dashboard import create_dashboard

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BENCHMARKING_ROOT = project_root / 'data' / 'benchmarking'
PROMPTS_DIR = project_root / 'src' / 'benchmarking' / 'prompts' / '01_dataset_construction'
GT_XLSX_DIR = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'perfect_transcriptions_xlsx'
SAMPLED_PDFS_DIR = BENCHMARKING_ROOT / 'input_data' / 'sampled_pdfs'

MODELS = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-pro']

# --- Main Functions ---

def run_single_benchmark(model_name: str, prompt_name: str, max_workers: int = 20):
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
    perfect_comparison_dir = run_output_dir / 'perfect_transcriptions_xlsx'
    student_comparison_dir = run_output_dir / 'student_transcriptions_xlsx'
    
    run_output_dir.mkdir(parents=True, exist_ok=True)
    llm_csv_output_dir.mkdir(exist_ok=True)
    perfect_comparison_dir.mkdir(exist_ok=True)
    student_comparison_dir.mkdir(exist_ok=True)
    
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
                logging.info(f"Processing PDF: {pdf_path.name}")
                process_pdf(
                    model_name=model_name,
                    prompt_text=prompt_text,
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
    logging.info("--- Starting comparison phase. ---")

    # 2. Run comparisons against both ground truth types
    all_results = {}
    
    # Perfect transcriptions comparison
    perfect_gt_dir = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'perfect_transcriptions_xlsx'
    if perfect_gt_dir.exists():
        logging.info("Running comparison against perfect transcriptions...")
        perfect_results = run_comparison(
            llm_csv_dir=llm_csv_output_dir,
            gt_xlsx_dir=perfect_gt_dir,
            output_dir=perfect_comparison_dir,
            comparison_type="perfect"
        )
        if perfect_results:
            all_results['perfect'] = perfect_results
            logging.info("Perfect transcriptions comparison completed.")
        else:
            logging.warning("No perfect transcriptions comparison results generated.")
    else:
        logging.warning(f"Perfect transcriptions directory not found: {perfect_gt_dir}")
    
    # Student transcriptions comparison
    student_gt_dir = BENCHMARKING_ROOT / 'input_data' / 'transcriptions_xlsx' / 'student_transcriptions_xlsx'
    if student_gt_dir.exists():
        logging.info("Running comparison against student transcriptions...")
        student_results = run_comparison(
            llm_csv_dir=llm_csv_output_dir,
            gt_xlsx_dir=student_gt_dir,
            output_dir=student_comparison_dir,
            comparison_type="student"
        )
        if student_results:
            all_results['student'] = student_results
            logging.info("Student transcriptions comparison completed.")
        else:
            logging.warning("No student transcriptions comparison results generated.")
    else:
        logging.warning(f"Student transcriptions directory not found: {student_gt_dir}")
    
    # 3. Generate combined results.json at prompt level
    if all_results:
        combined_results = {
            'model': model_name,
            'prompt': prompt_stem,
            'timestamp': pd.Timestamp.now().isoformat(),
            'perfect': all_results.get('perfect', {}),
            'student': all_results.get('student', {}),
            'summary': {
                'perfect_cer': all_results.get('perfect', {}).get('character_error_rate', 0),
                'student_cer': all_results.get('student', {}).get('character_error_rate', 0),
                'perfect_match_rate': all_results.get('perfect', {}).get('overall_match_rate', 0),
                'student_match_rate': all_results.get('student', {}).get('overall_match_rate', 0),
                'files_processed': len(set(
                    all_results.get('perfect', {}).get('files_with_results', []) +
                    all_results.get('student', {}).get('files_with_results', [])
                ))
            }
        }
        
        results_path = run_output_dir / "results.json"
        with results_path.open('w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=4)
        logging.info(f"Combined results saved to {results_path}")
    else:
        logging.warning("No comparison results generated. Check if ground truth files exist.")
    
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
        help='The filename of the prompt to use (e.g., "v0.0_prompt.txt").'
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
    
    args = parser.parse_args()

    if args.run_all:
        logging.info("--- Starting full benchmark run for all models and prompts. ---")
        prompt_files = [f.name for f in PROMPTS_DIR.glob('*.txt')]
        
        for model_name in MODELS:
            for prompt_name in prompt_files:
                run_single_benchmark(model_name, prompt_name, args.max_workers)
        
        logging.info("--- All benchmark runs complete. Generating final dashboard. ---")
        create_dashboard(BENCHMARKING_ROOT)

    elif args.model and args.prompt:
        run_single_benchmark(args.model, args.prompt, args.max_workers)
        logging.info("--- Single benchmark run complete. ---")
        logging.info(f"To generate/update the main dashboard, run: python src/benchmarking/scripts/create_dashboard.py")

    else:
        parser.print_help()
        logging.warning("Please specify --model and --prompt, or use --run-all.")

if __name__ == "__main__":
    main() 