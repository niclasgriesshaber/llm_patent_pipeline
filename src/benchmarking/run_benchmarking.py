import argparse
import logging
from pathlib import Path
import sys

# Ensure the 'core' module can be found
# The script is in src/benchmarking, so we add src to the path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root / 'src'))

from benchmarking.core.llm_processing import process_pdf
from benchmarking.core.benchmarking import run_comparison
from benchmarking.create_dashboard import create_dashboard

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BENCHMARKING_ROOT = project_root / 'data' / 'benchmarking'
PROMPTS_DIR = project_root / 'src' / 'benchmarking' / 'prompts'
GT_XLSX_DIR = BENCHMARKING_ROOT / 'gt_xlsx'
SAMPLED_PDFS_DIR = BENCHMARKING_ROOT / 'sampled_pdfs'

MODELS = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-pro']

# --- Main Functions ---

def run_single_benchmark(model_name: str, prompt_name: str):
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
    run_output_dir = BENCHMARKING_ROOT / model_name / prompt_stem
    llm_csv_output_dir = run_output_dir / 'llm_csv'
    run_output_dir.mkdir(parents=True, exist_ok=True)
    llm_csv_output_dir.mkdir(exist_ok=True)
    
    logging.info(f"Output will be saved in: {run_output_dir}")

    # 1. Process all PDFs to generate LLM CSVs
    pdf_files = list(SAMPLED_PDFS_DIR.glob('*.pdf'))
    if not pdf_files:
        logging.error(f"No PDFs found in {SAMPLED_PDFS_DIR}. Cannot proceed.")
        return

    logging.info(f"Found {len(pdf_files)} PDFs to process.")
    for pdf_path in pdf_files:
        process_pdf(
            model_name=model_name,
            prompt_text=prompt_text,
            pdf_path=pdf_path,
            output_dir=llm_csv_output_dir
        )
    
    logging.info("--- LLM processing complete. Starting comparison. ---")

    # 2. Run comparison between generated CSVs and ground truth
    run_comparison(
        llm_csv_dir=llm_csv_output_dir,
        gt_xlsx_dir=GT_XLSX_DIR,
        output_dir=run_output_dir
    )
    
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
    
    args = parser.parse_args()

    if args.run_all:
        logging.info("--- Starting full benchmark run for all models and prompts. ---")
        prompt_files = [f.name for f in PROMPTS_DIR.glob('*.txt')]
        
        for model_name in MODELS:
            for prompt_name in prompt_files:
                run_single_benchmark(model_name, prompt_name)
        
        logging.info("--- All benchmark runs complete. Generating final dashboard. ---")
        create_dashboard(BENCHMARKING_ROOT)

    elif args.model and args.prompt:
        run_single_benchmark(args.model, args.prompt)
        logging.info("--- Single benchmark run complete. ---")
        logging.info(f"To generate/update the main dashboard, run: python src/benchmarking/create_dashboard.py")

    else:
        parser.print_help()
        logging.warning("Please specify --model and --prompt, or use --run-all.")

if __name__ == "__main__":
    main() 