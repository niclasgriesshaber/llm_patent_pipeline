import json
import pandas as pd
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dashboard(benchmark_data_dir: Path):
    """
    Scans for all 'results.json' files in the benchmark data directory,
    aggregates them, and creates a single HTML dashboard.
    """
    logging.info(f"Scanning for result files in {benchmark_data_dir}...")
    
    result_files = list(benchmark_data_dir.rglob("results.json"))
    
    if not result_files:
        logging.warning("No 'results.json' files found. Cannot generate dashboard.")
        return

    all_results = []
    for result_file in result_files:
        try:
            # The structure is data/benchmarking/results/01_dataset_construction/{model}/{prompt}/results.json
            # So, parent is {prompt}, parent.parent is {model}, parent.parent.parent is {benchmark_type}
            prompt_name = result_file.parent.name
            model_name = result_file.parent.parent.name
            
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Handle new combined results structure
            if 'summary' in data:
                # New structure with perfect/student comparisons
                summary = data['summary']
                result_entry = {
                    'model': model_name,
                    'prompt': prompt_name,
                    'perfect_cer_normalized': summary.get('perfect_cer_normalized', 0),
                    'perfect_cer_unnormalized': summary.get('perfect_cer_unnormalized', 0),
                    'student_cer_normalized': summary.get('student_cer_normalized', 0),
                    'student_cer_unnormalized': summary.get('student_cer_unnormalized', 0),
                    'perfect_match_rate': summary.get('perfect_match_rate', 0),
                    'student_match_rate': summary.get('student_match_rate', 0),
                    'files_processed': summary.get('files_processed', 0)
                }
            else:
                # Legacy structure - convert to new format
                result_entry = {
                    'model': model_name,
                    'prompt': prompt_name,
                    'perfect_cer_normalized': data.get('character_error_rate', 0),
                    'perfect_cer_unnormalized': data.get('character_error_rate', 0),
                    'student_cer_normalized': 0,
                    'student_cer_unnormalized': 0,
                    'perfect_match_rate': data.get('overall_match_rate', 0),
                    'student_match_rate': 0,
                    'files_processed': data.get('common_files_processed', 0)
                }
            
            all_results.append(result_entry)
        except (IndexError, FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Could not process file {result_file}: {e}")
            continue
            
    if not all_results:
        logging.error("Failed to load data from any result files. Aborting.")
        return

    # Create and style DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    column_order = [
        'model', 'prompt', 'perfect_cer_normalized', 'student_cer_normalized',
        'perfect_match_rate', 'student_match_rate', 'files_processed'
    ]
    # Add any columns that might be in the data but not in the list to the end
    df_cols = [col for col in column_order if col in df.columns]
    df = df[df_cols]
    
    # Sort by the most important metrics (normalized CER for perfect transcriptions)
    df = df.sort_values(by=['perfect_cer_normalized'], ascending=[True])
    
    df = df.rename(columns={
        'perfect_cer_normalized': 'Perfect CER (Norm) (%)',
        'perfect_cer_unnormalized': 'Perfect CER (Unnorm) (%)',
        'student_cer_normalized': 'Student CER (Norm) (%)',
        'student_cer_unnormalized': 'Student CER (Unnorm) (%)',
        'perfect_match_rate': 'Perfect Match Rate (%)',
        'student_match_rate': 'Student Match Rate (%)',
        'files_processed': 'Files'
    })

    html_table = df.to_html(index=False, classes='styled-table', border=0)
    
    # Create full HTML page with styles
    dashboard_path = benchmark_data_dir / "dashboard.html"
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Benchmarking Dashboard</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f8f9fa; color: #212529; padding: 20px; }}
            h1 {{ color: #343a40; text-align: center; }}
            .styled-table {{
                width: 90%;
                margin: 20px auto;
                border-collapse: collapse;
                box-shadow: 0 2px 15px rgba(0,0,0,0.1);
                background-color: white;
            }}
            .styled-table thead tr {{
                background-color: #007bff;
                color: #ffffff;
                text-align: left;
            }}
            .styled-table th, .styled-table td {{
                padding: 12px 15px;
                border-bottom: 1px solid #dee2e6;
            }}
            .styled-table tbody tr:nth-of-type(even) {{
                background-color: #f2f2f2;
            }}
            .styled-table tbody tr:hover {{
                background-color: #e2e6ea;
            }}
            .styled-table tbody tr:last-of-type {{
                border-bottom: 2px solid #007bff;
            }}
        </style>
    </head>
    <body>
        <h1>Benchmarking Dashboard</h1>
        {html_table}
    </body>
    </html>
    """
    
    dashboard_path.write_text(html_content, encoding='utf-8')
    logging.info(f"Dashboard successfully generated at: {dashboard_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a benchmarking dashboard from results.")
    # Assuming this script is in src/benchmarking, the project root is 3 levels up
    project_root = Path(__file__).resolve().parents[2]
    default_data_dir = project_root / 'data' / 'benchmarking'
    
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=default_data_dir,
        help=f"The root directory containing the benchmarking data. Defaults to {default_data_dir}"
    )
    args = parser.parse_args()
    
    create_dashboard(args.data_dir)

if __name__ == "__main__":
    main() 