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
            # The structure is data/benchmarking/{model}/{prompt}/results.json
            # So, parent is {prompt}, and parent.parent is {model}
            prompt_name = result_file.parent.name
            model_name = result_file.parent.parent.name
            
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            data['model'] = model_name
            data['prompt'] = prompt_name
            all_results.append(data)
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
        'model', 'prompt', 'overall_match_rate', 'character_error_rate',
        'total_gt_matched', 'total_gt_entries', 'total_llm_matched', 'total_llm_entries', 
        'common_files_processed'
    ]
    # Add any columns that might be in the data but not in the list to the end
    df_cols = [col for col in column_order if col in df.columns]
    df = df[df_cols]
    
    # Sort by the most important metrics
    df = df.sort_values(by=['overall_match_rate', 'character_error_rate'], ascending=[False, True])
    
    df = df.rename(columns={
        'overall_match_rate': 'Match Rate (%)',
        'character_error_rate': 'CER (%)',
        'total_gt_matched': 'GT Matched',
        'total_gt_entries': 'GT Total',
        'total_llm_matched': 'LLM Matched',
        'total_llm_entries': 'LLM Total',
        'common_files_processed': 'Files'
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