import json
import pandas as pd
from pathlib import Path
import argparse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_results_data(benchmark_data_dir: Path):
    """
    Load all results.json files and organize them by benchmark type.
    """
    logging.info(f"Scanning for result files in {benchmark_data_dir}...")
    
    result_files = list(benchmark_data_dir.rglob("results.json"))
    
    if not result_files:
        logging.warning("No 'results.json' files found. Cannot generate dashboard.")
        return {}

    results_by_type = {
        '01_dataset_construction': [],
        '02_dataset_cleaning': [],
        '03_variable_extraction': []
    }
    
    for result_file in result_files:
        try:
            # Extract benchmark type, model, and prompt from path
            # Structure: data/benchmarking/results/{benchmark_type}/{model}/{prompt}/results.json
            path_parts = result_file.parts
            if 'results' in path_parts:
                results_idx = path_parts.index('results')
                if results_idx + 3 < len(path_parts):
                    benchmark_type = path_parts[results_idx + 1]
                    model_name = path_parts[results_idx + 2]
                    prompt_name = path_parts[results_idx + 3]
                    
                    if benchmark_type in results_by_type:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        
                        # Add metadata
                        data['model'] = model_name
                        data['prompt'] = prompt_name
                        data['benchmark_type'] = benchmark_type
                        data['file_path'] = str(result_file)
                        
                        results_by_type[benchmark_type].append(data)
                        
        except (IndexError, FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Could not process file {result_file}: {e}")
            continue
    
    return results_by_type

def create_dataset_construction_overview(construction_results):
    """
    Create overview for 01_dataset_construction with CER metrics.
    """
    if not construction_results:
        return None
    
    overview_data = []
    
    for result in construction_results:
        model = result.get('model', 'Unknown')
        prompt = result.get('prompt', 'Unknown')
        
        # Extract CER data from perfect and student comparisons
        perfect_data = result.get('perfect', {})
        student_data = result.get('student', {})
        
        # Get additional metrics
        total_gt_entries = perfect_data.get('total_gt_entries', 0) + student_data.get('total_gt_entries', 0)
        total_llm_entries = perfect_data.get('total_llm_entries', 0) + student_data.get('total_llm_entries', 0)
        total_matches = perfect_data.get('total_gt_matched', 0) + student_data.get('total_gt_matched', 0)
        
        overview_data.append({
            'Model': model,
            'Prompt': prompt,
            'Perfect CER (%)': round(perfect_data.get('character_error_rate', 0), 2),
            'Student CER (%)': round(student_data.get('character_error_rate', 0), 2),
            'Perfect Match Rate (%)': round(perfect_data.get('overall_match_rate', 0), 2),
            'Student Match Rate (%)': round(student_data.get('overall_match_rate', 0), 2),
            'Total GT Entries': total_gt_entries,
            'Total LLM Entries': total_llm_entries,
            'Total Matches': total_matches,
            'Files Processed': perfect_data.get('common_files_processed', 0) + student_data.get('common_files_processed', 0)
        })
    
    if not overview_data:
        return None
    
    df = pd.DataFrame(overview_data)
    df = df.sort_values(by=['Perfect CER (%)'], ascending=[True])
    
    return df

def create_match_rate_overview(construction_results, cleaning_results):
    """
    Create match rate overview for both construction and cleaning phases.
    """
    overview_data = []
    
    # Process construction results
    for result in construction_results:
        model = result.get('model', 'Unknown')
        prompt = result.get('prompt', 'Unknown')
        
        perfect_data = result.get('perfect', {})
        student_data = result.get('student', {})
        
        # Construction phase
        overview_data.append({
            'Phase': '01_Construction',
            'Model': model,
            'Prompt': prompt,
            'GT Perspective': 'Perfect',
            'Match Rate (%)': round(perfect_data.get('overall_match_rate', 0), 2),
            'Total Matches': perfect_data.get('total_gt_matched', 0),
            'Total GT Fields': perfect_data.get('total_gt_entries', 0),
            'Total LLM Fields': perfect_data.get('total_llm_entries', 0),
            'LLM Match Rate (%)': round((perfect_data.get('total_llm_matched', 0) / perfect_data.get('total_llm_entries', 1)) * 100, 2) if perfect_data.get('total_llm_entries', 0) > 0 else 0,
            'CER (%)': round(perfect_data.get('character_error_rate', 0), 2)
        })
        
        overview_data.append({
            'Phase': '01_Construction',
            'Model': model,
            'Prompt': prompt,
            'GT Perspective': 'Student',
            'Match Rate (%)': round(student_data.get('overall_match_rate', 0), 2),
            'Total Matches': student_data.get('total_gt_matched', 0),
            'Total GT Fields': student_data.get('total_gt_entries', 0),
            'Total LLM Fields': student_data.get('total_llm_entries', 0),
            'LLM Match Rate (%)': round((student_data.get('total_llm_matched', 0) / student_data.get('total_llm_entries', 1)) * 100, 2) if student_data.get('total_llm_entries', 0) > 0 else 0,
            'CER (%)': round(student_data.get('character_error_rate', 0), 2)
        })
    
    # Process cleaning results
    for result in cleaning_results:
        model = result.get('model', 'Unknown')
        prompt = result.get('prompt', 'Unknown')
        
        perfect_data = result.get('perfect', {})
        student_data = result.get('student', {})
        
        # Cleaning phase
        overview_data.append({
            'Phase': '02_Cleaning',
            'Model': model,
            'Prompt': prompt,
            'GT Perspective': 'Perfect',
            'Match Rate (%)': round(perfect_data.get('overall_match_rate', 0), 2),
            'Total Matches': perfect_data.get('total_gt_matched', 0),
            'Total GT Fields': perfect_data.get('total_gt_entries', 0),
            'Total LLM Fields': perfect_data.get('total_llm_entries', 0),
            'LLM Match Rate (%)': round((perfect_data.get('total_llm_matched', 0) / perfect_data.get('total_llm_entries', 1)) * 100, 2) if perfect_data.get('total_llm_entries', 0) > 0 else 0,
            'CER (%)': round(perfect_data.get('character_error_rate', 0), 2)
        })
        
        overview_data.append({
            'Phase': '02_Cleaning',
            'Model': model,
            'Prompt': prompt,
            'GT Perspective': 'Student',
            'Match Rate (%)': round(student_data.get('overall_match_rate', 0), 2),
            'Total Matches': student_data.get('total_gt_matched', 0),
            'Total GT Fields': student_data.get('total_gt_entries', 0),
            'Total LLM Fields': student_data.get('total_llm_entries', 0),
            'LLM Match Rate (%)': round((student_data.get('total_llm_matched', 0) / student_data.get('total_llm_entries', 1)) * 100, 2) if student_data.get('total_llm_entries', 0) > 0 else 0,
            'CER (%)': round(student_data.get('character_error_rate', 0), 2)
        })
    
    if not overview_data:
        return None
    
    df = pd.DataFrame(overview_data)
    df = df.sort_values(by=['Phase', 'Model', 'Prompt', 'GT Perspective'])
    
    return df

def create_variable_extraction_overview(extraction_results):
    """
    Create overview for 03_variable_extraction.
    """
    if not extraction_results:
        return None
    
    overview_data = []
    
    for result in extraction_results:
        model = result.get('model', 'Unknown')
        prompt = result.get('prompt', 'Unknown')
        
        perfect_data = result.get('perfect', {})
        student_data = result.get('student', {})
        
        # Get variable-specific rates
        perfect_variable_rates = perfect_data.get('variable_match_rates', {})
        student_variable_rates = student_data.get('variable_match_rates', {})
        
        overview_data.append({
            'Model': model,
            'Prompt': prompt,
            'GT Perspective': 'Perfect',
            'Overall Match Rate (%)': round(perfect_data.get('overall_match_rate', 0), 2),
            'Total Cells': perfect_data.get('total_cells', 0),
            'Matched Cells': perfect_data.get('matched_cells', 0),
            'Patent ID (%)': round(perfect_variable_rates.get('patent_id', 0), 2),
            'Name (%)': round(perfect_variable_rates.get('name', 0), 2),
            'Address (%)': round(perfect_variable_rates.get('address', 0), 2),
            'Description (%)': round(perfect_variable_rates.get('description', 0), 2),
            'Date (%)': round(perfect_variable_rates.get('date', 0), 2),
            'Files Processed': perfect_data.get('files_processed', 0)
        })
        
        overview_data.append({
            'Model': model,
            'Prompt': prompt,
            'GT Perspective': 'Student',
            'Overall Match Rate (%)': round(student_data.get('overall_match_rate', 0), 2),
            'Total Cells': student_data.get('total_cells', 0),
            'Matched Cells': student_data.get('matched_cells', 0),
            'Patent ID (%)': round(student_variable_rates.get('patent_id', 0), 2),
            'Name (%)': round(student_variable_rates.get('name', 0), 2),
            'Address (%)': round(student_variable_rates.get('address', 0), 2),
            'Description (%)': round(student_variable_rates.get('description', 0), 2),
            'Date (%)': round(student_variable_rates.get('date', 0), 2),
            'Files Processed': student_data.get('files_processed', 0)
        })
    
    if not overview_data:
        return None
    
    df = pd.DataFrame(overview_data)
    df = df.sort_values(by=['Model', 'Prompt', 'GT Perspective'])
    
    return df

def create_html_table(df, title, css_class='styled-table'):
    """Create HTML table with styling."""
    if df is None or df.empty:
        return f"""
        <div class="section">
            <h2>{title}</h2>
            <div class="no-data">
                <p>No data available for this section.</p>
            </div>
        </div>
        """
    
    html_table = df.to_html(index=False, classes=css_class, border=0)
    return f"""
        <div class="section">
            <h2>{title}</h2>
            <div class="table-container">
                {html_table}
            </div>
        </div>
    """

def create_summary_stats(results_by_type):
    """Create summary statistics section."""
    construction_count = len(results_by_type['01_dataset_construction'])
    cleaning_count = len(results_by_type['02_dataset_cleaning'])
    extraction_count = len(results_by_type['03_variable_extraction'])
    
    total_results = construction_count + cleaning_count + extraction_count
    
    # Get unique models and prompts
    all_models = set()
    all_prompts = set()
    
    for benchmark_type, results in results_by_type.items():
        for result in results:
            all_models.add(result.get('model', 'Unknown'))
            all_prompts.add(result.get('prompt', 'Unknown'))
    
    return f"""
        <div class="summary-stats">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{total_results}</div>
                    <div class="stat-label">Total Results</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(all_models)}</div>
                    <div class="stat-label">Models Tested</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(all_prompts)}</div>
                    <div class="stat-label">Prompts Evaluated</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{construction_count}</div>
                    <div class="stat-label">Construction Results</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{cleaning_count}</div>
                    <div class="stat-label">Cleaning Results</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{extraction_count}</div>
                    <div class="stat-label">Extraction Results</div>
                </div>
            </div>
        </div>
    """

def create_dashboard(benchmark_data_dir: Path):
    """
    Create a world-class comprehensive dashboard with three main overviews.
    """
    # Load all results data
    results_by_type = load_results_data(benchmark_data_dir)
    
    if not any(results_by_type.values()):
        logging.error("No results data found. Cannot generate dashboard.")
        return
    
    # Create the three overviews
    construction_df = create_dataset_construction_overview(results_by_type['01_dataset_construction'])
    match_rate_df = create_match_rate_overview(
        results_by_type['01_dataset_construction'], 
        results_by_type['02_dataset_cleaning']
    )
    extraction_df = create_variable_extraction_overview(results_by_type['03_variable_extraction'])
    
    # Generate HTML content
    construction_html = create_html_table(
        construction_df, 
        "1. Dataset Construction - Character Error Rate (CER) Overview"
    )
    
    match_rate_html = create_html_table(
        match_rate_df, 
        "2. Match Rate Overview - Construction and Cleaning Phases"
    )
    
    extraction_html = create_html_table(
        extraction_df, 
        "3. Variable Extraction Overview"
    )
    
    summary_stats = create_summary_stats(results_by_type)
    
    # Create full HTML page with world-class styling
    dashboard_path = benchmark_data_dir / "dashboard.html"
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>World-Class Benchmarking Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #2d3748;
                line-height: 1.6;
                min-height: 100vh;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 40px;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            }}
            
            .header h1 {{
                font-size: 3rem;
                font-weight: 700;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
            }}
            
            .header p {{
                font-size: 1.2rem;
                color: #718096;
                font-weight: 400;
            }}
            
            .summary-stats {{
                margin-bottom: 40px;
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .stat-card {{
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }}
            
            .stat-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
            }}
            
            .stat-number {{
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
            }}
            
            .stat-label {{
                font-size: 1rem;
                color: #718096;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .section {{
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 40px;
                margin-bottom: 30px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            }}
            
            .section h2 {{
                font-size: 2rem;
                font-weight: 600;
                color: #2d3748;
                margin-bottom: 30px;
                padding-bottom: 15px;
                border-bottom: 3px solid #e2e8f0;
                position: relative;
            }}
            
            .section h2::after {{
                content: '';
                position: absolute;
                bottom: -3px;
                left: 0;
                width: 60px;
                height: 3px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                border-radius: 2px;
            }}
            
            .table-container {{
                overflow-x: auto;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            }}
            
            .styled-table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 15px;
                overflow: hidden;
                font-size: 0.9rem;
            }}
            
            .styled-table thead {{
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
            }}
            
            .styled-table th {{
                padding: 20px 15px;
                text-align: left;
                font-weight: 600;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border: none;
            }}
            
            .styled-table td {{
                padding: 15px;
                border-bottom: 1px solid #e2e8f0;
                border-right: 1px solid #e2e8f0;
            }}
            
            .styled-table tbody tr:nth-child(even) {{
                background-color: #f7fafc;
            }}
            
            .styled-table tbody tr:hover {{
                background-color: #edf2f7;
                transform: scale(1.01);
                transition: all 0.2s ease;
            }}
            
            .styled-table tbody tr:last-child td {{
                border-bottom: none;
            }}
            
            .no-data {{
                text-align: center;
                padding: 60px 20px;
                color: #718096;
                font-size: 1.1rem;
            }}
            
            .notes {{
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 40px;
                margin-top: 30px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            }}
            
            .notes h3 {{
                font-size: 1.5rem;
                font-weight: 600;
                color: #2d3748;
                margin-bottom: 20px;
            }}
            
            .notes ul {{
                list-style: none;
                padding: 0;
            }}
            
            .notes li {{
                padding: 10px 0;
                border-bottom: 1px solid #e2e8f0;
                display: flex;
                align-items: center;
            }}
            
            .notes li:last-child {{
                border-bottom: none;
            }}
            
            .notes li::before {{
                content: '•';
                color: #667eea;
                font-weight: bold;
                margin-right: 15px;
                font-size: 1.5rem;
            }}
            
            .metric-highlight {{
                font-weight: 600;
                color: #667eea;
            }}
            
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                color: rgba(255, 255, 255, 0.8);
                font-size: 0.9rem;
            }}
            
            @media (max-width: 768px) {{
                .container {{
                    padding: 10px;
                }}
                
                .header h1 {{
                    font-size: 2rem;
                }}
                
                .section {{
                    padding: 20px;
                }}
                
                .stats-grid {{
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                }}
                
                .stat-card {{
                    padding: 20px;
                }}
                
                .stat-number {{
                    font-size: 2rem;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>World-Class Benchmarking Dashboard</h1>
                <p>Comprehensive Analysis of LLM Patent Processing Pipeline</p>
                <p style="margin-top: 10px; font-size: 0.9rem; color: #a0aec0;">
                    Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                </p>
            </div>
            
            {summary_stats}
            
            {construction_html}
            
            {match_rate_html}
            
            {extraction_html}
            
            <div class="notes">
                <h3>📊 Understanding the Metrics</h3>
                <ul>
                    <li><span class="metric-highlight">CER (Character Error Rate):</span> Lower values indicate better performance. Measures character-level accuracy between ground truth and LLM output.</li>
                    <li><span class="metric-highlight">Match Rate:</span> Percentage of entries successfully matched between ground truth and LLM output using fuzzy matching.</li>
                    <li><span class="metric-highlight">GT Perspective:</span> Match rate calculated from ground truth perspective (how many GT entries were matched).</li>
                    <li><span class="metric-highlight">LLM Perspective:</span> Match rate calculated from LLM output perspective (how many LLM entries were matched).</li>
                    <li><span class="metric-highlight">Perfect vs Student:</span> Comparison against perfect transcriptions vs. student transcriptions (different quality levels).</li>
                    <li><span class="metric-highlight">Variable Extraction:</span> Cell-level accuracy for specific patent fields (patent_id, name, address, description, date).</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>🔬 Imperial Germany Patent Dataset • LLM Processing Pipeline</p>
        </div>
    </body>
    </html>
    """
    
    dashboard_path.write_text(html_content, encoding='utf-8')
    logging.info(f"🎉 World-class dashboard successfully generated at: {dashboard_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate a world-class comprehensive benchmarking dashboard from results.")
    # Assuming this script is in src/benchmarking/scripts, the project root is 3 levels up
    project_root = Path(__file__).resolve().parents[3]
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