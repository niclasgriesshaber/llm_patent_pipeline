#!/usr/bin/env python3

import json
import glob
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Pricing configuration (in USD)
PRICING = {
    "gemini-2.5-pro": {
        "input_tokens": 1.25,  # USD per 1M tokens
        "output_tokens": 10.00  # USD per 1M tokens (thought + candidate)
    },
    "gemini-2.5-flash": {
        "input_tokens": 0.1,   # USD per 1M tokens
        "output_tokens": 0.4   # USD per 1M tokens (thought + candidate)
    },
    "gemini-2.5-flash-lite": {
        "input_tokens": 0.1,   # USD per 1M tokens
        "output_tokens": 0.4   # USD per 1M tokens (thought + candidate)
    },
    "gemini-2.0-flash": {
        "input_tokens": 0.1,   # USD per 1M tokens
        "output_tokens": 0.4   # USD per 1M tokens (thought + candidate)
    }
}

# Currency conversion rate (USD to EUR)
USD_TO_EUR_RATE = 0.87  # 1 EUR = 1.15 USD, so USD to EUR = 1/1.15 = 0.87

def find_all_log_files() -> Dict[str, List[str]]:
    """Find all log files across the three pipeline stages."""
    # Try different possible base paths
    possible_paths = [
        Path("../../data"),  # From cost_tracker directory
        Path("data"),        # From main project directory
        Path("../data")      # Alternative relative path
    ]
    
    base_path = None
    for path in possible_paths:
        if path.exists():
            base_path = path
            break
    
    if not base_path:
        print("Warning: Could not find data directory. Using relative path.")
        base_path = Path("../../data")
    
    log_files = {
        "dataset_construction": [],
        "dataset_cleaning": [],
        "variable_extraction": []
    }
    
    # Dataset construction logs (in csvs subdirectories)
    construction_pattern = base_path / "01_dataset_construction/csvs/*/*_log.json"
    log_files["dataset_construction"] = glob.glob(str(construction_pattern))
    
    # Dataset cleaning logs
    cleaning_pattern = base_path / "02_dataset_cleaning/check_merge_xlsx/logs/*_cleaned_log.json"
    log_files["dataset_cleaning"] = glob.glob(str(cleaning_pattern))
    
    # Variable extraction logs
    extraction_pattern = base_path / "03_variable_extraction/cleaned_with_variables_csvs/logs/*_variable_extraction_logs.json"
    log_files["variable_extraction"] = glob.glob(str(extraction_pattern))
    
    return log_files

def load_log_file(file_path: str) -> Dict[str, Any]:
    """Load JSON data from a log file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def calculate_file_cost(log_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate cost for a single file based on token counts."""
    if not log_data:
        return {"cost": 0, "input_cost": 0, "output_cost": 0}
    
    model = log_data.get("model", "unknown")
    pricing = PRICING.get(model, {"input_tokens": 0, "output_tokens": 0})
    
    # Get token counts
    input_tokens = log_data.get("total_input_tokens", 0)
    thought_tokens = log_data.get("total_thought_tokens", 0)
    candidate_tokens = log_data.get("total_candidate_tokens", 0)
    
    # Combine thought and candidate tokens into output tokens
    output_tokens = thought_tokens + candidate_tokens
    
    # Calculate costs in USD (convert to millions)
    input_cost_usd = (input_tokens / 1_000_000) * pricing["input_tokens"]
    output_cost_usd = (output_tokens / 1_000_000) * pricing["output_tokens"]
    total_cost_usd = input_cost_usd + output_cost_usd
    
    # Convert to EUR
    input_cost_eur = input_cost_usd * USD_TO_EUR_RATE
    output_cost_eur = output_cost_usd * USD_TO_EUR_RATE
    total_cost_eur = total_cost_usd * USD_TO_EUR_RATE
    
    return {
        "cost": total_cost_eur,
        "input_cost": input_cost_eur,
        "output_cost": output_cost_eur,
        "cost_usd": total_cost_usd,
        "input_cost_usd": input_cost_usd,
        "output_cost_usd": output_cost_usd,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model
    }

def get_cost_summary() -> Dict[str, Any]:
    """Get comprehensive cost summary across all pipeline stages."""
    log_files = find_all_log_files()
    
    summary = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {
            "dataset_construction": {"files": 0, "cost": 0, "pages": 0, "input_tokens": 0, "output_tokens": 0, "file_details": []},
            "dataset_cleaning": {"files": 0, "cost": 0, "rows": 0, "input_tokens": 0, "output_tokens": 0, "file_details": []},
            "variable_extraction": {"files": 0, "cost": 0, "rows": 0, "input_tokens": 0, "output_tokens": 0, "file_details": []}
        },
        "total_cost": 0,
        "total_files": 0,
        "total_pages": 0,
        "total_rows": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0
    }
    
    # Process each stage
    for stage, files in log_files.items():
        for file_path in files:
            log_data = load_log_file(file_path)
            if not log_data:
                continue
                
            cost_info = calculate_file_cost(log_data)
            
            # Create file detail
            file_detail = {
                "file_name": log_data.get("file_name", Path(file_path).stem),
                "cost": cost_info["cost"],
                "input_cost": cost_info["input_cost"],
                "output_cost": cost_info["output_cost"],
                "input_tokens": cost_info["input_tokens"],
                "output_tokens": cost_info["output_tokens"],
                "model": cost_info["model"],
                "processing_time": log_data.get("processing_time_seconds", 0),
                "max_workers": log_data.get("max_workers", 0)
            }
            
            # Add stage-specific metrics
            if stage == "dataset_construction":
                file_detail["pages"] = log_data.get("number_of_pages", 0)
                file_detail["rows"] = 0
                summary["stages"][stage]["pages"] += file_detail["pages"]
            else:
                file_detail["pages"] = 0
                file_detail["rows"] = log_data.get("number_of_rows", 0)
                summary["stages"][stage]["rows"] += file_detail["rows"]
            
            # Update stage totals
            summary["stages"][stage]["files"] += 1
            summary["stages"][stage]["cost"] += cost_info["cost"]
            summary["stages"][stage]["input_tokens"] += cost_info["input_tokens"]
            summary["stages"][stage]["output_tokens"] += cost_info["output_tokens"]
            summary["stages"][stage]["file_details"].append(file_detail)
            
            # Update global totals
            summary["total_cost"] += cost_info["cost"]
            summary["total_files"] += 1
            summary["total_pages"] += file_detail["pages"]
            summary["total_rows"] += file_detail["rows"]
            summary["total_input_tokens"] += cost_info["input_tokens"]
            summary["total_output_tokens"] += cost_info["output_tokens"]
    
    return summary

def print_dashboard(summary: Dict[str, Any]):
    """Print the cost tracking dashboard."""
    print("=" * 80)
    print("🚀 LLM PATENT PIPELINE - COST TRACKING DASHBOARD")
    print("=" * 80)
    print(f"📊 Last Updated: {summary['last_updated']}")
    print()
    
    # Overall summary
    print("💰 OVERALL SUMMARY")
    print("-" * 40)
    print(f"Total Cost:          €{summary['total_cost']:>8.2f}")
    print(f"Total Files:         {summary['total_files']:>8d}")
    print(f"Total Pages:         {summary['total_pages']:>8d}")
    print(f"Total Rows:          {summary['total_rows']:>8d}")
    print(f"Total Input Tokens:  {summary['total_input_tokens']:>8,d}")
    print(f"Total Output Tokens: {summary['total_output_tokens']:>8,d}")
    print()
    
    # Stage breakdown
    print("📋 STAGE BREAKDOWN")
    print("-" * 40)
    
    stages = summary["stages"]
    
    # Dataset Construction
    const = stages["dataset_construction"]
    print(f"🔧 Dataset Construction:")
    print(f"   Files: {const['files']:>3d} | Cost: €{const['cost']:>8.2f} | Pages: {const['pages']:>5d} | "
          f"Input: {const['input_tokens']:>8,d} | Output: {const['output_tokens']:>8,d}")
    
    # Dataset Cleaning
    clean = stages["dataset_cleaning"]
    print(f"🧹 Dataset Cleaning:")
    print(f"   Files: {clean['files']:>3d} | Cost: €{clean['cost']:>8.2f} | Rows: {clean['rows']:>5d} | "
          f"Input: {clean['input_tokens']:>8,d} | Output: {clean['output_tokens']:>8,d}")
    
    # Variable Extraction
    extract = stages["variable_extraction"]
    print(f"📊 Variable Extraction:")
    print(f"   Files: {extract['files']:>3d} | Cost: €{extract['cost']:>8.2f} | Rows: {extract['rows']:>5d} | "
          f"Input: {extract['input_tokens']:>8,d} | Output: {extract['output_tokens']:>8,d}")
    print()
    
    # Detailed breakdown by stage
    for stage_name, stage_data in stages.items():
        if stage_data["files"] > 0:
            print(f"📁 {stage_name.replace('_', ' ').title()} - DETAILED BREAKDOWN")
            print("-" * 40)
            
            # Sort by cost (highest first)
            sorted_files = sorted(stage_data["file_details"], key=lambda x: x["cost"], reverse=True)
            
            for i, file_info in enumerate(sorted_files, 1):
                if stage_name == "dataset_construction":
                    print(f"{i:2d}. {file_info['file_name']:<25} €{file_info['cost']:>8.2f} "
                          f"({file_info['pages']:>3d} pages)")
                else:
                    print(f"{i:2d}. {file_info['file_name']:<25} €{file_info['cost']:>8.2f} "
                          f"({file_info['rows']:>3d} rows)")
            print()

def export_to_csv(summary: Dict[str, Any], filename: str = "cost_analysis.csv"):
    """Export cost analysis to CSV."""
    import pandas as pd
    
    all_files = []
    for stage_name, stage_data in summary["stages"].items():
        for file_info in stage_data["file_details"]:
            file_info["stage"] = stage_name
            all_files.append(file_info)
    
    if all_files:
        df = pd.DataFrame(all_files)
        df = df.sort_values("cost", ascending=False)
        df.to_csv(filename, index=False)
        print(f"📊 Cost analysis exported to {filename}")
    else:
        print("⚠️  No files found to export")

if __name__ == "__main__":
    summary = get_cost_summary()
    print_dashboard(summary) 