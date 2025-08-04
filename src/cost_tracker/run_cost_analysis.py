#!/usr/bin/env python3
"""
Cost Analysis Runner for LLM Patent Pipeline
"""

import sys
import argparse
from cost_calculator import get_cost_summary, print_dashboard, export_to_csv
from live_monitor import run_live_dashboard

def main():
    parser = argparse.ArgumentParser(description="LLM Patent Pipeline Cost Analysis")
    parser.add_argument("mode", choices=["dashboard", "live", "csv"], 
                       help="Analysis mode: dashboard (one-time), live (continuous), csv (export)")
    parser.add_argument("--interval", "-i", type=int, default=5,
                       help="Update interval for live mode (seconds, default: 5)")
    parser.add_argument("--output", "-o", default="cost_analysis.csv",
                       help="Output filename for CSV mode (default: cost_analysis.csv)")
    
    args = parser.parse_args()
    
    if args.mode == "dashboard":
        print("📊 Running one-time cost analysis...")
        summary = get_cost_summary()
        print_dashboard(summary)
        
    elif args.mode == "live":
        print("🔄 Starting live cost monitoring...")
        run_live_dashboard(args.interval)
        
    elif args.mode == "csv":
        print("📊 Exporting cost analysis to CSV...")
        summary = get_cost_summary()
        export_to_csv(summary, args.output)

if __name__ == "__main__":
    main() 