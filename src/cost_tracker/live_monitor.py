#!/usr/bin/env python3

import time
import os
from cost_calculator import get_cost_summary, print_dashboard

def get_current_log_files():
    """Get current set of log files for comparison."""
    from cost_calculator import find_all_log_files
    
    log_files = find_all_log_files()
    all_files = []
    
    for stage, files in log_files.items():
        for file_path in files:
            all_files.append(file_path)
    
    return set(all_files)

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def run_live_dashboard(update_interval: int = 5):
    """Run the live cost tracking dashboard."""
    print("🚀 Starting live cost monitoring dashboard...")
    print("Press Ctrl+C to stop")
    print()
    
    previous_files = set()
    
    try:
        while True:
            clear_screen()
            
            # Get current summary
            summary = get_cost_summary()
            current_files = get_current_log_files()
            
            # Check for new files
            new_files = current_files - previous_files
            if new_files and previous_files:  # Don't show on first run
                print("🆕 NEW FILES DETECTED:")
                for file_path in new_files:
                    print(f"   📄 {os.path.basename(file_path)}")
                print()
            
            # Print dashboard
            print_dashboard(summary)
            
            # Update file tracking
            previous_files = current_files
            
            # Wait for next update
            print(f"⏰ Next update in {update_interval} seconds...")
            print("Press Ctrl+C to stop")
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Live monitor stopped.")
        print("Final summary:")
        print("=" * 80)
        summary = get_cost_summary()
        print_dashboard(summary)

if __name__ == "__main__":
    run_live_dashboard() 