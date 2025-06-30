import pandas as pd
import os
from pathlib import Path
import glob

print("Starting the dataset merging script...")

# Path Configuration

script_location = Path(__file__).resolve().parent
project_root = script_location.parent.parent
input_dir = project_root / "data" / "06_csvs_with_variables_harmonized"
output_dir = project_root / "data" / "07_merged_csv"
output_csv_file = output_dir / "imperial-patent-office.csv"
output_excel_file = output_dir / "imperial-patent-office.xlsx"

print(f"Script location: {script_location}")
print(f"Project root determined as: {project_root}")
print(f"Input CSVs directory: {input_dir}")
print(f"Output directory: {output_dir}")

# Find CSV files

file_pattern = "Patentamt_*.csv"
csv_files = sorted(list(input_dir.glob(file_pattern)))

if not csv_files:
    print(f"Error: No CSV files found matching '{file_pattern}' in {input_dir}")
    print("Please check the input directory and file naming convention.")
    exit()

print(f"Found {len(csv_files)} CSV files to merge:")
for f in csv_files:
    print(f"  - {f.name}")

# Merge CSV files

all_dataframes = []

for file_path in csv_files:
    try:
        # Extract the year from the filename
        filename_stem = file_path.stem
        year_str = filename_stem.split('_', 1)[1] # e.g., "1877_1888" or "1889"
        print(f"  Processing {file_path.name}... extracted year: '{year_str}'")

        # Read the current CSV file
        df = pd.read_csv(file_path)

        # Add the 'year' column at the beginning
        df.insert(0, 'year', year_str)

        # Append the dataframe to the list
        all_dataframes.append(df)

    except Exception as e:
        print(f"  Error processing file {file_path.name}: {e}")
        print("  Skipping this file.")

# Concatenate all dataframes in the list all_dataframes

if not all_dataframes:
    print("Error: No dataframes were successfully processed. Cannot merge.")
    exit()

print("\nConcatenating all dataframes...")
merged_df = pd.concat(all_dataframes, ignore_index=True)

# Column order
expected_first_columns = ["year", "id", "page", "entry", "category"]

# Check if all expected columns (minus 'year' initially) are present
original_cols = list(all_dataframes[0].columns) # Get cols from first df after adding year
missing_original = [col for col in expected_first_columns if col not in original_cols]
if missing_original:
     print(f"Warning: Original CSVs seem to be missing expected columns: {missing_original}")

# Reorder the columns: first the expected ones, then any others in their original order
all_columns = list(merged_df.columns)
additional_columns = [col for col in all_columns if col not in expected_first_columns]
final_column_order = expected_first_columns + additional_columns
try:
    merged_df = merged_df[final_column_order]
    print("Columns reordered successfully.")
except KeyError as e:
     print(f"Error: Could not find expected column {e} in the merged data. Please check input CSVs.")
     print("Cannot save output files.")
     exit()

print(f"Total rows in merged dataframe: {len(merged_df)}")
print(f"Final columns: {list(merged_df.columns)}")

# Save output files

# Create the output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\nEnsured output directory exists: {output_dir}")

# Save to CSV
try:
    print(f"Saving merged data to CSV: {output_csv_file}")
    merged_df.to_csv(output_csv_file, index=False, encoding='utf-8')
    print("  CSV file saved successfully.")
except Exception as e:
    print(f"  Error saving CSV file: {e}")

# Save to Excel
try:
    print(f"Saving merged data to Excel: {output_excel_file}")
    merged_df.to_excel(output_excel_file, index=False, engine='openpyxl')
    print("  Excel file saved successfully.")
except Exception as e:
    print(f"  Error saving Excel file: {e}")

print("\nScript finished.")