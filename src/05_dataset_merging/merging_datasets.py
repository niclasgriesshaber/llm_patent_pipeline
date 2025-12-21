import pandas as pd
import os
from pathlib import Path
import glob
import re

print("Starting the dataset merging script...")

# Path Configuration

script_location = Path(__file__).resolve().parent
project_root = script_location.parent.parent
input_dir = project_root / "data" / "05_dataset_merging" / "input_xlsx"
output_dir = project_root / "data" / "05_dataset_merging"
output_csv_file = output_dir / "german-patents-1877-1918.csv"
output_excel_file = output_dir / "german-patents-1877-1918.xlsx"

print(f"Script location: {script_location}")
print(f"Project root determined as: {project_root}")
print(f"Input Excel files directory: {input_dir}")
print(f"Output directory: {output_dir}")

# Find Excel files

file_pattern = "Patentamt_*.xlsx"
excel_files = sorted(list(input_dir.glob(file_pattern)))

if not excel_files:
    print(f"Error: No Excel files found matching '{file_pattern}' in {input_dir}")
    print("Please check the input directory and file naming convention.")
    exit()

print(f"Found {len(excel_files)} Excel files to merge:")
for f in excel_files:
    print(f"  - {f.name}")

# Define the expected column order
expected_column_order = [
    "global_id", "book", "book_id", "page", "entry", "category", 
    "patent_id", "name", "location", "description", "date", 
    "check_if_patent_complete", "successful_variable_extraction", 
    "patent_id_cleaned", "validation_notes"
]

# Merge Excel files

all_dataframes = []

for file_path in excel_files:
    try:
        # Extract the year from the filename using regex
        # Pattern: Patentamt_YYYY_cleaned_with_variables_validated_repaired.xlsx
        filename_stem = file_path.stem
        year_match = re.search(r'Patentamt_(\d{4})_', filename_stem)
        
        if year_match:
            year_str = year_match.group(1)
            print(f"  Processing {file_path.name}... extracted year: '{year_str}'")
        else:
            print(f"  Warning: Could not extract year from filename {file_path.name}")
            year_str = "unknown"
            print(f"  Using 'unknown' as year for {file_path.name}")

        # Read the current Excel file
        df = pd.read_excel(file_path)

        # Add the 'book' column at the beginning
        df.insert(0, 'book', year_str)

        # Rename 'id' to 'book_id' if it exists
        if 'id' in df.columns:
            df = df.rename(columns={'id': 'book_id'})

        # Ensure 'book_id' column is integer (never float)
        if 'book_id' in df.columns:
            df['book_id'] = pd.to_numeric(df['book_id'], errors='coerce').fillna(0).astype(int)

        # Ensure 'patent_id' column is string (to avoid float formatting in CSV)
        if 'patent_id' in df.columns:
            df['patent_id'] = pd.to_numeric(df['patent_id'], errors='coerce')
            df['patent_id'] = df['patent_id'].apply(lambda x: str(int(x)) if pd.notnull(x) and x == int(x) else str(x))

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

# Sort by 'book' chronologically (assuming book is a year, sorted as string is sufficient for now)
merged_df = merged_df.sort_values(by='book', kind='stable').reset_index(drop=True)

# Add global_id as the first column, counting from 1 to N
merged_df.insert(0, 'global_id', range(1, len(merged_df) + 1))

# Check for missing expected columns
missing_columns = [col for col in expected_column_order if col not in merged_df.columns]
if missing_columns:
    print(f"Warning: Missing expected columns: {missing_columns}")

# Get all columns that exist in the merged dataframe
existing_columns = list(merged_df.columns)

# Reorder columns: first the expected ones in order, then any additional ones
ordered_columns = []
for col in expected_column_order:
    if col in existing_columns:
        ordered_columns.append(col)

# Add any additional columns that weren't in the expected order
additional_columns = [col for col in existing_columns if col not in expected_column_order]
ordered_columns.extend(additional_columns)

# Reorder the dataframe
try:
    merged_df = merged_df[ordered_columns]
    print("Columns reordered successfully.")
    print(f"Final column order: {list(merged_df.columns)}")
except KeyError as e:
    print(f"Error: Could not find expected column {e} in the merged data. Please check input Excel files.")
    print("Cannot save output files.")
    exit()

print(f"Total rows in merged dataframe: {len(merged_df)}")

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