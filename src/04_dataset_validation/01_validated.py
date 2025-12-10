import argparse
import os
import sys
import pandas as pd
import glob
import re
from pathlib import Path


def validate_required_columns(df, csv_path):
    """
    Validate that the CSV has exactly the required columns in the correct order.
    
    Args:
        df: DataFrame to validate
        csv_path: Path to CSV file for error reporting
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    required_columns = ["id", "page", "entry", "category", "check_if_patent_complete", 
                       "cleaning_API_fail", "double_incomplete", "patent_id", "name", 
                       "location", "description", "date", "successful_variable_extraction", 
                       "variable_API_fail"]
    
    actual_columns = list(df.columns)
    
    if actual_columns != required_columns:
        print(f"Error in {csv_path}: Column validation failed.")
        print(f"Expected: {required_columns}")
        print(f"Found: {actual_columns}")
        return False
    
    return True


def reorder_columns(df):
    """
    Reorder columns to the specified order.
    
    Args:
        df: DataFrame to reorder
    
    Returns:
        DataFrame: Reordered DataFrame
    """
    new_order = ["id", "page", "entry", "category", "patent_id", "name", 
                 "location", "description", "date", "check_if_patent_complete", 
                 "double_incomplete", "cleaning_API_fail", "successful_variable_extraction", 
                 "variable_API_fail"]
    
    return df[new_order]


def clean_patent_id_column(df):
    """
    Clean patent_id column by removing .0 suffix while preserving NaN values.
    
    Args:
        df: DataFrame with patent_id column
    
    Returns:
        DataFrame: DataFrame with cleaned patent_id column
    """
    def clean_patent_id(val):
        if pd.isna(val):
            return val  # Keep NaN as NaN
        val_str = str(val).strip()
        if val_str.endswith('.0'):
            return val_str[:-2]  # Remove .0 suffix
        return val_str
    
    df['patent_id'] = df['patent_id'].apply(clean_patent_id)
    return df


def parse_category(category_str):
    """
    Parse category string to extract main number and subcategory letter.
    
    Args:
        category_str: Category string (e.g., "1", "1a", "2b", "3")
    
    Returns:
        tuple: (main_number, subcategory_letter) where subcategory_letter is None if no letter
    """
    if pd.isna(category_str):
        return None, None
    
    category_str = str(category_str).strip()
    
    # Check if it's a simple number
    if category_str.isdigit():
        return int(category_str), None
    
    # Check if it's a number followed by a letter
    match = re.match(r'^(\d+)([a-z])$', category_str)
    if match:
        return int(match.group(1)), match.group(2)
    
    # Invalid format
    return None, None


def validate_category_sequence(df, csv_path, year):
    """
    Validate that categories follow ascending order and report violations.
    
    For years < 1900: All integers 1-89 must be present, monotonically ascending
    For years >= 1900: Flexible main classes with optional subclasses (e.g., 1, 1a, 1b, 2, 2a, 3)
    
    Args:
        df: DataFrame to validate
        csv_path: Path to CSV file for error reporting
        year: The year being processed (string)
    
    Returns:
        dict: Dictionary with validation results and violations
    """
    violations = []
    year_int = int(year)
    prev_category = None
    prev_main_num = None
    prev_sub_letter = None
    
    # Collect all categories for completeness check (years < 1900)
    all_main_categories = set()
    
    for idx, row in df.iterrows():
        category = str(row['category']).strip()
        entry_id = row['id']
        
        # Parse the category
        main_num, sub_letter = parse_category(category)
        
        if main_num is None:
            violations.append(f"Invalid category format '{category}' at row id {entry_id}")
            prev_category = category
            continue
        
        # Track all main categories for completeness check
        all_main_categories.add(main_num)
        
        # Check monotonic ascending order
        if prev_main_num is not None:
            violation = False
            
            if main_num < prev_main_num:
                # Main number decreased
                violations.append(f"Category '{category}' follows '{prev_category}' at row id {entry_id}")
                violation = True
            elif main_num == prev_main_num:
                # Same main number, check subclass order
                if prev_sub_letter is not None and sub_letter is not None:
                    if sub_letter < prev_sub_letter:
                        # Subclass decreased (not ascending)
                        violations.append(f"Category '{category}' follows '{prev_category}' at row id {entry_id}")
                        violation = True
                elif prev_sub_letter is not None and sub_letter is None:
                    # Going from subclass back to main class (invalid)
                    violations.append(f"Category '{category}' follows '{prev_category}' at row id {entry_id}")
                    violation = True
        
        # Update previous category tracking
        prev_category = category
        prev_main_num = main_num
        prev_sub_letter = sub_letter
    
    # For years < 1900, check that all categories 1-89 are present
    if year_int < 1900:
        expected_categories = set(range(1, 90))
        missing_categories = expected_categories - all_main_categories
        if missing_categories:
            missing_str = ', '.join(str(x) for x in sorted(missing_categories))
            violations.append(f"Missing categories (expected 1-89): {missing_str}")
    
    return {
        'valid': len(violations) == 0,
        'violations': violations
    }





def load_patent_ranges(ranges_csv_path):
    """
    Load patent ranges from CSV file.
    
    Args:
        ranges_csv_path: Path to the CSV file containing patent ranges
    
    Returns:
        dict: Dictionary mapping file prefix to (start, end) tuple
    """
    try:
        ranges_df = pd.read_csv(ranges_csv_path)
        ranges_dict = {}
        for _, row in ranges_df.iterrows():
            ranges_dict[row['file']] = (int(row['start']), int(row['end']))
        return ranges_dict
    except Exception as e:
        print(f"Error loading patent ranges from {ranges_csv_path}: {e}")
        return {}


def extract_year_from_filename(filename):
    """
    Extract year from filename like 'Patentamt_1886_RA_row_cleaned.xlsx'
    
    Args:
        filename: The filename to extract year from
    
    Returns:
        str: The year (e.g., '1886') or None if not found
    """
    match = re.search(r'Patentamt_(\d{4})_RA_row_cleaned\.xlsx', filename)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(description="Validate patent_id uniqueness and sequence in CSV files.")
    parser.add_argument('--csv', help='Path to a specific CSV file')
    args = parser.parse_args()

    # Load patent ranges from CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ranges_csv_path = os.path.join(script_dir, 'patent_ranges.csv')
    patent_ranges = load_patent_ranges(ranges_csv_path)
    
    if not patent_ranges:
        print("Error: Could not load patent ranges. Exiting.")
        sys.exit(1)
    
    print(f"Loaded patent ranges for {len(patent_ranges)} years: {list(patent_ranges.keys())}")

    # Define input and output directories
    input_dir = 'data/04_dataset_validation/RA_input/'
    output_dir = 'data/04_dataset_validation/01_validated'

    # Determine input files
    if args.csv:
        # Single file mode - try to find the file
        csv_file = args.csv
        
        # Check if it's an absolute path
        if os.path.isabs(csv_file):
            if os.path.exists(csv_file):
                csv_files = [csv_file]
            else:
                print(f"Error: File not found: {csv_file}")
                sys.exit(1)
        else:
            # Check if file exists in current directory
            if os.path.exists(csv_file):
                csv_files = [csv_file]
            else:
                # Check if file exists in input directory
                project_root = os.path.abspath(os.path.join(script_dir, '../../'))
                full_input_dir = os.path.join(project_root, input_dir)
                full_csv_path = os.path.join(full_input_dir, csv_file)
                
                if os.path.exists(full_csv_path):
                    csv_files = [full_csv_path]
                else:
                    print(f"Error: File not found: {csv_file}")
                    print(f"Checked in current directory and: {full_input_dir}")
                    sys.exit(1)
    else:
        # Directory mode - find all CSV files
        if not os.path.isdir(input_dir):
            # Try relative to project root
            project_root = os.path.abspath(os.path.join(script_dir, '../../'))
            input_dir = os.path.join(project_root, input_dir)
            if not os.path.isdir(input_dir):
                print(f"Error: Input directory '{input_dir}' does not exist.")
                sys.exit(1)
        
        csv_files = glob.glob(os.path.join(input_dir, "*.xlsx"))
        if not csv_files:
            print(f"No XLSX files found in {input_dir}")
            sys.exit(1)

    # Create output directories
    output_base_dir = output_dir
    if not os.path.isabs(output_base_dir):
        # Make relative to project root
        project_root = os.path.abspath(os.path.join(script_dir, '../../'))
        output_base_dir = os.path.join(project_root, output_base_dir)
    
    os.makedirs(os.path.join(output_base_dir, 'xlsx'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'csv'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'logs'), exist_ok=True)

    # Process each CSV file
    processed_count = 0
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        
        # Extract year from filename and get corresponding patent range
        filename = os.path.basename(csv_file)
        year = extract_year_from_filename(filename)
        
        if year is None:
            print(f"Warning: Could not extract year from filename '{filename}'. Skipping.")
            continue
        
        file_prefix = f"Patentamt_{year}"
        if file_prefix not in patent_ranges:
            print(f"Warning: No patent range found for year {year}. Skipping.")
            continue
        
        start_id, end_id = patent_ranges[file_prefix]
        print(f"Using patent range for {year}: {start_id} - {end_id}")
        
        # Read the XLSX file
        try:
            df = pd.read_excel(csv_file, dtype=str)
        except Exception as e:
            print(f"Error reading XLSX {csv_file}: {e}")
            continue
        
        # Validate required columns
        if not validate_required_columns(df, csv_file):
            continue
        
        # Reorder columns
        df = reorder_columns(df)
        
        # Clean patent_id column
        df = clean_patent_id_column(df)
        
        # Special preprocessing for 1879 and 1888: Keep rows with patent_id < start but flag them
        rows_flagged = 0
        if year in ['1879', '1888']:
            # Create temporary cleaned column for identification
            def get_patent_id_int(val):
                if pd.isna(val):
                    return None
                val_str = str(val).strip()
                if val_str.isdigit():
                    return int(val_str)
                return None
            
            df['_temp_patent_id'] = df['patent_id'].apply(get_patent_id_int)
            
            # Count rows that will be flagged (patent_id < start_id)
            rows_flagged = ((df['_temp_patent_id'].notna()) & (df['_temp_patent_id'] < start_id)).sum()
            
            if rows_flagged > 0:
                print(f"  Flagged {rows_flagged} rows with patent_id < {start_id} for auto-deletion (special handling for {year})")
            
            df = df.drop(columns=['_temp_patent_id'])
        
        # Validate category sequence
        category_validation = validate_category_sequence(df, csv_file, year)
        
        # Continue with the rest of validation...
        nan_patent_id_count = df['patent_id'].isna().sum()
        
        # Clean and convert patent_id for validation (keep original for display)
        def clean_patent_id_for_validation(val):
            """
            Clean and convert patent_id to integer.
            - Handles NaN values (returns None)
            - Strips whitespace (leading, trailing, and internal)
            - Removes .0 suffix if present (e.g., "6729.0" -> 6729)
            - Returns integer (not float)
            """
            if pd.isna(val):
                return None
            
            # Convert to string, strip leading/trailing whitespace, and remove internal spaces
            val_str = str(val).strip().replace(' ', '')
            
            # Remove .0 suffix if present (pandas sometimes converts ints to floats)
            if val_str.endswith('.0'):
                val_str = val_str[:-2]
            
            # Check if the cleaned value is a valid integer
            if not val_str.isdigit():
                raise ValueError(f"patent_id '{val}' is not convertible to integer.")
            
            # Return as integer (not float)
            return int(val_str)

        try:
            # Use Int64 (nullable integer) to avoid .0 suffix when mixing integers and NaN
            df['patent_id_cleaned'] = df['patent_id'].apply(clean_patent_id_for_validation)
            df['patent_id_cleaned'] = df['patent_id_cleaned'].astype('Int64')
        except Exception as e:
            print(f"Error in {csv_file}: {e}")
            continue

        # Only use valid integer patent_ids for further checks
        valid_patent_ids_df = df[df['patent_id_cleaned'].notna()].copy()
        # Already Int64, no need to convert again
        all_patent_ids = valid_patent_ids_df['patent_id_cleaned'].tolist()

        # Determine start and end for gap checking
        if start_id is not None:
            check_start_id = start_id
        else:
            check_start_id = min(all_patent_ids) if all_patent_ids else None
        if end_id is not None:
            check_end_id = end_id
        else:
            check_end_id = max(all_patent_ids) if all_patent_ids else None

        # Find out-of-range patent_ids with id and page
        smaller_than_start = []
        greater_than_end = []
        if check_start_id is not None:
            smaller_df = valid_patent_ids_df[valid_patent_ids_df['patent_id_cleaned'] < check_start_id]
            # For 1879 and 1888, don't include these in the validation report (they're auto-flagged)
            if year not in ['1879', '1888']:
                for pid, group in smaller_df.groupby('patent_id_cleaned'):
                    entries = [(row['id'], row['page']) for _, row in group.iterrows()]
                    smaller_than_start.append((pid, entries))
        if check_end_id is not None:
            greater_df = valid_patent_ids_df[valid_patent_ids_df['patent_id_cleaned'] > check_end_id]
            for pid, group in greater_df.groupby('patent_id_cleaned'):
                entries = [(row['id'], row['page']) for _, row in group.iterrows()]
                greater_than_end.append((pid, entries))

        # Find gaps within [check_start_id, check_end_id]
        if check_start_id is not None and check_end_id is not None:
            in_range_ids = set([pid for pid in all_patent_ids if check_start_id <= pid <= check_end_id])
            full_range = set(range(check_start_id, check_end_id + 1))
            missing_ids = sorted(full_range - in_range_ids)
        else:
            missing_ids = []

        # Find duplicates
        duplicates = valid_patent_ids_df[valid_patent_ids_df.duplicated('patent_id_cleaned', keep=False)]
        all_duplicate_ids = set(duplicates['patent_id_cleaned'].unique())  # For validation_notes
        
        # For 1879 and 1888, filter out duplicates with patent_id < start for the log file
        if year in ['1879', '1888'] and check_start_id is not None:
            duplicates_for_log = duplicates[duplicates['patent_id_cleaned'] >= check_start_id]
        else:
            duplicates_for_log = duplicates
        
        duplicate_groups = duplicates_for_log.groupby('patent_id_cleaned')

        # Prepare output file paths
        filestem = f"Patentamt_{year}"
        
        # Create XLSX file with validation results
        xlsx_path = os.path.join(output_base_dir, 'xlsx', f"{filestem}_validated.xlsx")
        
        # Create a new DataFrame for the XLSX with validation information
        xlsx_df = df.copy()
        xlsx_df['validation_notes'] = ''
        
        # Add validation notes
        for idx, row in xlsx_df.iterrows():
            notes = []
            
            # Category validation notes - check if this row is mentioned in violations
            row_id = str(row['id'])
            for violation in category_validation['violations']:
                if f"row id {row_id}" in violation or f"at row id {row_id}" in violation:
                    notes.append(f"Category violation: {violation}")
            
            # Patent ID validation notes
            if pd.isna(row['patent_id_cleaned']):
                notes.append("NaN patent_id")
            else:
                try:
                    # Use the already-cleaned patent_id_cleaned value
                    pid_clean = int(row['patent_id_cleaned'])
                    
                    # Check for duplicates FIRST (applies to all years, checks against ALL duplicates)
                    if pid_clean in all_duplicate_ids:
                        notes.append(f"Duplicate patent_id {pid_clean}")
                    
                    # Special handling for 1879 and 1888: Mark rows with patent_id < start for auto-deletion
                    if year in ['1879', '1888'] and check_start_id is not None and pid_clean < check_start_id:
                        notes.append(f"Auto-flagged for deletion: patent_id {pid_clean} < start ({check_start_id})")
                    else:
                        # Normal validation for other years or other issues
                        if check_start_id is not None and pid_clean < check_start_id:
                            notes.append(f"patent_id {pid_clean} < start ({check_start_id})")
                        if check_end_id is not None and pid_clean > check_end_id:
                            notes.append(f"patent_id {pid_clean} > end ({check_end_id})")
                except:
                    notes.append("Invalid patent_id format")
            
            xlsx_df.at[idx, 'validation_notes'] = '; '.join(notes) if notes else 'Valid'
        
        # Save XLSX file
        try:
            xlsx_df.to_excel(xlsx_path, index=False)
            print(f"Created XLSX file: {xlsx_path}")
        except Exception as e:
            print(f"Error creating XLSX file {xlsx_path}: {e}")
            xlsx_path = None
        
        # Save CSV file with validation notes
        csv_validated_path = os.path.join(output_base_dir, 'csv', f"{filestem}_validated.csv")
        try:
            xlsx_df.to_csv(csv_validated_path, index=False)
            print(f"Created CSV file: {csv_validated_path}")
        except Exception as e:
            print(f"Error creating CSV file {csv_validated_path}: {e}")

        # Create log file
        log_path = os.path.join(output_base_dir, 'logs', f"{filestem}_validation.txt")
        
        # Calculate summary counts
        num_duplicates = duplicate_groups.ngroups
        num_smaller_than_start = len(smaller_than_start)
        num_greater_than_end = len(greater_than_end)
        num_missing_ids = len(missing_ids)
        total_issues_excl_missing = nan_patent_id_count + num_duplicates + num_smaller_than_start + num_greater_than_end
        difference = abs(num_missing_ids - total_issues_excl_missing)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Validation Report for {os.path.basename(csv_file)} ===\n\n")
            
            # Log auto-flagged rows for 1879 and 1888
            if rows_flagged > 0:
                f.write(f"NOTE: {rows_flagged} rows with patent_id < {start_id} were auto-flagged for deletion (year {year})\n")
                f.write(f"These rows are kept in the dataset but marked for automatic deletion during manual validation.\n\n")
            
            # Summary Table at the top
            f.write("=" * 60 + "\n")
            f.write("VALIDATION SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Issue Type':<40} {'Count':>15}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'NaN patent_id values':<40} {nan_patent_id_count:>15}\n")
            f.write(f"{'Duplicate patent_id entries':<40} {num_duplicates:>15}\n")
            f.write(f"{'Patent IDs < start ({})'.format(check_start_id if check_start_id is not None else 'N/A'):<40} {num_smaller_than_start:>15}\n")
            f.write(f"{'Patent IDs > end ({})'.format(check_end_id if check_end_id is not None else 'N/A'):<40} {num_greater_than_end:>15}\n")
            f.write(f"{'Missing patent IDs (gaps)':<40} {num_missing_ids:>15}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Total issues (excl. missing)':<40} {total_issues_excl_missing:>15}\n")
            f.write(f"{'Difference (|Missing - Total|)':<40} {difference:>15}\n")
            f.write("=" * 60 + "\n\n")
            
            # Category validation section
            f.write("=== Category Sequence Validation ===\n")
            if category_validation['valid']:
                f.write("Valid\n")
            else:
                f.write("Invalid\n")
                f.write("\nViolations:\n")
                for violation in category_validation['violations']:
                    f.write(f"  - {violation}\n")
            f.write("\n")
            
            f.write("=== NaN patent_id values ===\n")
            if nan_patent_id_count == 0:
                f.write("No NaN patent_id values found.\n")
            else:
                f.write(f"Number of NaN patent_id values: {nan_patent_id_count}\n")
            f.write("\n")
            f.write("=== Duplicate patent_id entries ===\n")
            if year in ['1879', '1888'] and check_start_id is not None:
                f.write(f"Note: Only showing duplicates with patent_id >= {check_start_id}\n")
                f.write(f"(Duplicates with patent_id < {check_start_id} are auto-flagged for deletion)\n\n")
            if duplicate_groups.ngroups == 0:
                f.write("No duplicate patent_id values found.\n")
            else:
                for pid, group in duplicate_groups:
                    f.write(f"patent_id: {pid}\n")
                    for _, row in group.iterrows():
                        f.write(f"  id: {row['id']}, page: {row['page']}\n")
                    f.write("\n")
            f.write("\n=== patent_id values smaller than start ({}) ===\n".format(check_start_id if check_start_id is not None else 'N/A'))
            if not smaller_than_start:
                f.write("None found.\n")
            else:
                for pid, entries in smaller_than_start:
                    f.write(f"patent_id: {pid}\n")
                    for id_val, page_val in entries:
                        f.write(f"  id: {id_val}, page: {page_val}\n")
                    f.write("\n")
            f.write("\n=== patent_id values greater than end ({}) ===\n".format(check_end_id if check_end_id is not None else 'N/A'))
            if not greater_than_end:
                f.write("None found.\n")
            else:
                for pid, entries in greater_than_end:
                    f.write(f"patent_id: {pid}\n")
                    for id_val, page_val in entries:
                        f.write(f"  id: {id_val}, page: {page_val}\n")
                    f.write("\n")
            f.write("\n=== Missing patent_id values (gaps) between {} and {} ===\n".format(check_start_id, check_end_id))
            if not missing_ids:
                f.write("No gaps found in patent_id sequence.\n")
            else:
                f.write(", ".join(str(x) for x in missing_ids) + "\n")

        print(f"Created log file: {log_path}")
        processed_count += 1

    print(f"\nValidation complete. Processed {processed_count} files.")
    print(f"XLSX files saved to: {os.path.join(output_base_dir, 'xlsx')}")
    print(f"CSV files saved to: {os.path.join(output_base_dir, 'csv')}")
    print(f"Log files saved to: {os.path.join(output_base_dir, 'logs')}")


if __name__ == "__main__":
    main() 