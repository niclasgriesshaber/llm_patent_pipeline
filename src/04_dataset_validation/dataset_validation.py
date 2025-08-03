import argparse
import os
import sys
import pandas as pd
import glob
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
    required_columns = ["id", "page", "entry", "category", "complete_patent", "patent_id", 
                       "name", "location", "description", "date", "successful_variable_extraction"]
    
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
    new_order = ["id", "page", "entry", "category", "patent_id", "name", "location", 
                 "description", "date", "complete_patent", "successful_variable_extraction"]
    
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


def validate_category_sequence(df, csv_path):
    """
    Validate that categories follow ascending order and report violations.
    
    Args:
        df: DataFrame to validate
        csv_path: Path to CSV file for error reporting
    
    Returns:
        dict: Dictionary with validation results and violations
    """
    # Define valid category sequence
    valid_categories = []
    for i in range(1, 90):  # 1 to 89
        valid_categories.append(str(i))
        if i <= 89:  # Add subclasses for categories that can have them
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                valid_categories.append(f"{i}{letter}")
    
    # Track violations
    violations = []
    current_category = None
    first_violation_id = None
    
    for idx, row in df.iterrows():
        category = str(row['category']).strip()
        entry_id = row['id']
        
        # Check if category is valid
        if category not in valid_categories:
            if first_violation_id is None or category != current_category:
                violations.append({
                    'id': entry_id,
                    'category': category,
                    'type': 'invalid_category',
                    'message': f"Invalid category '{category}' at id {entry_id}"
                })
                first_violation_id = entry_id
                current_category = category
            continue
        
        # Check ascending order
        if current_category is not None:
            current_idx = valid_categories.index(current_category)
            new_idx = valid_categories.index(category)
            
            if new_idx < current_idx:
                if first_violation_id is None or category != current_category:
                    violations.append({
                        'id': entry_id,
                        'category': category,
                        'type': 'sequence_violation',
                        'message': f"Category '{category}' appears before '{current_category}' at id {entry_id}"
                    })
                    first_violation_id = entry_id
                    current_category = category
                continue
        
        current_category = category
    
    return {
        'valid': len(violations) == 0,
        'violations': violations,
        'first_violation_ids': [v['id'] for v in violations]
    }


def validate_csv_file(csv_path, output_base_dir, start_id=None, end_id=None):
    """
    Validate a single CSV file and create both XLSX and log files.
    
    Args:
        csv_path: Path to the input CSV file
        output_base_dir: Base directory for outputs
        start_id: Lower bound of patent_id range to check for gaps
        end_id: Upper bound of patent_id range to check for gaps
    
    Returns:
        tuple: (xlsx_path, log_path) - paths to created files
    """
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None, None

    # Step 1: Validate required columns
    if not validate_required_columns(df, csv_path):
        return None, None

    # Step 2: Reorder columns
    df = reorder_columns(df)

    # Step 3: Clean patent_id column
    df = clean_patent_id_column(df)

    # Step 4: Validate category sequence
    category_validation = validate_category_sequence(df, csv_path)

    # Step 5: Original patent_id validation logic
    nan_patent_id_count = df['patent_id'].isna().sum()
    
    # Clean and convert patent_id for validation (keep original for display)
    def clean_patent_id_for_validation(val):
        if pd.isna(val):
            return None
        val = str(val).strip()
        if not val.isdigit():
            raise ValueError(f"patent_id '{val}' is not convertible to integer.")
        return int(val)

    try:
        df['patent_id_clean'] = df['patent_id'].apply(clean_patent_id_for_validation)
    except Exception as e:
        print(f"Error in {csv_path}: {e}")
        return None, None

    # Only use valid integer patent_ids for further checks
    valid_patent_ids_df = df[df['patent_id_clean'].notna()].copy()
    valid_patent_ids_df['patent_id_clean'] = valid_patent_ids_df['patent_id_clean'].astype(int)
    all_patent_ids = valid_patent_ids_df['patent_id_clean'].tolist()

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
        smaller_df = valid_patent_ids_df[valid_patent_ids_df['patent_id_clean'] < check_start_id]
        for pid, group in smaller_df.groupby('patent_id_clean'):
            entries = [(row['id'], row['page']) for _, row in group.iterrows()]
            smaller_than_start.append((pid, entries))
    if check_end_id is not None:
        greater_df = valid_patent_ids_df[valid_patent_ids_df['patent_id_clean'] > check_end_id]
        for pid, group in greater_df.groupby('patent_id_clean'):
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
    duplicates = valid_patent_ids_df[valid_patent_ids_df.duplicated('patent_id_clean', keep=False)]
    duplicate_groups = duplicates.groupby('patent_id_clean')

    # Prepare output file paths
    filestem = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Create XLSX file with validation results
    xlsx_path = os.path.join(output_base_dir, 'validated_xlsx', f"{filestem}_validated.xlsx")
    
    # Create a new DataFrame for the XLSX with validation information
    xlsx_df = df.copy()
    xlsx_df['validation_notes'] = ''
    
    # Add validation notes
    for idx, row in xlsx_df.iterrows():
        notes = []
        
        # Category validation notes
        category = str(row['category']).strip()
        for violation in category_validation['violations']:
            if violation['category'] == category and violation['id'] == row['id']:
                notes.append(violation['message'])
        
        # Patent ID validation notes
        if pd.isna(row['patent_id']):
            notes.append("NaN patent_id")
        else:
            try:
                pid_clean = clean_patent_id_for_validation(row['patent_id'])
                if pid_clean is not None:
                    if check_start_id is not None and pid_clean < check_start_id:
                        notes.append(f"patent_id {pid_clean} < start ({check_start_id})")
                    if check_end_id is not None and pid_clean > check_end_id:
                        notes.append(f"patent_id {pid_clean} > end ({check_end_id})")
                    if pid_clean in [dup_pid for dup_pid, _ in duplicate_groups]:
                        notes.append(f"Duplicate patent_id {pid_clean}")
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

    # Create log file
    log_path = os.path.join(output_base_dir, 'logs', f"{filestem}_validation_log.txt")
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Validation Report for {os.path.basename(csv_path)} ===\n\n")
        
        # Category validation section
        f.write("=== Category Sequence Validation ===\n")
        if category_validation['valid']:
            f.write("Category sequence is valid.\n")
        else:
            f.write("Category sequence violations found:\n")
            for violation in category_validation['violations']:
                f.write(f"  {violation['message']}\n")
        f.write("\n")
        
        f.write("=== NaN patent_id values ===\n")
        if nan_patent_id_count == 0:
            f.write("No NaN patent_id values found.\n")
        else:
            f.write(f"Number of NaN patent_id values: {nan_patent_id_count}\n")
        f.write("\n")
        f.write("=== Duplicate patent_id entries ===\n")
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
    return xlsx_path, log_path


def main():
    parser = argparse.ArgumentParser(description="Validate patent_id uniqueness and sequence in CSV files.")
    parser.add_argument('--csv', help='Path to a specific CSV file')
    parser.add_argument('--start', type=int, default=None, help='Lower bound of patent_id range to check for gaps')
    parser.add_argument('--end', type=int, default=None, help='Upper bound of patent_id range to check for gaps')
    args = parser.parse_args()

    # Hardcode input directory
    input_dir = 'data/03_variable_extraction/cleaned_with_variables_csvs/'
    output_dir = 'data/04_dataset_validation'

    # Determine input files
    if args.csv:
        # Single file mode
        csv_files = [args.csv]
    else:
        # Directory mode - find all CSV files
        if not os.path.isdir(input_dir):
            # Try relative to project root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, '../../'))
            input_dir = os.path.join(project_root, input_dir)
            if not os.path.isdir(input_dir):
                print(f"Error: Input directory '{input_dir}' does not exist.")
                sys.exit(1)
        
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        if not csv_files:
            print(f"No CSV files found in {input_dir}")
            sys.exit(1)

    # Create output directories
    output_base_dir = output_dir
    if not os.path.isabs(output_base_dir):
        # Make relative to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '../../'))
        output_base_dir = os.path.join(project_root, output_base_dir)
    
    os.makedirs(os.path.join(output_base_dir, 'validated_xlsx'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'validated_repaired_xlsx'), exist_ok=True)

    # Process each CSV file
    processed_count = 0
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        xlsx_path, log_path = validate_csv_file(csv_file, output_base_dir, args.start, args.end)
        if xlsx_path and log_path:
            processed_count += 1

    print(f"\nValidation complete. Processed {processed_count} files.")
    print(f"XLSX files saved to: {os.path.join(output_base_dir, 'validated_xlsx')}")
    print(f"Log files saved to: {os.path.join(output_base_dir, 'logs')}")
    print(f"Repaired XLSX files can be saved to: {os.path.join(output_base_dir, 'validated_repaired_xlsx')}")


if __name__ == "__main__":
    main() 