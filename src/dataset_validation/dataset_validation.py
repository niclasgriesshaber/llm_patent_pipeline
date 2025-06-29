import argparse
import os
import sys
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Validate patent_id uniqueness and sequence in a CSV file.")
    parser.add_argument('--csv', required=True, help='Path to the input CSV file')
    parser.add_argument('--start', type=int, default=None, help='Lower bound of patent_id range to check for gaps')
    parser.add_argument('--end', type=int, default=None, help='Upper bound of patent_id range to check for gaps')
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.isfile(csv_path):
        # Find the project root (the directory containing 'data')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '../../'))
        alt_path = os.path.join(project_root, 'data', 'csvs_with_variables', os.path.basename(csv_path))
        if os.path.isfile(alt_path):
            csv_path = alt_path
        else:
            print(f"Error: File '{args.csv}' does not exist in the current directory or in 'data/csvs_with_variables' relative to the project root.")
            sys.exit(1)

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Check required columns
    for col in ['patent_id', 'id', 'page']:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV.")
            sys.exit(1)

    nan_patent_id_count = df['patent_id'].isna().sum()
    
    # Clean and convert patent_id
    def clean_patent_id(val):
        if pd.isna(val):
            return None  # Allow NaN to pass through for counting, but exclude from further checks
        val = str(val).strip()
        if not val.isdigit():
            raise ValueError(f"patent_id '{val}' is not convertible to integer.")
        return int(val)

    try:
        df['patent_id_clean'] = df['patent_id'].apply(clean_patent_id)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Only use valid integer patent_ids for further checks
    valid_patent_ids_df = df[df['patent_id_clean'].notna()].copy()
    valid_patent_ids_df['patent_id_clean'] = valid_patent_ids_df['patent_id_clean'].astype(int)
    all_patent_ids = valid_patent_ids_df['patent_id_clean'].tolist()

    # Determine start and end for gap checking
    if args.start is not None:
        start_id = args.start
    else:
        start_id = min(all_patent_ids) if all_patent_ids else None
    if args.end is not None:
        end_id = args.end
    else:
        end_id = max(all_patent_ids) if all_patent_ids else None

    # Find out-of-range patent_ids with id and page
    smaller_than_start = []
    greater_than_end = []
    if start_id is not None:
        smaller_df = valid_patent_ids_df[valid_patent_ids_df['patent_id_clean'] < start_id]
        for pid, group in smaller_df.groupby('patent_id_clean'):
            entries = [(row['id'], row['page']) for _, row in group.iterrows()]
            smaller_than_start.append((pid, entries))
    if end_id is not None:
        greater_df = valid_patent_ids_df[valid_patent_ids_df['patent_id_clean'] > end_id]
        for pid, group in greater_df.groupby('patent_id_clean'):
            entries = [(row['id'], row['page']) for _, row in group.iterrows()]
            greater_than_end.append((pid, entries))

    # Find gaps within [start_id, end_id]
    if start_id is not None and end_id is not None:
        in_range_ids = set([pid for pid in all_patent_ids if start_id <= pid <= end_id])
        full_range = set(range(start_id, end_id + 1))
        missing_ids = sorted(full_range - in_range_ids)
    else:
        missing_ids = []

    # Find duplicates
    duplicates = valid_patent_ids_df[valid_patent_ids_df.duplicated('patent_id_clean', keep=False)]
    duplicate_groups = duplicates.groupby('patent_id_clean')

    # Prepare output file path
    filestem = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(os.path.dirname(__file__), f"{filestem}_check.txt")

    with open(output_path, 'w', encoding='utf-8') as f:
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
        f.write("\n=== patent_id values smaller than start ({}) ===\n".format(start_id if start_id is not None else 'N/A'))
        if not smaller_than_start:
            f.write("None found.\n")
        else:
            for pid, entries in smaller_than_start:
                f.write(f"patent_id: {pid}\n")
                for id_val, page_val in entries:
                    f.write(f"  id: {id_val}, page: {page_val}\n")
                f.write("\n")
        f.write("\n=== patent_id values greater than end ({}) ===\n".format(end_id if end_id is not None else 'N/A'))
        if not greater_than_end:
            f.write("None found.\n")
        else:
            for pid, entries in greater_than_end:
                f.write(f"patent_id: {pid}\n")
                for id_val, page_val in entries:
                    f.write(f"  id: {id_val}, page: {page_val}\n")
                f.write("\n")
        f.write("\n=== Missing patent_id values (gaps) between {} and {} ===\n".format(start_id, end_id))
        if not missing_ids:
            f.write("No gaps found in patent_id sequence.\n")
        else:
            f.write(", ".join(str(x) for x in missing_ids) + "\n")

    print(f"Validation complete. Results written to {output_path}")

if __name__ == "__main__":
    main() 