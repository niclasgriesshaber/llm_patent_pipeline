#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Set the input folder as defined in variable_extraction.py
INPUT_DIR="$SCRIPT_DIR/../../data/04_cleaned_csvs"

usage() {
    echo "Usage: $0 --csvs all | --csvs <csv1> <csv2> ..."
    exit 1
}

if [ "$1" != "--csvs" ]; then
    usage
fi

shift

CSV_LIST=()

if [ "$1" == "all" ]; then
    for file in "$INPUT_DIR"/*.csv; do
        [ -e "$file" ] || continue
        CSV_LIST+=("$file")
    done
    shift
else
    # Use provided filenames, check if they exist in the input directory
    while [ "$1" != "" ]; do
        CSV_PATH="$INPUT_DIR/$1"
        if [ -f "$CSV_PATH" ]; then
            CSV_LIST+=("$CSV_PATH")
        else
            echo "[WARNING] File not found: $CSV_PATH (skipping)"
        fi
        shift
    done
fi

if [ ${#CSV_LIST[@]} -eq 0 ]; then
    echo "[ERROR] No CSV files found to process."
    exit 1
fi

SUCCESS=0
FAIL=0

for CSV in "${CSV_LIST[@]}"; do
    CSV_BASENAME=$(basename "$CSV")
    echo "\n[INFO] Processing $CSV_BASENAME ..."
    python "$SCRIPT_DIR/variable_extraction.py" --csv "$CSV_BASENAME"
    if [ $? -eq 0 ]; then
        echo "[INFO] Successfully processed $CSV_BASENAME"
        SUCCESS=$((SUCCESS+1))
    else
        echo "[ERROR] Failed to process $CSV_BASENAME"
        FAIL=$((FAIL+1))
    fi
    echo "----------------------------------------"
    # Continue to next file regardless of error
done

echo "\n[SUMMARY]"
echo "Processed: $((SUCCESS+FAIL)) CSVs"
echo "Success:   $SUCCESS"
echo "Failed:    $FAIL"

exit 0 