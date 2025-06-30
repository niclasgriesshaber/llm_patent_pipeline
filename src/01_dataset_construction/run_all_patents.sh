#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Set the correct input folder for PDFs
INPUT_DIR="$SCRIPT_DIR/../../data/pdfs/patent_pdfs"

usage() {
    echo "Usage: $0 --pdfs all | --pdfs <pdf1> <pdf2> ..."
    exit 1
}

if [ "$1" != "--pdfs" ]; then
    usage
fi

shift

PDF_LIST=()

if [ "$1" == "all" ]; then
    # Find all PDFs in the input directory (POSIX compatible)
    for file in "$INPUT_DIR"/*.pdf; do
        [ -e "$file" ] || continue
        PDF_LIST+=("$file")
    done
    shift
else
    # Use provided filenames, check if they exist in the input directory
    while [ "$1" != "" ]; do
        PDF_PATH="$INPUT_DIR/$1"
        if [ -f "$PDF_PATH" ]; then
            PDF_LIST+=("$PDF_PATH")
        else
            echo "[WARNING] File not found: $PDF_PATH (skipping)"
        fi
        shift
    done
fi

if [ ${#PDF_LIST[@]} -eq 0 ]; then
    echo "[ERROR] No PDF files found to process."
    exit 1
fi

SUCCESS=0
FAIL=0

for PDF in "${PDF_LIST[@]}"; do
    PDF_BASENAME=$(basename "$PDF")
    echo "\n[INFO] Processing $PDF_BASENAME ..."
    python "$SCRIPT_DIR/gemini-2.5-parallel.py" --pdf "$PDF_BASENAME"
    if [ $? -eq 0 ]; then
        echo "[INFO] Successfully processed $PDF_BASENAME"
        SUCCESS=$((SUCCESS+1))
    else
        echo "[ERROR] Failed to process $PDF_BASENAME"
        FAIL=$((FAIL+1))
    fi
    echo "----------------------------------------"
    # Continue to next file regardless of error
done

echo "\n[SUMMARY]"
echo "Processed: $((SUCCESS+FAIL)) PDFs"
echo "Success:   $SUCCESS"
echo "Failed:    $FAIL"

exit 0 