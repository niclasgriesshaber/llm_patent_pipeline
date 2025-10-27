#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Set the correct input folder for PDFs
INPUT_DIR="$SCRIPT_DIR/../../data/pdfs/patent_pdfs"

usage() {
    echo "Usage: $0 --pdfs all | --pdfs <pdf1> <pdf2> ... [--max_workers <number>]"
    exit 1
}

if [ "$1" != "--pdfs" ]; then
    usage
fi

shift

PDF_LIST=()
MAX_WORKERS=""

if [ "$1" == "all" ]; then
    # Find all PDFs in the input directory (POSIX compatible)
    for file in "$INPUT_DIR"/*.pdf; do
        [ -e "$file" ] || continue
        PDF_LIST+=("$file")
    done
    shift
else
    # Use provided filenames, check if they exist in the input directory
    while [ "$1" != "" ] && [ "$1" != "--max_workers" ]; do
        PDF_PATH="$INPUT_DIR/$1"
        if [ -f "$PDF_PATH" ]; then
            PDF_LIST+=("$PDF_PATH")
        else
            echo "[WARNING] File not found: $PDF_PATH (skipping)"
        fi
        shift
    done
fi

# Check for --max_workers parameter
if [ "$1" == "--max_workers" ]; then
    shift
    if [ "$1" != "" ] && [ "$1" -gt 0 ] 2>/dev/null; then
        MAX_WORKERS="--max_workers $1"
    else
        echo "[ERROR] --max_workers requires a positive number"
        exit 1
    fi
fi

if [ ${#PDF_LIST[@]} -eq 0 ]; then
    echo "[ERROR] No PDF files found to process."
    exit 1
fi

SUCCESS=0
FAIL=0

for PDF in "${PDF_LIST[@]}"; do
    PDF_BASENAME=$(basename "$PDF")
    
    # Extract year from filename and determine prompt file
    # For files like "Patentamt_1878.pdf", extract "1878"
    YEAR=$(echo "$PDF_BASENAME" | sed 's/Patentamt_\([0-9]\{4\}\)\.pdf/\1/')
    
    if [ "$YEAR" = "1878" ] || [ "$YEAR" = "1879" ]; then
        PROMPT_ARG="--prompt special_volumes_prompt.txt"
        echo "\n[INFO] Processing $PDF_BASENAME (Year: $YEAR, using special volumes prompt)..."
    else
        PROMPT_ARG="--prompt prompt.txt"
        echo "\n[INFO] Processing $PDF_BASENAME (Year: $YEAR, using default prompt)..."
    fi
    
    python "$SCRIPT_DIR/gemini-2.5-parallel.py" --pdf "$PDF_BASENAME" $PROMPT_ARG $MAX_WORKERS
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