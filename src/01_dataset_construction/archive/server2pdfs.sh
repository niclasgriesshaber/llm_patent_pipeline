#!/usr/bin/env bash

set -euo pipefail

##############################################
# - Lists *.tif files from an HTML index
# - Converts each to PDF with compression
# - Merges into one PDF for one or all folders
##############################################

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE_DIR="$( dirname "$SCRIPT_DIR" )"

# Server URL
BASE_URL="https://digi.bib.uni-mannheim.de/Akademie_Projekt/"

# Output directory relative to the workspace root (one level above script dir)
OUTPUT_DIR="${WORKSPACE_DIR}/data/pdfs/full_pdfs"
MAIN_TMP_DIR=""

# Logging function
# Usage: log_msg LEVEL "Message"
# Example: log_msg INFO "Script started."
log_msg() {
  local LEVEL="$1"
  local MSG="$2"
  # ISO 8601 format timestamp
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [$LEVEL] $MSG" >&2
}

# Ensure MAIN_TMP_DIR is cleaned up on exit, error, or interrupt
# This trap will be set after MAIN_TMP_DIR is created.
cleanup() {
  if [[ -n "$MAIN_TMP_DIR" && -d "$MAIN_TMP_DIR" ]]; then
    log_msg INFO "Cleaning up temporary directory: $MAIN_TMP_DIR"
    rm -rf "$MAIN_TMP_DIR"
  fi
}

usage() {
  echo "Usage: $0 [--list] | [--folder <folder_name|all>]" >&2
  echo "  --list             : List available folders on the server." >&2
  echo "  --folder <folder_name>: Process the specified folder." >&2
  echo "  --folder all       : Process all available folders (starting with 'Patentamt')." >&2
  exit 1
}

# Function to list folders from the base URL
list_folders() {
  log_msg INFO "Fetching directory listing from $BASE_URL..."
  local HTML_INDEX
  HTML_INDEX=$(curl -fsS "$BASE_URL")
  if [[ $? -ne 0 ]]; then
    log_msg ERROR "Failed to fetch directory listing from $BASE_URL"
    return 1 # Use return code within function
  fi

  # Extract directory links (ending with /), ignore parent link '../'
  # Use grep -oP for Perl-compatible regex might be more robust if available
  # Using extended regex -E for broader compatibility
  local FOLDERS
  FOLDERS=$(echo "$HTML_INDEX" \
    | grep -oE 'href="[a-zA-Z0-9_.-]+/"' \
    | grep -v 'href="\.\./"' \
    | sed 's/href="//; s/\/"$//')

  if [[ -z "$FOLDERS" ]]; then
    log_msg WARN "No folders found at $BASE_URL"
    # Print HTML for debugging if needed
    # log_msg DEBUG "--- START HTML INDEX ---"
    # log_msg DEBUG "$HTML_INDEX"
    # log_msg DEBUG "--- END HTML INDEX ---"
    return 1
  fi

  # Only echo the folder list itself to stdout for capturing
  echo "$FOLDERS"
  return 0
}

# Function to process a single folder
process_folder() {
  local FOLDER_NAME_ORIG="$1" # Keep original name for URL and temp dir
  FOLDER_NAME_ORIG="${FOLDER_NAME_ORIG%/}" # remove trailing slash if any

  # --- Clean Folder Name for Output PDF ---
  local FOLDER_NAME_CLEAN
  if [[ "$FOLDER_NAME_ORIG" == *_qk ]]; then
      FOLDER_NAME_CLEAN="${FOLDER_NAME_ORIG%_qk}"
      log_msg INFO "Cleaned folder name for output: '$FOLDER_NAME_ORIG' -> '$FOLDER_NAME_CLEAN'"
  else
      FOLDER_NAME_CLEAN="$FOLDER_NAME_ORIG"
  fi
  # --- End Clean Folder Name ---

  local REMOTE_URL="${BASE_URL}${FOLDER_NAME_ORIG}/" # Use original name for URL
  local FINAL_PDF="${OUTPUT_DIR}/${FOLDER_NAME_CLEAN}.pdf" # Use cleaned name for final PDF
  # Create a unique temp dir for this folder inside the main temp dir, use original name to match server structure if needed
  local TMP_DIR="$MAIN_TMP_DIR/$FOLDER_NAME_ORIG"
  mkdir -p "$TMP_DIR"

  log_msg INFO "================================="
  log_msg INFO "Processing Folder: $FOLDER_NAME_ORIG (Output: ${FOLDER_NAME_CLEAN}.pdf)"
  log_msg INFO "Remote dir URL:  $REMOTE_URL"
  log_msg INFO "Output PDF:      $FINAL_PDF"
  log_msg INFO "Temp dir:        $TMP_DIR"
  log_msg INFO "================================="
  echo # Keep one newline for visual separation between folders

  #####################################
  # 2) Grab all *.tif links from HTML #
  #####################################
  log_msg INFO "Fetching directory listing for $FOLDER_NAME_ORIG..."
  local HTML_INDEX
  HTML_INDEX=$(curl -fsS "$REMOTE_URL")
  if [[ $? -ne 0 ]]; then
      log_msg ERROR "Failed to fetch directory listing from $REMOTE_URL. Skipping folder $FOLDER_NAME_ORIG."
      return 1 # Indicate failure for this folder
  fi

  # Parse out "href="somefile.tif"" lines (case-insensitive)
  local ALL_TIFS
  ALL_TIFS=$(echo "$HTML_INDEX" \
    | grep -oEi 'href="[^"]+\.tif"' \
    | sed 's/^href="//i; s/"$//') # Added 'i' flag for robustness

  if [[ -z "$ALL_TIFS" ]]; then
    log_msg WARN "No .tif links found in $REMOTE_URL for folder $FOLDER_NAME_ORIG. Skipping."
    # Print HTML for debugging if needed
    # log_msg DEBUG "--- START HTML INDEX ($FOLDER_NAME_ORIG) ---"
    # log_msg DEBUG "$HTML_INDEX"
    # log_msg DEBUG "--- END HTML INDEX ($FOLDER_NAME_ORIG) ---"
    return 1 # Indicate failure/nothing to do
  fi

  ###################################################
  # 3) Download, Cleanup Name, Convert & Compress   #
  ###################################################
  local PDF_LIST=()
  # No longer cleaning _qk from TIF files here
  local TIF_LINK BASENAME TMP_TIF FULL_TIF_URL SINGLE_PDF CONVERT_CMD CONVERT_OUTPUT
  while IFS= read -r TIF_LINK; do
    # Build the full URL
    FULL_TIF_URL="${REMOTE_URL}${TIF_LINK}"
    BASENAME="$(basename "$TIF_LINK")"
    TMP_TIF="$TMP_DIR/$BASENAME"

    log_msg INFO "Downloading '$BASENAME'..."
    curl -fsSo "$TMP_TIF" "$FULL_TIF_URL"
    if [[ $? -ne 0 ]]; then
        log_msg ERROR "Failed to download $FULL_TIF_URL for folder $FOLDER_NAME_ORIG. Skipping file '$BASENAME'."
        continue # Skip this file
    fi

    SINGLE_PDF="$TMP_DIR/${BASENAME%.*}.pdf" # Use parameter expansion

    log_msg INFO "Converting '$BASENAME' to compressed PDF '$SINGLE_PDF'..."
    CONVERT_CMD=""
    CONVERT_OUTPUT=""
    if command -v magick &>/dev/null; then
      CONVERT_CMD="magick"
    elif command -v convert &>/dev/null; then
      CONVERT_CMD="convert"
    else
        log_msg ERROR "Neither 'magick' nor 'convert' (ImageMagick) found in PATH. Cannot convert TIFs."
        # No point continuing without converter for this folder
        # Clean up already downloaded TIF for this folder before returning
        rm -rf "$TMP_DIR" # Remove the folder-specific temp dir
        return 1 # Stop processing this folder
    fi

    # --- Add Compression during Conversion ---
    # Using JPEG compression (lossy) with quality 85.
    # Lower quality -> smaller size but more artifacts. Adjust if needed.
    # Alternatives: Zip (lossless), LZW, Fax, Group4
    log_msg INFO "Applying lossy JPEG compression (quality 85) during conversion..."
    CONVERT_OUTPUT=$($CONVERT_CMD "$TMP_TIF" -compress JPEG -quality 85 "$SINGLE_PDF" 2>&1)
    # --- End Compression during Conversion ---

    local CONVERT_STATUS=$?
    if [[ $CONVERT_STATUS -ne 0 ]]; then
        log_msg ERROR "Failed to convert '$BASENAME' to PDF for folder $FOLDER_NAME_ORIG."
        log_msg ERROR "Converter Output: $CONVERT_OUTPUT"
        log_msg WARN "Skipping file '$BASENAME' and continuing..."
        # rm -f "$TMP_TIF" "$SINGLE_PDF" # Clean up potentially broken files
        continue # Skip this file
    fi

    # Optional: Log size difference if needed (can be slow)
    # TIF_SIZE=$(ls -lh "$TMP_TIF" | awk '{print $5}')
    # PDF_SIZE=$(ls -lh "$SINGLE_PDF" | awk '{print $5}')
    # log_msg INFO "Conversion successful: $BASENAME ($TIF_SIZE) -> ${SINGLE_PDF##*/} ($PDF_SIZE)"
    log_msg INFO "Successfully converted '$BASENAME' to '$SINGLE_PDF'."

    PDF_LIST+=("$SINGLE_PDF")
    rm -f "$TMP_TIF" # Remove downloaded TIF now to save space
    log_msg INFO "Removed temporary TIF: $TMP_TIF"
  done <<< "$ALL_TIFS"

  local TOTAL_PDFS="${#PDF_LIST[@]}"
  if [[ "$TOTAL_PDFS" -eq 0 ]]; then
    log_msg WARN "No single-page PDFs were successfully created for folder $FOLDER_NAME_ORIG. Skipping merge."
    return 1 # Indicate failure/nothing to do
  fi

  log_msg INFO "Successfully created $TOTAL_PDFS compressed single-page PDF(s) for $FOLDER_NAME_ORIG."

  ##############################################
  # 4) Merge All Single-Page PDFs Into One PDF #
  ##############################################
  log_msg INFO "Merging $TOTAL_PDFS PDFs into final compressed PDF ($FINAL_PDF) using Ghostscript..."

  local MERGE_OUTPUT
  local MERGE_STATUS

  # Always use Ghostscript for merging to apply compression settings
  if ! command -v gs &>/dev/null; then
    log_msg ERROR "Ghostscript (gs) not found in PATH. Cannot merge and compress PDFs."
    return 1
  fi

  # Note: -dPDFSETTINGS=/printer typically uses 300dpi and JPEG compression (lossy)
  # Other options: /ebook (150dpi), /screen (72dpi), /prepress (high quality), /default
  log_msg INFO "Applying Ghostscript settings: -dPDFSETTINGS=/printer"
  MERGE_OUTPUT=$(gs -sDEVICE=pdfwrite \
                   -dCompatibilityLevel=1.4 \
                   -dPDFSETTINGS=/printer \
                   -dNOPAUSE \
                   -dBATCH \
                   -dQUIET \
                   -sOutputFile="$FINAL_PDF" \
                   "${PDF_LIST[@]}" 2>&1) # Input files are the list of single-page PDFs
  MERGE_STATUS=$?

  if [[ $MERGE_STATUS -ne 0 ]]; then
      log_msg ERROR "gs failed to merge and compress PDFs for folder $FOLDER_NAME_ORIG."
      log_msg ERROR "gs Output: $MERGE_OUTPUT"
      # Clean up potentially corrupted final PDF if gs failed
      rm -f "$FINAL_PDF"
      return 1
  fi

  log_msg INFO "Successfully created final compressed PDF for $FOLDER_NAME_ORIG: $FINAL_PDF"
  # Optional: Log final size
  FINAL_SIZE=$(ls -lh "$FINAL_PDF" | awk '{print $5}')
  log_msg INFO "Final PDF size: $FINAL_SIZE"

  # Cleanup of single-page PDFs happens via the main trap removing TMP_DIR
  log_msg INFO "Temporary single-page PDFs in $TMP_DIR will be removed on script exit."

  return 0 # Indicate success for this folder
}

# ====================
# Main Script Logic
# ====================

# 1) Parse arguments
if [[ $# -eq 0 ]]; then
  usage
fi

log_msg INFO "Script execution started."
log_msg INFO "Script directory: $SCRIPT_DIR"
log_msg INFO "Workspace directory: $WORKSPACE_DIR"
log_msg INFO "Output directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
# Create the main temporary directory *once*
MAIN_TMP_DIR="$(mktemp -d)"
log_msg INFO "Created main temporary directory: $MAIN_TMP_DIR"
# Set trap *after* MAIN_TMP_DIR is created
trap cleanup EXIT TERM INT

MODE=""
FOLDER_ARG=""

# Simple argument parsing loop
while [[ $# -gt 0 ]]; do
  case $1 in
    --list)
      MODE="list"
      shift # past argument
      ;;
    --folder)
      if [[ -z "${2:-}" ]]; then # Check if $2 exists and is not empty
        log_msg ERROR "--folder requires an argument (<folder_name> or 'all')."
        usage
      fi
      MODE="folder"
      FOLDER_ARG="$2"
      shift # past argument
      shift # past value
      ;;
    *) # unknown option
      log_msg ERROR "Unknown option: $1"
      usage
      ;;
  esac
done

# Validate that a mode was selected
if [[ -z "$MODE" ]]; then
    log_msg ERROR "No operation specified (--list or --folder)."
    usage
fi

# Execute based on mode
case "$MODE" in
  list)
    # Capture the output of list_folders
    FOLDER_LIST=$(list_folders)
    LIST_STATUS=$?
    # Check if list_folders succeeded
    if [[ $LIST_STATUS -eq 0 && -n "$FOLDER_LIST" ]]; then
        log_msg INFO "Available folders retrieved successfully."
        echo "Available folders:" # Keep this direct echo for stdout capture
        echo "$FOLDER_LIST"      # Keep this direct echo for stdout capture
    elif [[ $LIST_STATUS -eq 0 ]]; then
        # If list_folders succeeded but returned empty list
        log_msg WARN "Server listed, but no folders found."
    else
        # If list_folders failed (already logged error)
        exit 1
    fi
    log_msg INFO "Exiting after listing folders."
    exit 0 # Exit after listing
    ;;
  folder)
    if [[ "$FOLDER_ARG" == "all" ]]; then
      log_msg INFO "Processing mode: all folders (filtered for 'Patentamt*')."
      # Capture the list of folders
      ALL_FOLDERS=$(list_folders)
      LIST_STATUS=$?
      # Check if list_folders succeeded and returned folders
      if [[ $LIST_STATUS -ne 0 || -z "$ALL_FOLDERS" ]]; then
          log_msg ERROR "Could not retrieve folder list to process 'all'."
          exit 1
      fi

      # --- Filter for folders starting with Patentamt ---
      log_msg INFO "Filtering folder list for names starting with 'Patentamt'..."
      FILTERED_FOLDERS=$(echo "$ALL_FOLDERS" | grep '^Patentamt' || true) # Use || true to prevent exit if no match

      if [[ -z "$FILTERED_FOLDERS" ]]; then
          log_msg WARN "No folders starting with 'Patentamt' found in the list. Nothing to process."
          exit 0
      fi
      # --- End Filter ---

      log_msg INFO "Folders to process:"
      echo "$FILTERED_FOLDERS" # Log the filtered list
      echo # Add a newline for readability

      SUCCESS_COUNT=0
      FAIL_COUNT=0
      TOTAL_COUNT=0
      # Use process substitution and read loop for robust line handling
      while IFS= read -r FOLDER; do
          # Skip empty lines just in case (though grep shouldn't produce them)
          [[ -z "$FOLDER" ]] && continue
          TOTAL_COUNT=$((TOTAL_COUNT + 1))
          log_msg INFO "Starting processing for folder ($TOTAL_COUNT): $FOLDER"
          # Pass folder name to the processing function
          if process_folder "$FOLDER"; then
              SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
              log_msg INFO "Successfully completed processing for folder '$FOLDER'."
          else
              FAIL_COUNT=$((FAIL_COUNT + 1))
              # Error/Warning is logged within process_folder
              log_msg WARN "Processing failed or was skipped for folder '$FOLDER'."
          fi
      done <<< "$FILTERED_FOLDERS" # Feed the filtered list line by line

      log_msg INFO "================================="
      log_msg INFO "Finished processing all selected folders."
      log_msg INFO "Total folders attempted: $TOTAL_COUNT"
      log_msg INFO "Successful:            $SUCCESS_COUNT"
      log_msg INFO "Failed/Skipped:        $FAIL_COUNT"
      log_msg INFO "================================="
      # Exit with non-zero status if any folders failed
      if [[ "$FAIL_COUNT" -gt 0 ]]; then
          log_msg ERROR "Exiting with status 1 due to $FAIL_COUNT failed folder(s)."
          exit 1
      else
          log_msg INFO "All processed folders completed successfully."
          exit 0
      fi

    elif [[ -n "$FOLDER_ARG" ]]; then
      log_msg INFO "Processing single folder: $FOLDER_ARG"
      if process_folder "$FOLDER_ARG"; then
          log_msg INFO "Successfully completed processing for folder '$FOLDER_ARG'."
          exit 0
      else
          log_msg ERROR "Processing failed for folder '$FOLDER_ARG'. Exiting with status 1."
          exit 1
      fi
    else
      # Should not happen due to earlier checks, but safety first
      log_msg ERROR "Invalid state for --folder mode."
      usage
    fi
    ;;
  *)
    # Should not happen
    log_msg ERROR "Internal error: Invalid mode '$MODE'."
    exit 1
    ;;
esac

# Cleanup is handled by the trap
log_msg INFO "Script execution finished." # Should ideally be reached only in odd scenarios
exit 0 # Should be unreachable if logic above is correct, but good practice