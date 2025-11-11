# Manual Validation GUI - Usage Guide

## Overview

The `manually_validated.py` script provides an interactive GUI for research assistants to manually validate and correct patent data entries. It handles three types of validation issues:
- Duplicate patent IDs
- Patent IDs below the valid range (< start)
- Patent IDs above the valid range (> end)

## Prerequisites

### Installation

Ensure you have the required Python packages installed:

```bash
conda activate llm_patent_pipeline
pip install openpyxl  # If not already installed
```

### Data Structure

The script expects:
- Input: CSV files in `data/04_dataset_validation/validated_csv/`
- Images: PNG files in `data/01_dataset_construction/csvs/Patentamt_YEAR/page_by_page/PNG/`
- Output: 
  - CSV files in `data/04_dataset_validation/manually_validated_csv/`
  - XLSX files in `data/04_dataset_validation/manually_validated_xlsx/`

## Usage

### Basic Command

```bash
cd src/04_dataset_validation
python manually_validated.py --csv Patentamt_1878_cleaned_with_variables_validated.csv
```

### Running from Project Root

```bash
python src/04_dataset_validation/manually_validated.py --csv Patentamt_1878_cleaned_with_variables_validated.csv
```

## GUI Interface

### Layout

```
┌────────────────────────────────────────────────────────┐
│  Progress: Task 5/23 | Duplicate patent_id 2556       │
├────────────────────────────────────────────────────────┤
│              [Zoom Controls: - 100% + ]                │
│  ┌────────────────────────────────────────────────┐   │
│  │                                                 │   │
│  │         PAGE IMAGE (Scrollable)                │   │
│  │         Full Resolution Display                │   │
│  │                                                 │   │
│  └────────────────────────────────────────────────┘   │
├────────────────────────────────────────────────────────┤
│  Entry (from page_0001.png):                          │
│  ┌────────────────────────────────────────────────┐   │
│  │ [Multi-line text field for entry]              │   │
│  └────────────────────────────────────────────────┘   │
│  Patent ID: [_______]  [ ] Mark for deletion          │
├────────────────────────────────────────────────────────┤
│    [Manual Save]  [Back]  [Next Task]  [Finish&Exit]  │
└────────────────────────────────────────────────────────┘
```

## Keyboard Shortcuts (Left-Hand Friendly)

| Key | Action | Description |
|-----|--------|-------------|
| **A** | Back | Go to previous task (undo) |
| **D** | Next Task | Save and move to next task |
| **S** | Manual Save | Save current changes |
| **E** | Toggle Delete | Mark/unmark row for deletion |
| **Q** | Finish & Exit | Complete session and save |
| **Space** | Next Task | Alternative for Next Task |
| **Enter** | Next Field | Move between entry → patent_id → next task |

## Workflow

### 1. Start Session
- Run the script with the desired CSV file
- Window opens maximized showing the first validation issue

### 2. Review Entry
- View the PNG image at the top (full resolution, zoomable)
- Check the patent entry text against the image
- Verify the patent ID is correct

### 3. Edit if Needed
- Click in the Entry field or Patent ID field to edit
- Changes auto-save after 0.5 seconds of no typing
- Press Enter to move to next field

### 4. Mark for Deletion (Optional)
- If the entry is invalid, check "Mark for deletion"
- Or press **E** key to toggle
- Row will be deleted when you finish the session

### 5. Navigate
- Press **D** or **Space** to move to next task
- Press **A** to go back to previous task (full undo)
- Press **S** to manually save (redundant, but available)

### 6. Finish Session
- Press **Q** or click "Finish & Exit"
- Confirms deletion count
- Deletes marked rows
- Reindexes `id` column from 1 to N
- Saves final CSV and XLSX files
- Creates complete log file

## Features

### Auto-Save
- Changes are automatically saved 0.5 seconds after you stop typing
- Saves both CSV and XLSX versions
- Manual save button available for peace of mind

### Full Undo
- Press **A** to go back to previous task
- Unlimited undo throughout the session
- Restores previous data state

### Image Viewing
- Full resolution display
- Zoom in/out with **+** and **-** buttons
- Scrollbars for large images
- Mouse wheel scrolling supported

### Progress Tracking
- Shows current task number and total
- Displays validation issue type for current task
- Shows which page the current entry is from

### Logging
- Complete log file created at project root
- Format: `manually_validated_log_YEAR_TIMESTAMP.txt`
- Records all changes, deletions, and actions

## Output

### Files Created

1. **CSV File**: `data/04_dataset_validation/manually_validated_csv/Patentamt_YEAR_manually_validated.csv`
2. **XLSX File**: `data/04_dataset_validation/manually_validated_xlsx/Patentamt_YEAR_manually_validated.xlsx`
3. **Log File**: `manually_validated_log_YEAR_TIMESTAMP.txt` (at project root)

### Columns in Output

All original columns are preserved, plus:
- `manually_validated`: 1 for validated entries, 0 for not yet validated

### Row Indexing

At the end of the session:
- Marked rows are deleted
- `id` column is reindexed from 1 to N (sequential)

## Task Order

Tasks are processed in this order:
1. All duplicate patent ID issues
2. All "patent_id < start" issues
3. All "patent_id > end" issues

Each row with an issue is a separate task (one entry at a time).

## Tips for Research Assistants

1. **Keep left hand on keyboard**: A, D, S, E, Q keys control everything
2. **Use Enter key**: Press Enter to quickly move through fields
3. **Zoom as needed**: Use +/- buttons if text is hard to read
4. **Don't worry about saving**: Auto-save handles it, but manual save is there if you want
5. **Undo freely**: Press A to go back if you made a mistake
6. **Check carefully**: Look at the image to verify the correct patent ID and entry text
7. **Mark bad entries**: Use "Mark for deletion" for invalid entries

## Error Handling

- **Missing PNG file**: Warning shown, allows you to continue
- **Invalid CSV**: Clear error message with exit
- **Save errors**: Error dialog with details

## Example Session

```bash
# Activate environment
conda activate llm_patent_pipeline

# Navigate to script directory
cd src/04_dataset_validation

# Run validation for 1878
python manually_validated.py --csv Patentamt_1878_cleaned_with_variables_validated.csv

# GUI opens
# - Review first task (e.g., duplicate patent_id 2556)
# - Look at image, fix entry or patent ID if needed
# - Press D to go to next task
# - Continue through all 54 tasks
# - Press Q when done
# - Files saved to manually_validated_csv and manually_validated_xlsx folders
```

## Troubleshooting

### "Could not find CSV file"
- Ensure you're using the correct filename
- File should be in `data/04_dataset_validation/validated_csv/`

### "Image not found"
- Check that PNG files exist in `data/01_dataset_construction/csvs/Patentamt_YEAR/page_by_page/PNG/`
- Year in filename must match

### "ModuleNotFoundError: No module named 'openpyxl'"
- Run: `pip install openpyxl`

### Window too small
- Window should open maximized automatically
- If not, maximize manually or adjust screen resolution

## Questions?

Contact the project lead for any issues or questions about the validation workflow.

