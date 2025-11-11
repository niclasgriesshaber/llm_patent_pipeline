# Manual Validation Script - Version 2 Updates

## Major Changes

### 1. **New Folder Structure**

#### Old Structure:
```
data/04_dataset_validation/
├── validated_csv/
├── validated_xlsx/
├── manually_validated_csv/
├── manually_validated_xlsx/
└── logs/
```

#### New Structure:
```
data/04_dataset_validation/
├── validated/
│   ├── csv/         (output from dataset_validation.py)
│   ├── xlsx/        (output from dataset_validation.py)
│   └── logs/        (validation logs)
└── manually_validated/
    ├── csv/         (working file, auto-saved during session)
    ├── xlsx/        (final output, created only when complete)
    └── logs/        (manual validation session logs)
```

### 2. **Entry Position Display**

- Added prominent entry position indicator showing "7/30" format
- Bold, blue text with raised border for easy visibility
- Shows current entry's position among all entries on the same page
- Example: If reviewing entry 7 out of 30 entries on page_0001.png, displays "7/30"

### 3. **Resume Functionality**

- Script now checks for existing `manually_validated/csv/Patentamt_YEAR_manually_validated.csv`
- If found: Loads it and resumes from where RA left off
- If not found: Starts fresh from `validated/csv/`
- Only includes rows with `manually_validated == '0'` in task queue
- Tasks sorted deterministically by row index for consistent ordering
- Console messages:
  - `✓ Resuming from previous session: X tasks remaining` (if resuming)
  - `✓ Starting new validation session` (if fresh start)

### 4. **Removed "Finish & Exit" Button**

- No manual exit button needed
- After last task, automatic completion sequence triggers
- Cleaner, more streamlined workflow

### 5. **Auto-Completion on Last Task**

When RA completes the last validation task:
1. Shows completion dialog with summary
2. Deletes all marked rows
3. Reindexes `id` column from 1 to N
4. Saves final CSV
5. Creates XLSX (only at this point)
6. Closes application automatically

Completion message includes:
- Total tasks processed
- Number of rows deleted
- Final row count
- Paths to saved files
- Log file location

### 6. **XLSX Creation Logic**

- **During session**: Only CSV is saved (auto-save)
- **At completion**: Both CSV and XLSX are created
- XLSX only exists when all validation is complete
- Ensures XLSX always represents fully validated data

### 7. **Updated Keyboard Shortcuts**

Removed Q (Finish & Exit) shortcut since completion is automatic:
- **A** = Back (undo)
- **D** = Next Task
- **S** = Manual Save
- **E** = Toggle Delete
- **Space** = Next Task (alternative)
- **Enter** = Next field

### 8. **Both Scripts Updated**

#### `dataset_validation.py`
- Updated to use new folder structure: `validated/csv`, `validated/xlsx`, `validated/logs`
- Input: `data/03_variable_extraction/manually_cleaned_with_variables/`
- Output: `data/04_dataset_validation/validated/`

#### `manually_validated.py`
- Updated to use new folder structure
- Input: `data/04_dataset_validation/validated/csv/`
- Output: `data/04_dataset_validation/manually_validated/`
- Backward compatible: still checks old `validated_csv` folder if new location not found

## Migration Instructions

To migrate to the new folder structure, run these commands from the project root:

```bash
# Create new folder structure
mkdir -p data/04_dataset_validation/validated/csv
mkdir -p data/04_dataset_validation/validated/xlsx
mkdir -p data/04_dataset_validation/validated/logs
mkdir -p data/04_dataset_validation/manually_validated/csv
mkdir -p data/04_dataset_validation/manually_validated/xlsx
mkdir -p data/04_dataset_validation/manually_validated/logs

# Move existing validated files
mv data/04_dataset_validation/validated_csv/* data/04_dataset_validation/validated/csv/ 2>/dev/null || true
mv data/04_dataset_validation/validated_xlsx/* data/04_dataset_validation/validated/xlsx/ 2>/dev/null || true
mv data/04_dataset_validation/logs/* data/04_dataset_validation/validated/logs/ 2>/dev/null || true

# Move existing manually validated files
mv data/04_dataset_validation/manually_validated_csv/* data/04_dataset_validation/manually_validated/csv/ 2>/dev/null || true
mv data/04_dataset_validation/manually_validated_xlsx/* data/04_dataset_validation/manually_validated/xlsx/ 2>/dev/null || true

# Optional: Remove old directories (only after verifying migration)
# rmdir data/04_dataset_validation/validated_csv
# rmdir data/04_dataset_validation/validated_xlsx
# rmdir data/04_dataset_validation/manually_validated_csv
# rmdir data/04_dataset_validation/manually_validated_xlsx
# rmdir data/04_dataset_validation/logs
```

## Workflow Changes

### Before (V1):
1. Run script with `--csv` parameter
2. Validate entries one by one
3. Click "Finish & Exit" when done
4. Both CSV and XLSX saved throughout

### After (V2):
1. Run script with `--csv` parameter (same)
2. Script automatically resumes if previous session exists
3. Validate entries one by one (same)
4. Entry position displayed prominently (NEW)
5. After last entry: automatic completion
6. XLSX only created when all tasks complete (NEW)
7. Can stop and resume anytime (NEW)

## Benefits

1. **Better organization**: Clear separation of validated vs. manually validated data
2. **Resume capability**: Can stop and continue later without losing progress
3. **Deterministic**: Same order every time, makes it easier to coordinate between RAs
4. **Visual guidance**: Entry position helps RA know where they are on each page
5. **Cleaner workflow**: No manual finish button, automatic completion
6. **Quality control**: XLSX only exists for complete datasets

## Technical Details

### Resume Detection
- Checks: `data/04_dataset_validation/manually_validated/csv/Patentamt_{YEAR}_manually_validated.csv`
- If exists and has `manually_validated == '0'` rows: Resume
- If exists and all `manually_validated == '1'`: No tasks (complete)
- If doesn't exist: Fresh start from `validated/csv/`

### Entry Position Calculation
```python
current_page = str(row['page'])
entries_on_page = df[df['page'] == current_page]
entry_position = (entries_on_page.index <= row_idx).sum()
total_on_page = len(entries_on_page)
# Display: f"{entry_position}/{total_on_page}"
```

### Completion Trigger
- Triggered when: `current_task_idx >= len(tasks)`
- Actions: Delete marked rows → Reindex → Save CSV → Save XLSX → Show message → Exit

## Backward Compatibility

Both scripts check for files in old locations as fallback:
- `validated_csv/` → falls back if `validated/csv/` not found
- Ensures smooth transition period

---

**Version**: 2.0  
**Date**: November 11, 2025  
**Status**: ✅ Complete and tested

