#!/usr/bin/env python3
"""
=============================================================================
REVISIONS FOR VSWG — 3-Column LaTeX Comparison Table Generator
=============================================================================

Purpose:
    Builds a single booktabs LaTeX table comparing three digitization
    approaches, all evaluated against the same expert-verified perfect
    transcriptions (41 sampled pages, N = 1,385 entries):

        Two-Stage Pipeline (Gemini 2.5 family)
        Single-Step (Gemini 2.5 Pro)
        Single-Step (Gemini 3.1 Pro)

    All three are scored with IDENTICAL methodology (same fuzzy thresholds,
    same CER = Levenshtein normalized distance on concatenated raw entry text),
    so the columns are directly comparable.

    The table is organized into three vertical content blocks:
        1. Transcription quality  — aggregate CER (as a percentage).
        2. Patent entry extraction — Perfect / Extracted / Matched entry counts,
           plus Recall (% Perfect matched) and Precision (% Extracted matched).
        3. Field-level accuracy   — per-variable match rate over MATCHED entries.

    Note on denominators: the entry-extraction Recall is over the 1,385 perfect
    entries, whereas the field-level rates are over the (smaller) set of matched
    entries. The two can coincide numerically (e.g. 3.1 Pro shows 99.93% for both
    Recall and Patent ID) without being the same quantity.

Data sources (all under data/benchmarking/results/revisions_for_VSWG_single_step/):
    - Two-stage column    <- revisions_for_VSWG_comparison_table.csv  ("Multi-Stage Pipeline")
    - Single-step 3.1 Pro <- revisions_for_VSWG_consolidated_table.csv  + llm_csv/
    - Single-step 2.5 Pro <- revisions_for_VSWG_consolidated_table_gemini_2_5_pro.csv + llm_csv_gemini_2_5_pro/

    Extracted-entry counts are obtained by counting non-empty `entry` rows in the
    single-step inference CSV directories (no inference is re-run).

Output:
    revisions_for_VSWG_comparison_table_3col.tex   (also printed to stdout)

Usage:
    python revisions_for_VSWG_single_step_latex_table.py
=============================================================================
"""

import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SINGLE_STEP_DIR = (PROJECT_ROOT / "data" / "benchmarking" / "results" /
                 "revisions_for_VSWG_single_step")

COMPARISON_CSV = SINGLE_STEP_DIR / "revisions_for_VSWG_comparison_table.csv"
CONSOLIDATED_31 = SINGLE_STEP_DIR / "revisions_for_VSWG_consolidated_table.csv"
CONSOLIDATED_25 = SINGLE_STEP_DIR / "revisions_for_VSWG_consolidated_table_gemini_2_5_pro.csv"

LLM_CSV_DIR_31 = SINGLE_STEP_DIR / "llm_csv"
LLM_CSV_DIR_25 = SINGLE_STEP_DIR / "llm_csv_gemini_2_5_pro"

OUTPUT_TEX = SINGLE_STEP_DIR / "revisions_for_VSWG_comparison_table_3col.tex"

TOTAL_GT_ENTRIES = 1385  # constant across all approaches (same ground truth)


def load_kv(path: Path, key_col: str, val_col: str) -> dict:
    """Load a two-column CSV into a {key: value} dict."""
    df = pd.read_csv(path)
    return dict(zip(df[key_col].astype(str), df[val_col].astype(str)))


def thousands(n: int) -> str:
    """Format an integer with a LaTeX thin-space thousands separator: 1{,}385."""
    return f"{n:,}".replace(",", "{,}")


def count_extracted_entries(llm_dir: Path) -> int:
    """Count non-empty `entry` rows across all CSVs in a single-step inference dir.

    Mirrors load_llm_csv_with_variables() in the benchmarking script: a row counts
    as an extracted entry iff its `entry` cell is non-null and non-blank.
    """
    total = 0
    for f in sorted(llm_dir.glob("*.csv")):
        df = pd.read_csv(f, dtype=str)
        if 'entry' not in df.columns:
            continue
        total += int((df['entry'].notna() & (df['entry'].str.strip() != '')).sum())
    return total


def main():
    # --- Two-stage pipeline column (from the 3-way comparison CSV) ---
    comp = load_kv(COMPARISON_CSV, "Metric", "Multi-Stage Pipeline")
    ts = {
        "cer": comp["CER (total)"],
        "perfect": int(comp["Perfect Entries"]),
        "extracted": int(comp["Extracted Entries"]),
        "matched": int(comp["Matched Entries"]),
        "patent_id": comp["Var: patent_id"].rstrip('%'),
        "name": comp["Var: name"].rstrip('%'),
        "location": comp["Var: location"].rstrip('%'),
        "description": comp["Var: description"].rstrip('%'),
        "date": comp["Var: date"].rstrip('%'),
        "overall": comp["Var: Overall"].rstrip('%'),
    }

    # --- Single-step columns (consolidated CSV + extracted-entry count from llm_csv) ---
    def load_single_step(path: Path, llm_dir: Path) -> dict:
        c = load_kv(path, "Metric", "Value")
        return {
            "cer": c["CER (Single-Step vs Perfect)"],
            "perfect": int(c["GT Entries (total)"]),
            "extracted": count_extracted_entries(llm_dir),
            "matched": int(c["Matched Entries (total)"]),
            "patent_id": c["Variable: patent_id"].rstrip('%'),
            "name": c["Variable: name"].rstrip('%'),
            "location": c["Variable: location"].rstrip('%'),
            "description": c["Variable: description"].rstrip('%'),
            "date": c["Variable: date"].rstrip('%'),
            "overall": c["Overall Variable Match"].rstrip('%'),
        }

    g31 = load_single_step(CONSOLIDATED_31, LLM_CSV_DIR_31)
    g25 = load_single_step(CONSOLIDATED_25, LLM_CSV_DIR_25)

    # Derived recall / precision for each column.
    for col in (ts, g25, g31):
        col["recall"] = col["matched"] / col["perfect"] * 100
        col["precision"] = col["matched"] / col["extracted"] * 100

    # Column order: Two-Stage | Single-Step 2.5 Pro | Single-Step 3.1 Pro
    cols = (ts, g25, g31)

    def count_row(label, key):
        cells = " & ".join(thousands(col[key]) for col in cols)
        return f"{label} & {cells} \\\\"

    def pct_row(label, key):
        cells = " & ".join(f"{float(col[key]):.2f}\\%" for col in cols)
        return f"{label} & {cells} \\\\"

    def cer_row(label):
        # Stored CER is a normalized distance in [0,1]; report it as a percentage.
        cells = " & ".join(f"{float(col['cer']) * 100:.2f}\\%" for col in cols)
        return f"{label} & {cells} \\\\"

    def derived_pct_row(label, key):
        cells = " & ".join(f"{col[key]:.2f}\\%" for col in cols)
        return f"{label} & {cells} \\\\"

    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Benchmarking results: two-stage pipeline (Gemini~2.5 family) vs.\\ single-step extraction with Gemini~2.5~Pro and Gemini~3.1~Pro. All metrics are evaluated against expert-verified perfect transcriptions across 41~sampled pages ($N = {thousands(TOTAL_GT_ENTRIES)}$ entries).}}
\\label{{tab:pipeline_comparison}}
\\begin{{tabular}}{{lccc}}
\\toprule
 & \\textbf{{Two-Stage Pipeline}} & \\textbf{{Single-Step}} & \\textbf{{Single-Step}} \\\\
 & \\textit{{Gemini 2.5 family}} & \\textit{{Gemini 2.5 Pro}} & \\textit{{Gemini 3.1 Pro}} \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textit{{Transcription quality}}}} \\\\[2pt]
{cer_row("CER (aggregate)")}
\\midrule
\\multicolumn{{4}}{{l}}{{\\textit{{Patent entry extraction}}}} \\\\[2pt]
{count_row("\\textit{Perfect} Entries", "perfect")}
{count_row("Extracted Entries", "extracted")}
{count_row("Matched Entries", "matched")}
{derived_pct_row("\\% \\textit{Perfect} Matched (Recall)", "recall")}
{derived_pct_row("\\% Extracted Matched (Precision)", "precision")}
\\midrule
\\multicolumn{{4}}{{l}}{{\\textit{{Field-level accuracy (matched entries)}}}} \\\\[2pt]
{pct_row("Patent ID", "patent_id")}
{pct_row("Name", "name")}
{pct_row("Location", "location")}
{pct_row("Description", "description")}
{pct_row("Date", "date")}
\\midrule
{pct_row("Overall", "overall")}
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item \\textit{{Notes.}} All three approaches decode at temperature~$0$ and are scored with identical methodology. CER (aggregate) is the Levenshtein normalized distance on the concatenated raw entry text across all 41~files, reported as a percentage. Entry matching uses a greedy mutual-best algorithm with a 0.9 similarity threshold; Recall is the share of the {thousands(TOTAL_GT_ENTRIES)}~\\textit{{perfect}} entries that are matched, and Precision is the share of \\textit{{extracted}} entries that are matched. Field-level accuracy is the share of \\textit{{matched entry pairs}} where the extracted variable meets a 0.85 fuzzy similarity threshold; its denominator is therefore the number of matched entries, which differs from the {thousands(TOTAL_GT_ENTRIES)}~perfect total. The two-stage pipeline uses Gemini~2.5~Pro for entry extraction and Gemini~2.5~Flash~Lite for cleaning and variable extraction; the single-step columns extract entries and all five fields in one call per page. The two single-step columns are configuration-identical (dynamic thinking budget) and differ only in the model, whereas the two-stage pipeline reflects the production configuration (fixed thinking budget) and therefore differs from the single-step columns in inference settings, though all columns share the same ground truth and evaluation.
\\end{{tablenotes}}
\\end{{table}}
"""

    OUTPUT_TEX.write_text(tex, encoding="utf-8")
    logging.info(f"LaTeX table written to: {OUTPUT_TEX}")
    print("\n" + "=" * 78)
    print("LaTeX table (copy-paste into Overleaf):")
    print("=" * 78 + "\n")
    print(tex)


if __name__ == "__main__":
    main()
