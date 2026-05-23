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

Data sources (all under data/benchmarking/results/revisions_for_VSWG_zero_shot/):
    - Two-stage column   <- revisions_for_VSWG_comparison_table.csv  ("Multi-Stage Pipeline")
    - Single-step 3.1 Pro <- revisions_for_VSWG_consolidated_table.csv
    - Single-step 2.5 Pro <- revisions_for_VSWG_consolidated_table_gemini_2_5_pro.csv

Output:
    revisions_for_VSWG_comparison_table_3col.tex   (also printed to stdout)

Usage:
    python revisions_for_VSWG_zero_shot_latex_table.py
=============================================================================
"""

import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ZERO_SHOT_DIR = (PROJECT_ROOT / "data" / "benchmarking" / "results" /
                 "revisions_for_VSWG_zero_shot")

COMPARISON_CSV = ZERO_SHOT_DIR / "revisions_for_VSWG_comparison_table.csv"
CONSOLIDATED_31 = ZERO_SHOT_DIR / "revisions_for_VSWG_consolidated_table.csv"
CONSOLIDATED_25 = ZERO_SHOT_DIR / "revisions_for_VSWG_consolidated_table_gemini_2_5_pro.csv"

OUTPUT_TEX = ZERO_SHOT_DIR / "revisions_for_VSWG_comparison_table_3col.tex"

TOTAL_GT_ENTRIES = 1385  # constant across all approaches (same ground truth)


def load_kv(path: Path, key_col: str, val_col: str) -> dict:
    """Load a two-column CSV into a {key: value} dict."""
    df = pd.read_csv(path)
    return dict(zip(df[key_col].astype(str), df[val_col].astype(str)))


def thousands(n: int) -> str:
    """Format an integer with a LaTeX thin-space thousands separator: 1{,}385."""
    return f"{n:,}".replace(",", "{,}")


def pct_to_count(rate_str: str) -> int:
    """Convert a 'NN.NN%' rate string to an entry count out of TOTAL_GT_ENTRIES."""
    rate = float(rate_str.strip().rstrip('%'))
    return round(rate / 100.0 * TOTAL_GT_ENTRIES)


def entry_cell(matched: int, rate_str: str) -> str:
    """Render the 'X / N (NN.NN%)' entry-match cell."""
    rate = float(rate_str.strip().rstrip('%'))
    return f"{thousands(matched)}\\,/\\,{thousands(TOTAL_GT_ENTRIES)} ({rate:.2f}\\%)"


def main():
    # --- Two-stage pipeline column (from the 3-way comparison CSV) ---
    comp = load_kv(COMPARISON_CSV, "Metric", "Multi-Stage Pipeline")
    ts = {
        "cer": comp["CER (total)"],
        "entry_rate": comp["Entry Match Rate"].rstrip('%'),
        "entry_matched": pct_to_count(comp["Entry Match Rate"]),
        "patent_id": comp["Var: patent_id"].rstrip('%'),
        "name": comp["Var: name"].rstrip('%'),
        "location": comp["Var: location"].rstrip('%'),
        "description": comp["Var: description"].rstrip('%'),
        "date": comp["Var: date"].rstrip('%'),
        "overall": comp["Var: Overall"].rstrip('%'),
    }

    # --- Single-step columns (from consolidated tables) ---
    def load_single_step(path: Path) -> dict:
        c = load_kv(path, "Metric", "Value")
        return {
            "cer": c["CER (Zero-Shot vs Perfect)"],
            "entry_rate": c["Entry Match Rate (aggregate)"].rstrip('%'),
            "entry_matched": int(c["Matched Entries (total)"]),
            "patent_id": c["Variable: patent_id"].rstrip('%'),
            "name": c["Variable: name"].rstrip('%'),
            "location": c["Variable: location"].rstrip('%'),
            "description": c["Variable: description"].rstrip('%'),
            "date": c["Variable: date"].rstrip('%'),
            "overall": c["Overall Variable Match"].rstrip('%'),
        }

    g31 = load_single_step(CONSOLIDATED_31)
    g25 = load_single_step(CONSOLIDATED_25)

    # Column order requested: Two-Stage | Single-Step 2.5 Pro | Single-Step 3.1 Pro
    def row(label, key, is_pct=True):
        if key == "entry":
            a = entry_cell(ts["entry_matched"], ts["entry_rate"])
            b = entry_cell(g25["entry_matched"], g25["entry_rate"])
            c = entry_cell(g31["entry_matched"], g31["entry_rate"])
        elif key == "cer":
            a, b, c = ts["cer"], g25["cer"], g31["cer"]
        else:
            a = f"{float(ts[key]):.2f}\\%"
            b = f"{float(g25[key]):.2f}\\%"
            c = f"{float(g31[key]):.2f}\\%"
        return f"{label} & {a} & {b} & {c} \\\\"

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
{row("CER (aggregate)", "cer")}
{row("Entry match rate", "entry")}
\\midrule
\\multicolumn{{4}}{{l}}{{\\textit{{Field-level accuracy (matched entries)}}}} \\\\[2pt]
{row("Patent ID", "patent_id")}
{row("Name", "name")}
{row("Location", "location")}
{row("Description", "description")}
{row("Date", "date")}
\\midrule
{row("Overall", "overall")}
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item \\textit{{Notes.}} All three approaches decode at temperature~$0$ and are scored with identical methodology: CER is the Levenshtein normalized distance on the concatenated raw entry text across all 41~files; entry matching uses a greedy mutual-best algorithm with a 0.9 similarity threshold; field-level accuracy is the share of matched entry pairs where the extracted variable meets a 0.85 fuzzy similarity threshold. The two-stage pipeline uses Gemini~2.5~Pro for entry extraction and Gemini~2.5~Flash~Lite for cleaning and variable extraction; the single-step columns extract entries and all five fields in one call per page. The two single-step columns are configuration-identical (300~dpi page images, dynamic thinking budget) and differ only in the model. The two-stage pipeline reflects the production configuration (200~dpi images, fixed thinking budget) and so differs from the single-step columns in inference settings, though all columns share the same ground truth and evaluation.
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
