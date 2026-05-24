#!/usr/bin/env python3
"""
=============================================================================
REVISIONS FOR VSWG — CER-per-Year Line Plot (3 traces)
=============================================================================

Purpose:
    Creates an interactive Plotly.js line plot showing Character Error Rate (CER)
    by year with THREE traces:
    1. Multi-stage pipeline (Gemini-2.5-Pro + cleaning) — blue
    2. Single-step (Gemini-3.1-Pro-Preview, 1 call/page) — green
    3. Research Assistants (student baseline) — red

    This is copied and refined from create_cer_chart_html() in core/benchmarking.py
    (lines 981-1127), extended with the single-step trace.

    Uses identical CER computation to ensure fair comparison across all traces.

Data sources:
    - Multi-stage: data/benchmarking/results/02_dataset_cleaning/gemini-2.5-flash-lite/
                   cleaning_v0.1_prompt/llm_csv/ (the cleaned pipeline output)
    - Single-step:   data/benchmarking/results/revisions_for_VSWG_single_step/llm_csv/
    - Student:     data/benchmarking/input_data/transcriptions_xlsx/student_transcriptions_xlsx/
    - Perfect:     data/benchmarking/input_data/transcriptions_xlsx/perfect_transcriptions_xlsx/

Output:
    data/benchmarking/results/revisions_for_VSWG_single_step/revisions_for_VSWG_cer_by_year.html

Usage:
    python revisions_for_VSWG_single_step_cer_plot.py
=============================================================================
"""

import sys
import json
import logging
import re
from pathlib import Path
from rapidfuzz.distance import Levenshtein

# Add the scripts directory to path so we can import core modules
sys.path.insert(0, str(Path(__file__).parent))

from core.benchmarking import load_gt_file, load_llm_file

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKING_ROOT = PROJECT_ROOT / "data" / "benchmarking"

# Input directories
PERFECT_XLSX_DIR = BENCHMARKING_ROOT / "input_data" / "transcriptions_xlsx" / "perfect_transcriptions_xlsx"
STUDENT_XLSX_DIR = BENCHMARKING_ROOT / "input_data" / "transcriptions_xlsx" / "student_transcriptions_xlsx"

# Multi-stage pipeline results (after cleaning step — this is the published pipeline output)
MULTISTAGE_CSV_DIR = (BENCHMARKING_ROOT / "results" / "02_dataset_cleaning" /
                      "gemini-2.5-flash-lite" / "cleaning_v0.1_prompt" / "llm_csv")

# Single-step results
SINGLE_STEP_CSV_DIR = BENCHMARKING_ROOT / "results" / "revisions_for_VSWG_single_step" / "llm_csv"

# Output
OUTPUT_DIR = BENCHMARKING_ROOT / "results" / "revisions_for_VSWG_single_step"
OUTPUT_FILE = OUTPUT_DIR / "revisions_for_VSWG_cer_by_year.html"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_year(filename: str) -> str:
    """Extract 4-digit year from filename."""
    match = re.search(r'(\d{4})', filename)
    return match.group(1) if match else ''


def compute_cer_for_file(source_entries: list, perfect_entries: list) -> float:
    """
    Compute CER between source entries and perfect entries.
    Uses identical method as create_clean_text_for_cer() in core/benchmarking.py:
    - Concatenate all entries with spaces (raw text, no normalization)
    - Levenshtein normalized distance
    """
    source_text = ' '.join(str(e).strip() for e in source_entries if str(e).strip())
    perfect_text = ' '.join(str(e).strip() for e in perfect_entries if str(e).strip())

    if not perfect_text:
        return 0.0

    return Levenshtein.normalized_distance(perfect_text, source_text)


# =============================================================================
# DATA LOADING AND CER COMPUTATION
# =============================================================================

def compute_all_cer_by_year():
    """
    Compute CER per year for all three sources: multi-stage, single-step, student.
    Returns a list of dicts with year, multistage_cer, singlestep_cer, student_cer.
    """
    # Build lookup of perfect transcription files (normalize _perfected suffix)
    perfect_files = {}
    for f in PERFECT_XLSX_DIR.glob("*.xlsx"):
        stem = f.stem
        if stem.endswith('_perfected'):
            base_stem = stem[:-10]
        else:
            base_stem = stem
        perfect_files[base_stem] = f

    # Build lookup of student files
    student_files = {f.stem: f for f in STUDENT_XLSX_DIR.glob("*.xlsx")}

    # Build lookup of multi-stage files (have _cleaned suffix)
    multistage_files = {}
    for f in MULTISTAGE_CSV_DIR.glob("*.csv"):
        stem = f.stem
        if stem.endswith('_cleaned'):
            base_stem = stem[:-8]  # Remove '_cleaned'
        else:
            base_stem = stem
        multistage_files[base_stem] = f

    # Build lookup of single-step files
    singlestep_files = {f.stem: f for f in SINGLE_STEP_CSV_DIR.glob("*.csv")}

    # Find stems that have perfect transcription (required as ground truth)
    all_stems = sorted(perfect_files.keys())

    results = []
    for stem in all_stems:
        year = extract_year(stem)
        if not year:
            continue

        # Load perfect transcription entries
        perfect_df = load_gt_file(perfect_files[stem])
        if perfect_df.empty:
            continue
        perfect_entries = perfect_df['entry'].astype(str).tolist()

        row = {'year': int(year), 'file': stem}

        # Multi-stage CER
        if stem in multistage_files:
            ms_df = load_llm_file(multistage_files[stem])
            if not ms_df.empty:
                ms_entries = ms_df['entry'].astype(str).tolist()
                row['multistage_cer'] = compute_cer_for_file(ms_entries, perfect_entries)
            else:
                row['multistage_cer'] = None
        else:
            row['multistage_cer'] = None

        # Single-step CER
        if stem in singlestep_files:
            ss_df = load_llm_file(singlestep_files[stem])
            if not ss_df.empty:
                ss_entries = ss_df['entry'].astype(str).tolist()
                row['singlestep_cer'] = compute_cer_for_file(ss_entries, perfect_entries)
            else:
                row['singlestep_cer'] = None
        else:
            row['singlestep_cer'] = None

        # Student CER
        if stem in student_files:
            st_df = load_gt_file(student_files[stem])
            if not st_df.empty:
                st_entries = st_df['entry'].astype(str).tolist()
                row['student_cer'] = compute_cer_for_file(st_entries, perfect_entries)
            else:
                row['student_cer'] = None
        else:
            row['student_cer'] = None

        results.append(row)

    return sorted(results, key=lambda x: x['year'])


# =============================================================================
# PLOTLY HTML GENERATION (refined from core/benchmarking.py create_cer_chart_html)
# =============================================================================

def generate_plot_html(cer_data: list) -> str:
    """
    Generate standalone HTML with a 3-trace Plotly line chart.
    Refined from create_cer_chart_html() in core/benchmarking.py.
    """
    # Prepare data for each trace (only include years where data exists)
    ms_years = [d['year'] for d in cer_data if d.get('multistage_cer') is not None]
    ms_cers = [d['multistage_cer'] for d in cer_data if d.get('multistage_cer') is not None]

    ss_years = [d['year'] for d in cer_data if d.get('singlestep_cer') is not None]
    ss_cers = [d['singlestep_cer'] for d in cer_data if d.get('singlestep_cer') is not None]

    st_years = [d['year'] for d in cer_data if d.get('student_cer') is not None]
    st_cers = [d['student_cer'] for d in cer_data if d.get('student_cer') is not None]

    # Convert to JSON for JavaScript embedding
    ms_years_js = json.dumps(ms_years)
    ms_cers_js = json.dumps(ms_cers)
    ss_years_js = json.dumps(ss_years)
    ss_cers_js = json.dumps(ss_cers)
    st_years_js = json.dumps(st_years)
    st_cers_js = json.dumps(st_cers)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>VSWG Revisions — CER by Year (Single-Step vs Multi-Stage vs Student)</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #f8f9fa; }}
        .container {{ max-width: 1100px; margin: auto; padding: 30px; }}
        h1 {{ color: #2c3e50; text-align: center; }}
        .note {{ background: #e8f4fd; border-left: 4px solid #2196f3; padding: 15px; margin: 20px 0; font-size: 0.95em; }}
        .methodology {{ background: #f0f7e6; border-left: 4px solid #4caf50; padding: 15px; margin: 20px 0; font-size: 0.95em; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Character Error Rate by Year</h1>
    <p style="text-align:center; color:#666;">Comparison: Multi-Stage Pipeline vs Single-Step vs Research Assistants</p>

    <div class="methodology">
        <strong>Methodology:</strong> CER computed identically for all three traces —
        raw concatenated entry text compared via Levenshtein normalized distance
        against perfect (expert-verified) transcriptions. Lower CER = better performance.
    </div>

    <div id="cer-chart" style="height: 550px; margin: 30px 0;"></div>

    <div class="note">
        <strong>Vertical dashed lines</strong> mark transitions between typeface eras in the
        patent registers: Roman (1878–1893), Unger Fraktur (1894–1913), Breitkopf Fraktur (1914–1918).
    </div>

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script>
        // Trace 1: Multi-stage pipeline (Gemini-2.5-Pro + cleaning)
        var trace1 = {{
            x: {ms_years_js},
            y: {ms_cers_js},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Multi-Stage Pipeline',
            line: {{ color: '#2196f3', width: 3 }},
            marker: {{ size: 8, color: '#2196f3' }},
            hovertemplate: 'Year: %{{x}}<br>Multi-Stage CER: %{{y:.2%}}<extra></extra>'
        }};

        // Trace 2: Single-step (Gemini-3.1-Pro-Preview, 1 call/page)
        var trace2 = {{
            x: {ss_years_js},
            y: {ss_cers_js},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Single-Step (Gemini-3.1-Pro-Preview)',
            line: {{ color: '#ff9800', width: 3 }},
            marker: {{ size: 8, color: '#ff9800' }},
            hovertemplate: 'Year: %{{x}}<br>Single-Step CER: %{{y:.2%}}<extra></extra>'
        }};

        // Trace 3: Research Assistants (student baseline)
        var trace3 = {{
            x: {st_years_js},
            y: {st_cers_js},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Research Assistants',
            line: {{ color: '#f44336', width: 3 }},
            marker: {{ size: 8, color: '#f44336' }},
            hovertemplate: 'Year: %{{x}}<br>Research Assistants CER: %{{y:.2%}}<extra></extra>'
        }};

        var data = [trace1, trace2, trace3];

        var layout = {{
            title: {{
                text: 'Character Error Rate Comparison by Year',
                font: {{ size: 18, color: '#333' }}
            }},
            xaxis: {{
                title: 'Year',
                titlefont: {{ size: 14 }},
                tickfont: {{ size: 12 }},
                dtick: 2
            }},
            yaxis: {{
                title: 'Character Error Rate (CER)',
                titlefont: {{ size: 14 }},
                tickfont: {{ size: 12 }},
                tickformat: ',.1%'
            }},
            legend: {{
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.9)',
                bordercolor: '#ccc',
                borderwidth: 1
            }},
            shapes: [{{
                type: 'line',
                x0: 1893.5,
                x1: 1893.5,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: {{ color: '#666', width: 2, dash: 'dot' }}
            }}, {{
                type: 'line',
                x0: 1913.5,
                x1: 1913.5,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: {{ color: '#666', width: 2, dash: 'dot' }}
            }}],
            annotations: [{{
                x: 1886,
                y: 1.02,
                yref: 'paper',
                xanchor: 'center',
                text: '<i>Roman</i>',
                showarrow: false,
                font: {{ size: 11, color: '#444' }}
            }}, {{
                x: 1903.5,
                y: 1.02,
                yref: 'paper',
                xanchor: 'center',
                text: '<i>Unger Fraktur</i>',
                showarrow: false,
                font: {{ size: 11, color: '#444' }}
            }}, {{
                x: 1916,
                y: 1.02,
                yref: 'paper',
                xanchor: 'center',
                text: '<i>Breitkopf Fraktur</i>',
                showarrow: false,
                font: {{ size: 11, color: '#444' }}
            }}],
            margin: {{ t: 80, b: 60, l: 80, r: 40 }},
            plot_bgcolor: 'rgba(248,248,248,0.5)',
            paper_bgcolor: 'white'
        }};

        Plotly.newPlot('cer-chart', data, layout);
    </script>
</div>
</body>
</html>'''


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Compute CER by year for all three sources and generate the plot."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Computing CER by year for all three sources...")
    cer_data = compute_all_cer_by_year()

    if not cer_data:
        logging.error("No CER data computed. Check that input files exist.")
        return

    # Log summary
    ms_count = sum(1 for d in cer_data if d.get('multistage_cer') is not None)
    ss_count = sum(1 for d in cer_data if d.get('singlestep_cer') is not None)
    st_count = sum(1 for d in cer_data if d.get('student_cer') is not None)
    logging.info(f"Data points: Multi-stage={ms_count}, Single-step={ss_count}, Student={st_count}")

    # Generate HTML
    html_content = generate_plot_html(cer_data)
    OUTPUT_FILE.write_text(html_content, encoding='utf-8')
    logging.info(f"Plot saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
