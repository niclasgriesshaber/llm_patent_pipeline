import os
import pandas as pd
from pathlib import Path
from rapidfuzz.distance import JaroWinkler
from collections import defaultdict
from typing import List, Tuple, Dict
import re
import argparse

BENCHMARKING_DIR = Path(__file__).parent.parent.parent / 'data' / 'benchmarking'
GT_DIR = BENCHMARKING_DIR / 'gt_xlsx'
LLM_DIR = BENCHMARKING_DIR / 'llm_csv'
STRICT_HTML = Path(__file__).parent / 'benchmarking_strict.html'
FUZZY_HTML = Path(__file__).parent / 'benchmarking_fuzzy.html'

# Helper to get all file stems in a directory with a given extension
def get_file_stems(directory: Path, ext: str) -> set:
    return set(f.stem for f in directory.glob(f'*.{ext}'))

def load_gt_file(filepath: Path) -> pd.DataFrame:
    df = pd.read_excel(filepath, dtype=str)
    if 'id' not in df.columns or 'entry' not in df.columns:
        raise ValueError(f"File {filepath} missing 'id' or 'entry' column.")
    df = df[['id', 'entry']].dropna(subset=['entry'])
    df = df[df['entry'].astype(str).str.strip() != '']
    return df.reset_index(drop=True)

def load_llm_file(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, dtype=str)
    if 'id' not in df.columns or 'entry' not in df.columns:
        raise ValueError(f"File {filepath} missing 'id' or 'entry' column.")
    df = df[['id', 'entry']].dropna(subset=['entry'])
    df = df[df['entry'].astype(str).str.strip() != '']
    return df.reset_index(drop=True)

def match_entries_with_ids(gt_df: pd.DataFrame, llm_df: pd.DataFrame, mode: str, threshold: float = 0.8) -> Tuple[List[bool], List[bool], List[str], List[str]]:
    gt_entries = gt_df['entry'].astype(str).tolist()
    llm_entries = llm_df['entry'].astype(str).tolist()
    gt_ids = gt_df['id'].astype(str).tolist()
    llm_ids = llm_df['id'].astype(str).tolist()
    gt_matches = [False] * len(gt_entries)
    llm_matches = [False] * len(llm_entries)
    gt_match_ids = ['—'] * len(gt_entries)
    llm_match_ids = ['—'] * len(llm_entries)
    if mode == 'strict':
        llm_lookup = defaultdict(list)
        for i, e in enumerate(llm_entries):
            llm_lookup[e].append(i)
        for i, gt in enumerate(gt_entries):
            if gt in llm_lookup:
                idx = llm_lookup[gt][0]  # Only the first match
                gt_matches[i] = True
                llm_matches[idx] = True
                gt_match_ids[i] = llm_ids[idx]
                llm_match_ids[idx] = gt_ids[i]
    elif mode == 'fuzzy':
        # For GT -> LLM
        for i, gt in enumerate(gt_entries):
            best_score = threshold
            best_idx = -1
            for j, llm in enumerate(llm_entries):
                score = JaroWinkler.similarity(gt, llm, prefix_weight=0.9)
                if score > best_score:
                    best_score = score
                    best_idx = j
            if best_idx != -1:
                gt_matches[i] = True
                llm_matches[best_idx] = True
                gt_match_ids[i] = llm_ids[best_idx]
        # For LLM -> GT
        for j, llm in enumerate(llm_entries):
            best_score = threshold
            best_idx = -1
            for i, gt in enumerate(gt_entries):
                score = JaroWinkler.similarity(gt, llm, prefix_weight=0.9)
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx != -1:
                llm_matches[j] = True
                gt_matches[best_idx] = True  # This may already be set
                llm_match_ids[j] = gt_ids[best_idx]
    else:
        raise ValueError('Unknown mode')
    return gt_matches, llm_matches, gt_match_ids, llm_match_ids

def html_escape(s: str) -> str:
    return (str(s)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))

def make_table_html(df: pd.DataFrame, matches: List[bool], match_ids: List[str], title: str) -> str:
    table = f'<table class="benchmark-table">\n<caption>{title}</caption>\n<tr><th>id</th><th>entry</th><th>match_id</th></tr>'
    for i, row in df.iterrows():
        color = '#d4edda' if matches[i] else '#f8d7da'  # green/red
        match_id_str = html_escape(match_ids[i])
        table += f'<tr style="background-color:{color}"><td>{html_escape(row["id"])}</td><td>{html_escape(row["entry"])}</td><td>{match_id_str}</td></tr>'
    table += '</table>'
    return table

def make_metrics_html(gt_matches, llm_matches, gt_len, llm_len) -> str:
    gt_matched = sum(gt_matches)
    llm_matched = sum(llm_matches)
    gt_unmatched = gt_len - gt_matched
    llm_unmatched = llm_len - llm_matched
    total = gt_len + llm_len
    total_matched = gt_matched + llm_matched
    match_rate = (total_matched / total * 100) if total else 0
    return f'''
    <div class="metrics">
        <b>GT entries:</b> {gt_len} &nbsp; <b>LLM entries:</b> {llm_len} &nbsp; <b>GT matched:</b> {gt_matched} &nbsp; <b>LLM matched:</b> {llm_matched} &nbsp; <b>GT unmatched:</b> {gt_unmatched} &nbsp; <b>LLM unmatched:</b> {llm_unmatched} &nbsp; <b>Overall match rate:</b> {match_rate:.1f}%<br>
        <span style="font-size:0.95em; color:#555;">Match rate = (GT matched + LLM matched) / (GT entries + LLM entries)</span>
    </div>
    '''

def extract_year_from_stem(stem: str) -> int:
    m = re.search(r'Patentamt_(\d{4})_sampled', stem)
    if m:
        return int(m.group(1))
    return -1

def make_pair_section_html(gt_df, llm_df, gt_matches, llm_matches, gt_match_ids, llm_match_ids, filename_stem, mode) -> str:
    metrics = make_metrics_html(gt_matches, llm_matches, len(gt_df), len(llm_df))
    gt_table = make_table_html(gt_df, gt_matches, gt_match_ids, 'Ground Truth')
    llm_table = make_table_html(llm_df, llm_matches, llm_match_ids, 'LLM Output')
    return f'''
    <section class="pair-section">
        <h2>File Pair: <span class="filename">{html_escape(filename_stem)}</span> <span class="mode">({mode.title()} Mode)</span></h2>
        {metrics}
        <div class="table-row">
            <div class="table-col">{gt_table}</div>
            <div class="table-col">{llm_table}</div>
        </div>
    </section>
    '''

def make_legend_html() -> str:
    return '''
    <section class="legend-section">
        <h1 style="text-align:center; margin-top:32px;">Benchmarking Results</h1>
        <div class="legend-box" style="background:#fff; border-radius:8px; max-width:900px; margin:0 auto 32px auto; padding:24px; box-shadow:0 2px 8px rgba(0,0,0,0.07);">
            <h2>Legend & Explanations</h2>
            <ul>
                <li><b>Green row</b>: This entry has at least one match in the other dataset.</li>
                <li><b>Red row</b>: This entry has no match in the other dataset.</li>
                <li><b>match_id</b>: The <code>id</code> value of the best matching entry in the other dataset (or '—' if no match).</li>
                <li><b>Match rate</b>: (GT matched + LLM matched) / (GT entries + LLM entries)</li>
                <li>File pairs are sorted by year, extracted from the filename (e.g., <code>Patentamt_1880_sampled.csv</code> → 1880).</li>
                <li>Fuzzy matching threshold can be set with <code>--scaling</code> (default: 0.85).</li>
            </ul>
        </div>
    </section>
    '''

def make_summary_table(metrics_by_year: List[Dict]) -> str:
    table = '<table class="summary-table">\n<tr><th>Year</th><th>GT entries</th><th>LLM entries</th><th>GT matched</th><th>LLM matched</th><th>GT unmatched</th><th>LLM unmatched</th><th>Match rate (%)</th></tr>'
    for m in metrics_by_year:
        table += f'<tr><td>{m["year"]}</td><td>{m["gt_len"]}</td><td>{m["llm_len"]}</td><td>{m["gt_matched"]}</td><td>{m["llm_matched"]}</td><td>{m["gt_unmatched"]}</td><td>{m["llm_unmatched"]}</td><td>{m["match_rate"]:.1f}</td></tr>'
    table += '</table>'
    return f'<section class="summary-section"><h2>Summary Metrics by Year</h2>{table}</section>'

def make_fuzzy_year_chart(metrics_by_year: List[Dict]) -> str:
    # SVG chart: x-axis = year, y-axis = match rate
    if not metrics_by_year:
        return ''
    years = [m['year'] for m in metrics_by_year]
    rates = [m['match_rate'] for m in metrics_by_year]
    width, height = 700, 300
    margin = 60
    x0, x1 = min(years), max(years)
    y0, y1 = 0, 100
    n = len(years)
    if x1 == x0:
        x1 = x0 + 1
    points = []
    for i, (year, rate) in enumerate(zip(years, rates)):
        x = margin + (year - x0) / (x1 - x0) * (width - 2 * margin)
        y = height - margin - (rate - y0) / (y1 - y0) * (height - 2 * margin)
        points.append((x, y))
    polyline = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
    # Rotated x-tick labels, moved further down
    xtick_y = height - margin + 35
    xtick_labels = ''.join(
        f'<text x="{margin + (year - x0) / (x1 - x0) * (width - 2 * margin):.1f}" y="{xtick_y}" text-anchor="end" font-size="12" transform="rotate(90 {margin + (year - x0) / (x1 - x0) * (width - 2 * margin):.1f},{xtick_y})">{year}</text>'
        for year in years)
    yticks = list(range(0, 101, 20))
    ytick_labels = ''.join(
        f'<text x="{margin - 10}" y="{height - margin - (y - y0) / (y1 - y0) * (height - 2 * margin):.1f}" text-anchor="end" font-size="12">{y}</text>'
        for y in yticks)
    axes = f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" stroke-width="2" />' \
           f'<line x1="{margin}" y1="{height - margin}" x2="{margin}" y2="{margin}" stroke="#333" stroke-width="2" />'
    xlabel = f'<text x="{width/2}" y="{height - 10}" text-anchor="middle" font-size="14">Year</text>'
    ylabel = f'<text x="20" y="{height/2}" text-anchor="middle" font-size="14" transform="rotate(-90 20,{height/2})">Match Rate (%)</text>'
    curve = f'<polyline fill="none" stroke="#2ca02c" stroke-width="3" points="{polyline}" />'
    dots = ''.join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2ca02c" />' for x, y in points)
    svg = f'''
    <section class="chart-section">
        <h2>Fuzzy Matching: Match Rate by Year (Scaling Factor)</h2>
        <svg width="{width}" height="{height}" style="background:#fff; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.07); display:block; margin:0 auto;">
            {axes}
            {curve}
            {dots}
            {xtick_labels}
            {ytick_labels}
            {xlabel}
            {ylabel}
        </svg>
    </section>
    '''
    return svg

def make_fuzzy_scaling_chart(metrics_by_threshold: List[Tuple[float, float]]) -> str:
    width, height = 600, 300
    margin = 50
    x0, x1 = 0.0, 1.0
    y0, y1 = 0, 100
    points = []
    for t, rate in metrics_by_threshold:
        x = margin + (t - x0) / (x1 - x0) * (width - 2 * margin)
        y = height - margin - (rate - y0) / (y1 - y0) * (height - 2 * margin)
        points.append((x, y))
    polyline = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
    xticks = [round(x0 + i * 0.1, 1) for i in range(11)]
    xtick_labels = ''.join(
        f'<text x="{margin + (t - x0) / (x1 - x0) * (width - 2 * margin):.1f}" y="{height - margin + 20}" text-anchor="middle" font-size="12">{t:.1f}</text>'
        for t in xticks)
    yticks = list(range(0, 101, 20))
    ytick_labels = ''.join(
        f'<text x="{margin - 10}" y="{height - margin - (y - y0) / (y1 - y0) * (height - 2 * margin):.1f}" text-anchor="end" font-size="12">{y}</text>'
        for y in yticks)
    axes = f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" stroke-width="2" />' \
           f'<line x1="{margin}" y1="{height - margin}" x2="{margin}" y2="{margin}" stroke="#333" stroke-width="2" />'
    xlabel = f'<text x="{width/2}" y="{height - 10}" text-anchor="middle" font-size="14">Scaling Factor (Threshold)</text>'
    ylabel = f'<text x="20" y="{height/2}" text-anchor="middle" font-size="14" transform="rotate(-90 20,{height/2})">Match Rate (%)</text>'
    curve = f'<polyline fill="none" stroke="#0074d9" stroke-width="3" points="{polyline}" />'
    dots = ''.join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#0074d9" />' for x, y in points)
    svg = f'''
    <section class="chart-section">
        <h2>Fuzzy Matching: Match Rate vs. Scaling Factor</h2>
        <svg width="{width}" height="{height}" style="background:#fff; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.07); display:block; margin:0 auto;">
            {axes}
            {curve}
            {dots}
            {xtick_labels}
            {ytick_labels}
            {xlabel}
            {ylabel}
        </svg>
    </section>
    '''
    return svg

def make_intro_html() -> str:
    return '''
    <section class="intro-section">
        <div class="intro-box" style="background:#fff; border-radius:8px; max-width:900px; margin:32px auto 24px auto; padding:20px; box-shadow:0 2px 8px rgba(0,0,0,0.07);">
            <p>
                The ground truth dataset was created by student assistants. The LLM output was generated by Gemini-2.0-Flash, which is not yet the best available model (the more advanced Gemini-2.5-Flash is forthcoming). The scaling factor in fuzzy matching allows for some differences between entries and is less strict than exact matching.
            </p>
        </div>
    </section>
    '''

def make_full_html(sections: List[str], intro: str, legend: str, summary: str, chart: str, scaling_chart: str, mode: str) -> str:
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Benchmarking Results ({mode.title()} Mode)</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #f4f4f4; margin: 0; padding: 0; }}
            .intro-section {{ margin-bottom: 0; }}
            .pair-section {{ background: #fff; margin: 32px auto; padding: 24px; border-radius: 8px; max-width: 1200px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
            .table-row {{ display: flex; gap: 32px; justify-content: flex-start; }}
            .table-col {{ flex: 1; }}
            .benchmark-table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
            .benchmark-table caption {{ font-weight: bold; margin-bottom: 8px; }}
            .benchmark-table th, .benchmark-table td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
            .benchmark-table th {{ background: #e9ecef; }}
            .metrics {{ margin-bottom: 12px; font-size: 1.05em; }}
            h2 {{ margin-top: 0; }}
            .legend-section {{ margin-bottom: 24px; }}
            .summary-section {{ background: #fff; margin: 32px auto; padding: 24px; border-radius: 8px; max-width: 900px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
            .summary-table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
            .summary-table th, .summary-table td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
            .summary-table th {{ background: #e9ecef; }}
            .chart-section {{ background: #fff; margin: 32px auto; padding: 24px; border-radius: 8px; max-width: 700px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
        </style>
    </head>
    <body>
        {intro}
        {legend}
        {''.join(sections)}
        {summary}
        {chart}
        {scaling_chart}
    </body>
    </html>
    '''

def compute_metrics(gt_matches, llm_matches, gt_len, llm_len, year) -> Dict:
    gt_matched = sum(gt_matches)
    llm_matched = sum(llm_matches)
    gt_unmatched = gt_len - gt_matched
    llm_unmatched = llm_len - llm_matched
    total = gt_len + llm_len
    total_matched = gt_matched + llm_matched
    match_rate = (total_matched / total * 100) if total else 0
    return dict(year=year, gt_len=gt_len, llm_len=llm_len, gt_matched=gt_matched, llm_matched=llm_matched, gt_unmatched=gt_unmatched, llm_unmatched=llm_unmatched, match_rate=match_rate)

def main():
    parser = argparse.ArgumentParser(description='Benchmark GT and LLM datasets with strict and fuzzy matching.')
    parser.add_argument('--scaling', type=float, default=0.8, help='Scaling factor (threshold) for fuzzy matching (default: 0.8)')
    args = parser.parse_args()
    scaling = args.scaling
    gt_stems = get_file_stems(GT_DIR, 'xlsx')
    llm_stems = get_file_stems(LLM_DIR, 'csv')
    common_stems = sorted(gt_stems & llm_stems, key=lambda s: extract_year_from_stem(s))
    if not common_stems:
        print('No matching file pairs found.')
        return
    intro_html = make_intro_html()
    legend_html = make_legend_html()
    # Strict mode
    strict_sections = []
    strict_metrics_by_year = []
    for stem in common_stems:
        gt_path = GT_DIR / f'{stem}.xlsx'
        llm_path = LLM_DIR / f'{stem}.csv'
        try:
            gt_df = load_gt_file(gt_path)
            llm_df = load_llm_file(llm_path)
        except Exception as e:
            print(f'Error loading {stem}: {e}')
            continue
        year = extract_year_from_stem(stem)
        gt_matches, llm_matches, gt_match_ids, llm_match_ids = match_entries_with_ids(gt_df, llm_df, 'strict')
        section_html = make_pair_section_html(gt_df, llm_df, gt_matches, llm_matches, gt_match_ids, llm_match_ids, stem, 'strict')
        strict_sections.append(section_html)
        strict_metrics_by_year.append(compute_metrics(gt_matches, llm_matches, len(gt_df), len(llm_df), year))
    strict_summary = make_summary_table(strict_metrics_by_year)
    strict_html = make_full_html(strict_sections, intro_html, legend_html, strict_summary, '', '', 'strict')
    with open(STRICT_HTML, 'w', encoding='utf-8') as f:
        f.write(strict_html)
    print(f'Wrote {STRICT_HTML}')
    # Fuzzy mode
    fuzzy_sections = []
    fuzzy_metrics_by_year = []
    for stem in common_stems:
        gt_path = GT_DIR / f'{stem}.xlsx'
        llm_path = LLM_DIR / f'{stem}.csv'
        try:
            gt_df = load_gt_file(gt_path)
            llm_df = load_llm_file(llm_path)
        except Exception as e:
            print(f'Error loading {stem}: {e}')
            continue
        year = extract_year_from_stem(stem)
        gt_matches, llm_matches, gt_match_ids, llm_match_ids = match_entries_with_ids(gt_df, llm_df, 'fuzzy', threshold=scaling)
        section_html = make_pair_section_html(gt_df, llm_df, gt_matches, llm_matches, gt_match_ids, llm_match_ids, stem, 'fuzzy')
        fuzzy_sections.append(section_html)
        fuzzy_metrics_by_year.append(compute_metrics(gt_matches, llm_matches, len(gt_df), len(llm_df), year))
    fuzzy_summary = make_summary_table(fuzzy_metrics_by_year)
    fuzzy_year_chart = make_fuzzy_year_chart(fuzzy_metrics_by_year)
    thresholds = [round(x * 0.1, 1) for x in range(0, 11)]
    metrics_by_threshold = []
    for threshold in thresholds:
        total_matched = 0
        total = 0
        for stem in common_stems:
            gt_path = GT_DIR / f'{stem}.xlsx'
            llm_path = LLM_DIR / f'{stem}.csv'
            try:
                gt_df = load_gt_file(gt_path)
                llm_df = load_llm_file(llm_path)
            except Exception:
                continue
            gt_matches, llm_matches, _, _ = match_entries_with_ids(gt_df, llm_df, 'fuzzy', threshold=threshold)
            total_matched += sum(gt_matches) + sum(llm_matches)
            total += len(gt_df) + len(llm_df)
        match_rate = (total_matched / total * 100) if total else 0
        metrics_by_threshold.append((threshold, match_rate))
    fuzzy_scaling_chart = make_fuzzy_scaling_chart(metrics_by_threshold)
    fuzzy_html = make_full_html(fuzzy_sections, intro_html, legend_html, fuzzy_summary, fuzzy_year_chart, fuzzy_scaling_chart, 'fuzzy')
    with open(FUZZY_HTML, 'w', encoding='utf-8') as f:
        f.write(fuzzy_html)
    print(f'Wrote {FUZZY_HTML}')

if __name__ == '__main__':
    main()
