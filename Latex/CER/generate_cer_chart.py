#!/usr/bin/env python3
"""
Generate academic-quality CER chart for LaTeX/Overleaf documents.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from the benchmarking results
years = [1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918]

llm_cers = [0.0013265531726730046, 0.009618629767127912, 0.002851033499643621, 0.004447607187333215, 0.0023932253313696614, 0.008262755629002272, 0.005350081414282391, 0.004852227613586237, 0.0028875654214040787, 0.018806875631951468, 0.005125486037469071, 0.0033247137052087182, 0.008455034588777863, 0.0025945144551519643, 0.003945885005636978, 0.0026301946344029457, 0.026758147512864493, 0.02611150317572336, 0.0304390243902439, 0.02679528403001072, 0.031050688150386037, 0.03745630098218745, 0.031141868512110725, 0.02927098674521355, 0.02509590792838875, 0.018037398643058082, 0.025016567263088138, 0.010576414595452142, 0.0351668169522092, 0.031660983925962005, 0.05493217216546642, 0.027963418923672177, 0.029281277728482696, 0.024323398424117848, 0.0339943342776204, 0.025789572042608858, 0.02566267094377849, 0.04123927150931547, 0.03083989501312336, 0.03219034289713086, 0.030932594644506]

student_cers = [0.0035374751271280125, 0.030411145090044174, 0.005879208979155532, 0.006588319088319089, 0.007733382434174185, 0.0167320801487296, 0.004654410053525715, 0.008160564622849581, 0.011197399313707784, 0.012239902080783354, 0.017497348886532343, 0.005544261689151728, 0.04650269023827825, 0.012421208750463477, 0.0033821871476888386, 0.007894736842105263, 0.013397457918241155, 0.012365306482953541, 0.09293244826239751, 0.046157148990983256, 0.02836522322927157, 0.047111703013151325, 0.00461361014994233, 0.025461254612546124, 0.025874460948730235, 0.018037398643058082, 0.005302402651201326, 0.026060926219404826, 0.04697380307136405, 0.05473204104903079, 0.059788980070339975, 0.035022879267863426, 0.11073646850044365, 0.023124357656731757, 0.024787535410764873, 0.02177177177177177, 0.0340793489318413, 0.0249371332774518, 0.044664466446644666, 0.04526266416510319, 0.0399814039981404]

# Convert to percentages
llm_cers_pct = [x * 100 for x in llm_cers]
student_cers_pct = [x * 100 for x in student_cers]

# Academic-style plot configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 4.5),
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

# Plot lines with markers
ax.plot(years, llm_cers_pct, 
        marker='o', markersize=4, linewidth=1.5,
        color='#1f77b4', label='LLM (Gemini-2.5-Pro)')

ax.plot(years, student_cers_pct, 
        marker='s', markersize=4, linewidth=1.5,
        color='#d62728', label='Research Assistants')

# Vertical lines for font changes
ax.axvline(x=1893.5, color='#666666', linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(x=1911.5, color='#666666', linestyle='--', linewidth=1, alpha=0.7)

# Axis labels
ax.set_xlabel('Year')
ax.set_ylabel('Character Error Rate (%)')

# Set axis limits
ax.set_xlim(1877, 1919)
ax.set_ylim(0, 12)

# Font period labels aligned at y=8 (on top of the grid line)
label_y = 8.2  # Just above the y=8 grid line

# Roman Font label (centered between 1878 and 1894)
ax.text(1886, label_y, 'Roman Font', fontsize=9, color='#444444', ha='center', va='bottom', 
        style='italic')

# Unger Gothic label (centered between 1894 and 1912)
ax.text(1903, label_y, 'Unger Gothic', fontsize=9, color='#444444', ha='center', va='bottom',
        style='italic')

# Breitkopf Gothic label (centered between 1912 and 1918, shifted right)
ax.text(1916, label_y, 'Breitkopf Gothic', fontsize=9, color='#444444', ha='center', va='bottom',
        style='italic')

# X-axis ticks every 5 years
ax.set_xticks(range(1880, 1920, 5))

# Y-axis ticks
ax.set_yticks(range(0, 13, 2))

# Legend
ax.legend(loc='upper left', framealpha=0.95, edgecolor='#cccccc')

# Remove top and right spines for cleaner academic look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Tight layout
plt.tight_layout()

# Save as PNG for Overleaf
output_path = '/Users/niclasgriesshaber/Desktop/workspace/Academia/DPhil Oxford/Dissertation/Mannheim Fellow/Papers/imperial-germany-dataset/Code/llm_patent_pipeline/Latex/CER/cer_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print(f"Chart saved to: {output_path}")
