#!/usr/bin/env zsh

# Explicit list of all PDF filenames
files=(
  Patentamt_1892_sampled.pdf
  Patentamt_1893_sampled.pdf
  Patentamt_1894_sampled.pdf
  Patentamt_1895_sampled.pdf
  Patentamt_1901_sampled.pdf
  Patentamt_1902_sampled.pdf
  Patentamt_1903_sampled.pdf
  Patentamt_1905_sampled.pdf
  Patentamt_1910_sampled.pdf
  Patentamt_1911_sampled.pdf
  Patentamt_1912_sampled.pdf
  Patentamt_1913_sampled.pdf
  Patentamt_1914_sampled.pdf
  Patentamt_1915_sampled.pdf
  Patentamt_1916_sampled.pdf
  Patentamt_1917_sampled.pdf
  Patentamt_1918_sampled.pdf
)

for pdf_file in "${files[@]}"; do
  python gemini-2.0-parallel.py --pdf "$pdf_file"
done