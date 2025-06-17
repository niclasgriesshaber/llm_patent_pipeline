#!/usr/bin/env zsh

# Explicit list of all PDF filenames
files=(
  Patentamt_1904.pdf
  Patentamt_1905.pdf
  Patentamt_1906.pdf
  Patentamt_1907.pdf
  Patentamt_1908.pdf
  Patentamt_1909.pdf
  Patentamt_1910.pdf
  Patentamt_1911.pdf
  Patentamt_1912.pdf
  Patentamt_1913.pdf
  Patentamt_1914.pdf
  Patentamt_1915.pdf
  Patentamt_1916.pdf
  Patentamt_1917.pdf
  Patentamt_1918.pdf
)

for pdf_file in "${files[@]}"; do
  python gemini-parallel.py --pdf "$pdf_file"
done