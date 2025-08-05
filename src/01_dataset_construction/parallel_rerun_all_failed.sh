#!/bin/bash

# Script to parallel rerun all failed pages for PDFs with error files
# This script activates the conda environment and runs the parallel Python script

echo "Starting PARALLEL automatic rerun of failed pages for all PDFs with error files..."
echo "This will process 28 PDFs that have error files using 4 parallel workers."
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm_patent_pipeline

# Run the parallel Python script
python src/01_dataset_construction/parallel_rerun_failed.py

echo ""
echo "Parallel script completed!" 