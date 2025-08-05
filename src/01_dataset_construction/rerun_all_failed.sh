#!/bin/bash

# Script to rerun all failed pages for PDFs with error files
# This script activates the conda environment and runs the Python script

echo "Starting automatic rerun of failed pages for all PDFs with error files..."
echo "This will process 28 PDFs that have error files."
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm_patent_pipeline

# Run the Python script
python src/01_dataset_construction/rerun_failed_pages.py

echo ""
echo "Script completed!" 