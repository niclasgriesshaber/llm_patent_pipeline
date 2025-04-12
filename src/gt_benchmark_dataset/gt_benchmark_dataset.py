import os
import random
import csv
from PyPDF2 import PdfReader, PdfWriter
from pathlib import Path

# Set the path to the patent PDFs
pdf_folder = Path("../../data/pdfs/patent_pdfs")

# Set random seed for reproducibility
random.seed(42)

# List all PDF files in sorted order
pdf_files = sorted([f for f in pdf_folder.glob("*.pdf") if f.is_file()])

# Count pages in each file
page_counts = {}
for pdf_file in pdf_files:
    try:
        reader = PdfReader(str(pdf_file))
        num_pages = len(reader.pages)
        page_counts[pdf_file.name] = num_pages
        print(f"{pdf_file.name}: {num_pages} pages")
    except Exception as e:
        print(f"Error reading {pdf_file.name}: {e}")
        continue

# Calculate total and average
total_pages = sum(page_counts.values())
average_pages = total_pages / len(page_counts) if page_counts else 0
print("\nSummary:")
print(f"Total files: {len(page_counts)}")
print(f"Total pages: {total_pages}")
print(f"Average pages per file: {average_pages:.2f}")

# Prepare output PDF and metadata CSV
output_dir = Path("../../data/pdfs/gt_pdfs")
os.makedirs(output_dir, exist_ok=True)
metadata_path = Path("gt_benchmark_dataset_metadata.csv")

with open(metadata_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "sampled_page_1", "sampled_page_2"])
    
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(str(pdf_file))
            num_pages = len(reader.pages)
            
            if num_pages < 2:
                print(f"Skipping {pdf_file.name} (only {num_pages} page{'s' if num_pages != 1 else ''})")
                continue

            sampled_pages = sorted(random.sample(range(num_pages), 2))
            print(f"Sampling {len(sampled_pages)} pages from {pdf_file.name}: {sampled_pages}")
            
            # Create a new PDF writer for each file
            file_output_pdf = PdfWriter()
            
            for page_num in sampled_pages:
                file_output_pdf.add_page(reader.pages[page_num])

            # Save individual PDF with sampled pages
            output_filename = output_dir / f"{pdf_file.stem}_sampled.pdf"
            with open(output_filename, "wb") as f_out:
                file_output_pdf.write(f_out)
                
            writer.writerow([pdf_file.name] + sampled_pages)

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
            continue

print("\nBenchmark dataset creation complete.")
print(f"Individual PDFs saved in: {output_dir}")
print(f"Metadata saved at: {metadata_path}")