import os
import random
import csv
from PyPDF2 import PdfReader, PdfWriter
from pathlib import Path

# Set the path to the patent PDFs (source)
pdf_folder = Path("sample_from_pdfs")

# Set random seed for reproducibility
random.seed(42)

# List all PDF files in sorted order
pdf_files = sorted([f for f in pdf_folder.glob("*.pdf") if f.is_file()])

# Prepare output PDF and metadata CSV
output_dir = Path("sampled_pdfs")
os.makedirs(output_dir, exist_ok=True)
metadata_path = Path("sampled_pages_second_batch.csv")

with open(metadata_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "sampled_page"])
    
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(str(pdf_file))
            num_pages = len(reader.pages)
            
            if num_pages < 1:
                print(f"Skipping {pdf_file.name} (no pages)")
                continue

            sampled_page = random.randint(0, num_pages - 1)
            print(f"Sampling page {sampled_page} from {pdf_file.name}")
            
            # Create a new PDF writer for the sampled page
            file_output_pdf = PdfWriter()
            file_output_pdf.add_page(reader.pages[sampled_page])

            # Save individual PDF with sampled page
            output_filename = output_dir / f"{pdf_file.stem}_sampled.pdf"
            with open(output_filename, "wb") as f_out:
                file_output_pdf.write(f_out)
                
            writer.writerow([pdf_file.name, sampled_page])

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
            continue

print("\nSecond batch sampling complete.")
print(f"Individual PDFs saved in: {output_dir}")
print(f"Metadata saved at: {metadata_path}") 