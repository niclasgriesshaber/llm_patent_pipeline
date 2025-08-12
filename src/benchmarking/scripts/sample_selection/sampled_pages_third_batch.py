import os
import random
import csv
from PyPDF2 import PdfReader, PdfWriter
from pathlib import Path

# Set the path to the patent PDFs (source)
pdf_folder = Path("sample_from_pdfs")

# Set random seed for reproducibility
random.seed(42)

# Specifically target Patentamt_1878.pdf
target_pdf = pdf_folder / "Patentamt_1878.pdf"

# Prepare output PDF and metadata CSV
output_dir = Path("sampled_pdfs")
os.makedirs(output_dir, exist_ok=True)
metadata_path = Path("sampled_pages_third_batch.csv")

with open(metadata_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "sampled_page"])
    
    # Check if the target PDF exists
    if not target_pdf.exists():
        print(f"Error: {target_pdf} not found in {pdf_folder}")
        print("Available PDFs:")
        for pdf_file in sorted(pdf_folder.glob("*.pdf")):
            print(f"  - {pdf_file.name}")
        exit(1)
    
    try:
        reader = PdfReader(str(target_pdf))
        num_pages = len(reader.pages)
        
        if num_pages < 1:
            print(f"Error: {target_pdf.name} has no pages")
            exit(1)

        sampled_page = random.randint(0, num_pages - 1)
        print(f"Sampling page {sampled_page} from {target_pdf.name}")
        
        # Create a new PDF writer for the sampled page
        file_output_pdf = PdfWriter()
        file_output_pdf.add_page(reader.pages[sampled_page])

        # Save individual PDF with sampled page
        output_filename = output_dir / f"{target_pdf.stem}_sampled.pdf"
        with open(output_filename, "wb") as f_out:
            file_output_pdf.write(f_out)
            
        writer.writerow([target_pdf.name, sampled_page])

    except Exception as e:
        print(f"Error processing {target_pdf.name}: {e}")
        exit(1)

print("\nThird batch sampling complete.")
print(f"Individual PDFs saved in: {output_dir}")
print(f"Metadata saved at: {metadata_path}")
