#!/usr/bin/env python3
import sys
import logging
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image

# ------------------------------------------------------------------------
# Directory / Path Setup
# ------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PDF_SRC_DIR = DATA_DIR / "pdfs" / "patent_pdfs"
CSVS_DIR = DATA_DIR / "01_dataset_construction" / "csvs"
# ------------------------------------------------------------------------

def chunkify(total_pages: int, chunk_size: int) -> list:
    """
    Yields (start_page, end_page) tuples in increments of `chunk_size`.
    Example: If total_pages=250 and chunk_size=100, yields:
      (1, 100), (101, 200), (201, 250)
    """
    for start in range(1, total_pages + 1, chunk_size):
        end = min(start + chunk_size - 1, total_pages)
        yield (start, end)

def convert_pdf_chunk(
    pdf_path: Path,
    png_dir: Path,
    start_page: int,
    end_page: int,
    dpi: int = 300
):
    """
    Converts a chunk of pages [start_page, end_page] from the given PDF into PNG.
    Saves them in the specified directory with filenames page_0001.png, etc.
    
    - dpi=300 for high quality.
    - 'optimize=True' and 'compression_level=9' for minimal PNG storage.
    """
    pages = convert_from_path(
        pdf_path.as_posix(),
        dpi=dpi,
        first_page=start_page,
        last_page=end_page,
    )
    for offset, pil_img in enumerate(pages, start=start_page):
        # Convert to RGB to avoid alpha channel (if any), which can help reduce file size
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        out_png = png_dir / f"page_{offset:04d}.png"
        # Use PNG optimization
        pil_img.save(
            out_png,
            format="PNG",
            optimize=True,
            compression_level=9
        )

def convert_pdf_to_pngs(pdf_path: Path, chunk_size: int, max_workers: int, dpi: int):
    """
    Converts a single PDF into PNG pages, using parallel chunking to speed up processing.

    :param pdf_path: The PDF file to convert
    :param chunk_size: Number of pages to process per chunk
    :param max_workers: Number of parallel processes
    :param dpi: Dots per inch (image resolution)
    """
    pdf_name = pdf_path.name
    pdf_stem = pdf_path.stem

    # Same output structure: csvs/<pdf_stem>/page_by_page/PNG
    pdf_base_out_dir = CSVS_DIR / pdf_stem
    page_by_page_dir = pdf_base_out_dir / "page_by_page"
    png_dir = page_by_page_dir / "PNG"
    png_dir.mkdir(parents=True, exist_ok=True)

    # Get total pages via pdfinfo
    try:
        info = pdfinfo_from_path(pdf_path.as_posix())
        total_pages = info.get("Pages", 0)
    except Exception as e_info:
        logging.error(f"Failed to retrieve page info from {pdf_name}: {e_info}")
        return

    if total_pages == 0:
        logging.warning(f"No pages found or pdfinfo failed for {pdf_name}. Skipping.")
        return

    logging.info(f"Converting {pdf_name} => {total_pages} page(s) at {dpi} dpi ...")

    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for (start_page, end_page) in chunkify(total_pages, chunk_size):
            tasks.append(
                executor.submit(
                    convert_pdf_chunk,
                    pdf_path,
                    png_dir,
                    start_page,
                    end_page,
                    dpi
                )
            )

        # Gather results
        completed_tasks = 0
        for fut in as_completed(tasks):
            completed_tasks += 1
            try:
                fut.result()
            except Exception as e_chunk:
                logging.error(f"Error in chunk task: {e_chunk}", exc_info=True)

    logging.info(f"Finished {pdf_name}. Total chunk tasks completed: {completed_tasks}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert PDFs to PNGs (parallel-chunked, high-quality with minimal storage)."
    )
    parser.add_argument(
        "--pdfs",
        required=True,
        nargs="+",
        help="PDF filenames in data/pdfs/patent_pdfs/, or 'all' to process all PDFs."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Pages per chunk (default=10)."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=0,
        help="Number of parallel processes (default=0)."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Image resolution in DPI (default=300). Higher => better quality but larger files."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # 1) Determine which PDFs to process
    if len(args.pdfs) == 1 and args.pdfs[0].lower() == "all":
        pdf_files = sorted(PDF_SRC_DIR.glob("*.pdf"))
        if not pdf_files:
            logging.error(f"No PDF files found in {PDF_SRC_DIR}")
            sys.exit(1)
    else:
        # Specific PDF(s)
        pdf_files = [PDF_SRC_DIR / name for name in args.pdfs]
        # Filter out invalid paths
        valid_pdfs = []
        for p in pdf_files:
            if not p.is_file():
                logging.error(f"PDF not found: {p}")
            else:
                valid_pdfs.append(p)
        pdf_files = valid_pdfs

    if not pdf_files:
        logging.error("No valid PDF files to process.")
        sys.exit(1)

    chunk_size = args.chunk_size
    max_workers = args.max_workers if args.max_workers > 0 else cpu_count()
    dpi = args.dpi

    logging.info("-" * 70)
    logging.info(f"PDF count: {len(pdf_files)} | chunk_size={chunk_size} | max_workers={max_workers} | dpi={dpi}")
    logging.info("-" * 70)

    # 2) Process each PDF
    for idx, pdf_path in enumerate(pdf_files, start=1):
        logging.info(f"[{idx}/{len(pdf_files)}] Converting {pdf_path.name}")
        convert_pdf_to_pngs(pdf_path, chunk_size, max_workers, dpi)
        logging.info("")

    logging.info("All PDF conversions completed successfully.")

if __name__ == "__main__":
    main()