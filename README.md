# Multimodal LLMs for Historical Dataset Construction from Archival Image Scans: German Patents (1877-1918)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An LLM-based data pipeline for constructing historical datasets from archival image scans. Originally developed for German patents (1877–1918). **Researchers can adapt this to their own image corpora using LLM-assisted coding tools** (e.g., Cursor).

## Pipeline Overview

This pipeline is tailored towards our image corpus, available at [digi.bib.uni-mannheim.de/sammlungen/patentregister](https://digi.bib.uni-mannheim.de/sammlungen/patentregister).
.

```
                                                                              ┐
                               ┌───────────────────────┐                      │
                               │   Image Corpus from   │                      │
                               │     specific Volume   │                      │
                               └───────────┬───────────┘                      │
                                           │ For each image                   │  
                                           ▼                                  │
  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐      ┌───────────────────────┐                      │
    Patent Entry               │    Gemini-2.5-Pro     │                      │
    Extraction Prompt   ─────▶ │                       │                      │
  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘      └───────────┬───────────┘                      │
                                           │                                  │
                                           ▼                                  │
                               ┌───────────────────────┐                      │
                               │  Dataset with         │                   Stage I
                               │  truncated entries    │                      │
                               └───────────┬───────────┘                      │
                                           │ For each row                     │
                                           ▼                                  │
  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐      ┌───────────────────────┐                      │
    Reparation Prompt   ─────▶ │ Gemini-2.5-Flash-Lite │                      │
  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘      └───────────┬───────────┘                      │
                                           │                                  │
                                           ▼                                  │
                               ┌───────────────────────┐                      │
                               │  Dataset with         │                      │
                               │  repaired entries     │                      │
                               └───────────────────────┘                      │
                                           │ For each row                     │
                                           ▼                                  ┘
  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐      ┌───────────────────────┐                      ┐
    Variable Extraction        │ Gemini-2.5-Flash-Lite │                      │
    Prompt              ─────▶ │                       │                   Stage II
  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘      └───────────┬───────────┘                      │
                                           │                                  │
                                           ▼                                  │
                               ┌───────────────────────┐                      │
                               │  LLM-generated        │                      │
                               │  Dataset              │                      │
                               └───────────────────────┘                      ┘
```

Dashed boxes represent carefully refined prompts (see `src/*/prompts/`). The output is an LLM-generated dataset per volume—we then merge all volumes to construct the complete dataset.

## Getting Started

1. **Download [Cursor](https://cursor.sh/)** (an AI-assisted code editor)
2. Tell the agent (`Cmd + L`) to clone this repository (`https://github.com/niclasgriesshaber/llm_patent_pipeline.git`) to your location of choice
3. Ask the agent about the pipeline and how to adapt it to your image corpus
4. If you want to use the Gemini model family, generate API keys at [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys)

## Benchmarking Data

The `data/benchmarking/` folder contains all benchmarking datasets:

- **Input Data** (`data/benchmarking/input_data/`):
  - `sampled_pdfs/` — Sampled PDF pages from each of the 41 volumes
  - `transcriptions_xlsx/perfect_transcriptions_xlsx/` — *Perfect* benchmarking datasets
  - `transcriptions_xlsx/student_transcriptions_xlsx/` — *Student-constructed* benchmarking datasets

- **Results** (`data/benchmarking/results/`):
  - `01_dataset_construction/` — *LLM-generated* outputs from Stage I Patent Entry Extractoin
  - `02_dataset_cleaning/` — *LLM-generated* outputs from Stage I Reparation
  - `03_variable_extraction/` — *LLM-generated* outputs from Stage II Variable Extraction
  - `student-constructed/` — Evaluation reports for *student-constructed* data
  - `revisions_for_VSWG_single_step/` — *LLM-generated* outputs from single-step digitization (Gemini-3.1-Pro-Preview, 1 API call per page)

Benchmarking results can be inspected visually at [historymind.ai](https://historymind.ai).

## Citation

```
@misc{griesshaber2025multimodalllmshistoricaldataset,
      title={Multimodal LLMs for Historical Dataset Construction from Archival Image Scans: German Patents (1877-1918)}, 
      author={Niclas Griesshaber and Jochen Streb},
      year={2025},
      eprint={2512.19675},
      archivePrefix={arXiv},
      primaryClass={econ.GN},
      url={https://arxiv.org/abs/2512.19675}, 
}
```

## License

MIT License—see [LICENSE](LICENSE) for details.

## Disclaimer

This repository does not endorse any product or organization. No legal or financial advice is provided.

## Contact

If you have any questions, please feel free to reach out to niclas.griesshaber@linacre.ox.ac.uk

