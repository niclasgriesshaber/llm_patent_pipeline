# Multimodal LLMs for Historical Dataset Construction from Archival Image Scans

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An LLM-based data pipeline for constructing historical datasets from archival image scans. Originally developed for German patents (1877–1918). **Researchers can adapt this to their own image corpora using LLM-assisted coding tools** (e.g., Cursor).

## Pipeline Overview

This pipeline is tailored towards our image corpus, available at [digi.bib.uni-mannheim.de](https://digi.bib.uni-mannheim.de).

```
                                                                              ┐
                          ┌─────────────────────────┐                         │
                          │  Image Corpus from a    │                         │
                          │    specific Volume      │                         │
                          └───────────┬─────────────┘                         │
                                      │ For each image                        │
                                      ▼                                       │
  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐      ┌───────────────────────┐                      │
    Patent Entry                │    Gemini-2.5-Pro     │                      │
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
3. Generate API keys for the Google models at [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys)
4. Ask the agent about the pipeline and how to adapt it to your image corpus

## Citation

Will follow shortly.

## License

MIT License—see [LICENSE](LICENSE) for details.

## Disclaimer

This repository does not endorse any product or organization. No legal or financial advice is provided.

