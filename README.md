# Multimodal LLMs for Historical Dataset Construction from Archival Image Scans

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An LLM-based data pipeline for constructing historical datasets from archival image scans. Originally developed for German patents (1877–1918). **Researchers can adapt this to their own image corpora using LLM-assisted coding tools** (e.g., Cursor).

## Pipeline Overview

```
                                                                           ┐
    ┌─────────────────────────┐                                            │
    │  Image Corpus from a    │                                            │
    │    specific Volume      │                                            │
    └───────────┬─────────────┘                                            │
                │ For each image                                           │
                ▼                                                          │
    ┌───────────────────────┐      ┌─────────────────────────┐             │
    │    Gemini-2.5-Pro     │◄─────│  Patent Entry           │             │
    │                       │      │  Extraction Prompt      │             │
    └───────────┬───────────┘      └─────────────────────────┘             │
                │                                                          │
                ▼                                                          │
    ┌───────────────────────┐                                              │
    │  Dataset with         │                                              │
    │  truncated entries    │                                           Stage I
    └───────────┬───────────┘                                              │
                │ For each row                                             │
                ▼                                                          │
    ┌───────────────────────┐      ┌─────────────────────────┐             │
    │ Gemini-2.5-Flash-Lite │◄─────│  Reparation Prompt      │             │
    │                       │      │                         │             │
    └───────────┬───────────┘      └─────────────────────────┘             │
                │                                                          │
                ▼                                                          │
    ┌───────────────────────┐                                              │
    │  Dataset with         │                                              │
    │  repaired entries     │                                              │
    └───────────┬───────────┘                                              ┘
                │ For each row                                             ┐
                ▼                                                          │
    ┌───────────────────────┐      ┌─────────────────────────┐             │
    │ Gemini-2.5-Flash-Lite │◄─────│  Variable Extraction    │             │
    │                       │      │  Prompt                 │          Stage II
    └───────────┬───────────┘      └─────────────────────────┘             │
                │                                                          │
                ▼                                                          │
    ┌───────────────────────┐                                              │
    │  LLM-generated        │                                              │
    │  Dataset              │                                              │
    └───────────────────────┘                                              ┘
```

Dashed boxes represent carefully refined prompts (see `src/*/prompts/`). The output is an LLM-generated dataset per volume—we then merge all volumes to construct the complete dataset.

## Getting Started

1. **Download [Cursor](https://cursor.sh/)** (an AI-assisted code editor)
2. Tell the agent (`Cmd + L`) to clone this repository (`https://github.com/niclasgriesshaber/llm_patent_pipeline.git`) to your location of choice
3. Ask the agent about the pipeline and how to adapt it to your image corpus

## Citation

Will follow shortly.

## License

MIT License—see [LICENSE](LICENSE) for details.

## Disclaimer

This repository does not endorse any product or organization. No legal or financial advice is provided.

