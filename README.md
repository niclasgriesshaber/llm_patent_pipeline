# LLM Patent Pipeline: German Imperial Patent Office Dataset Construction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-orange.svg)](https://ai.google.dev/gemini)

A sophisticated Large Language Model (LLM) pipeline for extracting and structuring patent data from historical German Imperial Patent Office documents (1877-1917). This pipeline leverages Google's Gemini model series to process scanned historical documents and construct a comprehensive, structured dataset of all patents issued during this period.

## 🎯 Project Overview

This pipeline transforms historical printed patent documents into a structured, machine-readable dataset containing:
- **Patent IDs** and registration information
- **Inventor/Company names** and locations
- **Patent descriptions** and technical details
- **Registration dates** and legal codes
- **Complete metadata** for historical research

The system is designed to be **adaptable to other historical printed documents** beyond patents, making it a versatile tool for digital humanities and historical research.

## 🏗️ Pipeline Architecture

The pipeline consists of **6 sequential processing stages**, each building upon the previous:

```
PDF Documents → JSON Extraction → CSV Consolidation → Data Cleaning → Variable Extraction → Harmonization → Validation → Final Dataset
```

### Stage 1: Dataset Construction
- **Input**: Historical PDF documents
- **Process**: LLM-powered text extraction from scanned pages
- **Output**: Structured JSON arrays → Consolidated CSV files
- **Technology**: Gemini 2.5-pro with vision capabilities

### Stage 2: Dataset Cleaning
- **Purpose**: Identify and handle incomplete/truncated entries
- **Process**: LLM classification of entry completeness
- **Output**: Cleaned dataset with merged incomplete entries
- **Technology**: Gemini 2.0-flash for classification

### Stage 3: Variable Extraction
- **Purpose**: Extract structured fields from unstructured text
- **Fields**: patent_id, name, location, description, date
- **Process**: Precise information extraction using LLM
- **Output**: Structured CSV with extracted variables
- **Technology**: Gemini 2.5-flash for extraction

### Stage 4: Variable Harmonization
- **Purpose**: Standardize and normalize extracted data
- **Process**: Data cleaning, formatting, and consistency checks
- **Output**: Harmonized dataset ready for analysis

### Stage 5: Dataset Validation
- **Purpose**: Quality assurance and error detection
- **Process**: Automated validation of data integrity
- **Output**: Validation reports and error logs

### Stage 6: Dataset Merging
- **Purpose**: Consolidate all processed data
- **Process**: Merge multiple files into final dataset
- **Output**: Complete historical patent dataset

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.10**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **Google GenAI**: LLM API integration
- **PDF2Image**: Document processing
- **PIL/Pillow**: Image handling

### LLM Models
- **Gemini 2.5-pro**: High-capacity model for complex document processing
- **Gemini 2.5-flash**: Fast model for variable extraction
- **Gemini 2.0-flash**: Efficient model for classification tasks

### Data Processing
- **JSON**: Intermediate data format
- **CSV/Excel**: Final output formats
- **RapidFuzz**: Fuzzy string matching for validation
- **ThreadPoolExecutor**: Parallel processing capabilities

## 📊 Data Structure

### Input Format
- **Source**: Historical PDF documents from German Imperial Patent Office
- **Period**: 1877-1917 (40 years of patent data)
- **Language**: German (with historical characters like ſ)

### Output Format
```csv
global_id,book,book_id,page,entry,category,patent_id,name,location,description,date
1,1877_1888,1,1,"55711. COOMES, M, F., Arzt...",18,55711,"COOMES, M, F.",in Louisville,Verfahren zur Bereitung von Stahl,26. Februar 1890.
```

### Extracted Fields
- **patent_id**: Unique patent identifier
- **name**: Inventor or company name
- **location**: Geographic location of patent holder
- **description**: Technical description of the invention
- **date**: Registration date
- **category**: Patent classification code

## 🚀 Key Features

### 🔄 Parallel Processing
- Multi-threaded LLM API calls for optimal performance
- Configurable worker pools for different processing stages
- Efficient resource utilization

### 🛡️ Robust Error Handling
- Comprehensive retry mechanisms for API failures
- Rate limiting protection and backoff strategies
- Detailed error tracking and reporting

### 📈 Comprehensive Benchmarking
- Model performance comparison across Gemini variants
- Prompt engineering optimization framework
- Detailed evaluation metrics and visualizations

### 🎯 Quality Assurance
- Multi-stage validation pipeline
- Automated error detection and reporting
- Data consistency checks and integrity validation

### 📊 Advanced Analytics
- Threshold sensitivity analysis for matching algorithms
- Detailed performance metrics and visualizations
- Comprehensive HTML reporting system

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.10+
- Google GenAI API key
- Conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/niclasgriesshaber/llm_patent_pipeline.git
   cd llm_patent_pipeline
   ```

2. **Set up the environment**
   ```bash
   conda env create -f config/environment.yml
   conda activate llm_patent_pipeline
   ```

3. **Configure API access**
   ```bash
   cp config/.env.example config/.env
   # Edit config/.env and add your Google GenAI API key
   ```

4. **Prepare your data**
   ```bash
   # Place your PDF documents in data/pdfs/patent_pdfs/
   ```

### Basic Usage

1. **Run the complete pipeline**
   ```bash
   # Stage 1: Dataset Construction
   python src/01_dataset_construction/gemini-2.5-parallel.py --pdf your_document.pdf
   
   # Stage 2: Dataset Cleaning
   python src/02_dataset_cleaning/complete_patent.py --csv your_constructed_data.csv
   
   # Stage 3: Variable Extraction
   python src/03_variable_extraction/variable_extraction.py --csv your_cleaned_data.csv
   ```

2. **Run benchmarking**
   ```bash
   cd src/benchmarking/scripts
   python 01_dataset_construction_benchmarking.py --model gemini-2.5-pro --prompt construction_v0.4_prompt.txt
   ```

## 📋 Detailed Usage Guide

### Stage-by-Stage Processing

#### 1. Dataset Construction
```bash
python src/01_dataset_construction/gemini-2.5-parallel.py \
  --pdf Patentamt_1877.pdf \
  --temperature 0.0 \
  --thinking_budget 32768
```

#### 2. Dataset Cleaning
```bash
python src/02_dataset_cleaning/complete_patent.py \
  --csv Patentamt_1877_constructed.csv
```

#### 3. Variable Extraction
```bash
python src/03_variable_extraction/variable_extraction.py \
  --csv Patentamt_1877_cleaned.csv \
  --temperature 0.0
```

### Benchmarking and Evaluation

The pipeline includes a comprehensive benchmarking framework:

```bash
# Dataset construction benchmarking
python src/benchmarking/scripts/01_dataset_construction_benchmarking.py \
  --model gemini-2.5-pro \
  --prompt construction_v0.4_prompt.txt

# Dataset cleaning benchmarking
python src/benchmarking/scripts/02_dataset_cleaning_benchmarking.py \
  --dataset_construction_model gemini-2.5-pro \
  --dataset_construction_prompt construction_v0.4_prompt.txt \
  --model gemini-2.0-flash \
  --prompt cleaning_v0.0_prompt.txt

# Variable extraction benchmarking
python src/benchmarking/scripts/03_variable_extraction_benchmarking.py \
  --dataset_cleaning_model gemini-2.0-flash \
  --dataset_cleaning_prompt cleaning_v0.0_prompt.txt \
  --model gemini-2.5-flash \
  --prompt variable_extraction_v0.0_prompt.txt
```

### Configuration Options

- **Temperature**: Control randomness in LLM responses (0.0 for deterministic)
- **Thinking Budget**: Allocate computational resources for complex reasoning
- **Max Workers**: Configure parallel processing capacity
- **Retry Attempts**: Set error recovery strategies

## 📊 Benchmarking Framework

### Model Comparison
The pipeline supports comprehensive benchmarking across:
- **Gemini 2.0-flash**: Fast, efficient processing
- **Gemini 2.5-flash**: Balanced performance and accuracy
- **Gemini 2.5-pro**: Highest accuracy for complex tasks

### Evaluation Metrics
- **Character Error Rate (CER)**: Text extraction accuracy
- **Match Rates**: Variable extraction precision
- **Processing Speed**: Time and cost efficiency
- **Completeness**: Data coverage analysis

### Visualization
- **HTML Reports**: Interactive benchmarking results
- **Threshold Analysis**: Sensitivity analysis for matching algorithms
- **Performance Dashboards**: Comprehensive evaluation overview

## 🔧 Advanced Configuration

### Environment Variables
```bash
GOOGLE_API_KEY=your_api_key_here
MAX_WORKERS=20
MAX_RETRIES=3
TEMPERATURE=0.0
```

### Custom Prompts
The pipeline uses configurable prompts for each stage:
- `src/benchmarking/prompts/01_dataset_construction/`
- `src/benchmarking/prompts/02_dataset_cleaning/`
- `src/benchmarking/prompts/03_variable_extraction/`

### Performance Optimization
- **Parallel Processing**: Configure worker pools for optimal throughput
- **Rate Limiting**: Built-in protection against API limits
- **Memory Management**: Efficient handling of large datasets
- **Error Recovery**: Robust retry mechanisms

## 🐛 Troubleshooting

### Common Issues

1. **API Rate Limits**
   ```bash
   # The pipeline automatically handles rate limiting
   # Check logs for retry attempts and backoff strategies
   ```

2. **Memory Issues**
   ```bash
   # Reduce MAX_WORKERS for large documents
   # Process documents in smaller batches
   ```

3. **JSON Parsing Errors**
   ```bash
   # Check prompt engineering for better LLM responses
   # Verify input document quality
   ```

### Debug Mode
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
python your_script.py
```

### Error Reports
The pipeline generates detailed error reports:
- Failed page analysis
- API error tracking
- Processing statistics
- Quality metrics

## 🔬 Research Applications

### Historical Research
- **Patent Analysis**: Study innovation patterns in Imperial Germany
- **Economic History**: Analyze industrial development and technology transfer
- **Social History**: Examine inventor demographics and geographic distribution

### Digital Humanities
- **Text Mining**: Large-scale analysis of historical documents
- **Network Analysis**: Study connections between inventors and companies
- **Geographic Analysis**: Map innovation centers and technology diffusion

### Machine Learning
- **Training Data**: High-quality labeled datasets for NLP research
- **Model Evaluation**: Benchmark LLM performance on historical documents
- **Information Extraction**: Develop specialized extraction models

## 🤝 Contributing

We welcome contributions to improve the pipeline:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests if applicable**
5. **Submit a pull request**

### Development Guidelines
- Follow Python PEP 8 style guidelines
- Add comprehensive docstrings
- Include error handling for robustness
- Test with sample data before submitting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{llm_patent_pipeline,
  title={LLM Patent Pipeline: German Imperial Patent Office Dataset Construction},
  author={Niclas Griesshaber},
  year={2024},
  url={https://github.com/niclasgriesshaber/llm_patent_pipeline}
}
```

## 🙏 Acknowledgments

- **Google Gemini Team**: For providing the LLM capabilities
- **Historical Research Community**: For inspiration and use cases
- **Open Source Community**: For the tools and libraries that make this possible

## 📞 Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/niclasgriesshaber/llm_patent_pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/niclasgriesshaber/llm_patent_pipeline/discussions)

---

**Built with ❤️ for historical research and digital humanities** 