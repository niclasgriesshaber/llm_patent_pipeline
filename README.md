# Multimodal LLMs for Dataset Construction from Image Scans: German Patents (1877-1918)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-orange.svg)](https://ai.google.dev/gemini)

**Transforming archival image scans into structured datasets using multimodal LLMs required.**

This repository contains a complete, production-ready pipeline that uses multimodal Large Language Models (specifically Google's Gemini) to extract structured data directly from scanned historical documents. Developed for economic history research, it converts 41 years of German Imperial Patent Office registers (1878-1918, ~240,000 patents) from scanned PDFs into a comprehensive structured dataset.

## 🎯 What This Does (For Economic Historians)

**The Challenge:** Historical economic data often exists only as scanned documents—patent registers, company directories, trade statistics, notarial records. Converting these into analyzable datasets traditionally requires either:
- Manual data entry (expensive, time-consuming, error-prone)
- OCR + manual correction (still labor-intensive, struggles with historical typography)
- Specialized handwriting recognition tools (limited accuracy, expensive)

**The Solution:** This pipeline uses **multimodal LLMs** (AI models that can "see" and understand images) to:
1. **Read scanned document pages directly** (no separate OCR step)
2. **Extract structured information** following your rules and categories
3. **Handle complex layouts** (multi-column text, tables, historical fonts like Fraktur)
4. **Process documents at scale** (parallel processing, thousands of pages)
5. **Achieve high accuracy** (benchmarked with detailed metrics)

**Example Use Cases for Economic Historians:**
- Converting historical patent registers into inventor/innovation databases
- Digitizing historical firm directories into company datasets
- Extracting trade statistics from printed commerce reports
- Processing notarial records into transaction databases
- Building datasets from historical newspapers, yearbooks, or government publications

## 🏗️ How It Works: Pipeline Architecture

The pipeline processes documents through **5 sequential stages**, each automatically building on the previous one:

```
PDF Scans → Image Pages → Structured Extraction → Data Cleaning → Variable Parsing → Validation → Analysis-Ready Dataset
```

### Stage 1: Dataset Construction (Image → Structured Text)
- **Input**: Scanned PDF pages (historical documents with complex layouts)
- **Process**: Multimodal LLM reads images and extracts information following detailed prompt instructions
- **Output**: Raw structured data (JSON → CSV), one row per entry
- **Why this works**: Modern multimodal LLMs can understand document layout, read historical fonts (including Fraktur, ſ character), and follow complex extraction rules—all from images alone

### Stage 2: Dataset Cleaning (Handling Page Breaks)
- **Problem**: Historical documents have entries that span multiple pages
- **Process**: LLM identifies incomplete entries and intelligently merges fragments
- **Output**: Complete, continuous patent entries
- **Example**: "55711. COOMES, M, F., Arzt..." (truncated at page bottom) + "...in Louisville" (continued on next page) → merged single entry

### Stage 3: Variable Extraction (Unstructured → Structured)
- **Input**: Complete patent entries as free text
- **Process**: LLM extracts specific fields (patent_id, inventor name, location, description, date)
- **Output**: Structured dataset with separate columns for each variable
- **Why needed**: Historical documents don't have clear field delimiters; LLM understands context

### Stage 4: Dataset Validation
- **Purpose**: Quality assurance via comparison with ground truth samples
- **Process**: Automated metrics (Character Error Rate, field match rates)
- **Output**: Validation reports, error identification, quality metrics

### Stage 5: Dataset Merging
- **Purpose**: Consolidate multi-year data into final research dataset
- **Output**: Complete historical dataset ready for economic analysis

## 💡 Why This Matters for Economic History Research

### Cost-Effective
- **API costs**: ~$50-100 per 10,000 patent entries (including all stages)
- **Traditional alternative**: $10,000+ for manual transcription or OCR correction
- **Time savings**: Hours instead of months for large document collections

### High Accuracy
- **Character Error Rate**: < 2% on historical German documents (Fraktur font)
- **Field extraction**: > 95% accuracy on structured variables
- **Benchmarked**: Comprehensive evaluation framework included

### Scalable & Adaptable
- **Volume**: Processes thousands of pages automatically (parallel processing)
- **Flexibility**: Customize for your document types with prompt engineering
- **Languages**: Works with any language/script the LLM can read (German, English, Latin, etc.)

### Future-Proof
- **Improving over time**: As LLMs improve, your pipeline gets better automatically
- **Version control**: Lock specific model versions for reproducibility
- **Open source**: MIT license, modify freely for your research

## 🛠️ Technical Stack

**Core Technologies:**
- **Python 3.10**: Easy to read, widely used in economics research
- **Google Gemini API**: Multimodal LLM (pay-per-use, no upfront costs)
- **Pandas**: Standard data analysis library
- **Standard libraries**: No exotic dependencies, easy to install

**LLM Models Used:**
- **Gemini 2.5-pro**: Most accurate, used for initial extraction (~$0.005/page)
- **Gemini 2.0-flash**: Fast and cheap, used for cleaning (~$0.0001/page)
- **Gemini 2.5-flash**: Balanced, used for variable extraction (~$0.0002/page)

**Note**: You don't need deep programming knowledge—see "Working with AI-Assisted Coding Tools" below.

## 📊 Example: What You Get

### Input
Scanned PDF pages like this:
- Multi-column layout with category headings
- Historical German typography (Fraktur font, ſ character)
- Patent entries with varying formats
- Entries spanning page breaks

### Output
Clean, structured CSV ready for analysis:
```csv
patent_id,name,location,description,date,category,page
55711,"COOMES, M, F.",in Louisville,Verfahren zur Bereitung von Stahl,26. Februar 1890,18,45
56181,"VERSEN, B.",in Dortmund,Verfahren und Vorrichtung zur Herstellung von Bessemer-Birnen-Böden,15. Mai 1890,18,45
```

### Key Variables Extracted
- **patent_id**: Unique identifier for linking to other datasets
- **name**: Inventor/company name (for studying actors)
- **location**: Geographic location (for spatial analysis)
- **description**: Technical description (for innovation content analysis)
- **date**: Registration date (for time series analysis)
- **category**: Patent classification (for technology sector analysis)

**For this project**: 41 years (1878-1918), ~240,000 patent entries, fully structured and validated.

## 🚀 Key Features

### 📖 No OCR Required
- Multimodal LLMs read document images directly
- Handles historical fonts, handwriting, and complex layouts
- Higher accuracy than traditional OCR pipelines

### ⚡ Parallel Processing at Scale
- Process hundreds/thousands of pages automatically
- Configurable parallelization (20+ concurrent API calls)
- Built-in rate limiting and error recovery

### 📊 Comprehensive Benchmarking
- Compare different LLM models and prompts systematically
- Detailed accuracy metrics (Character Error Rate, field-level precision)
- HTML reports with visualizations for evaluation

### 🎯 Quality Assurance Built-In
- Automated validation against ground truth samples
- Error tracking and reporting at every stage
- Data consistency checks across processing steps

### 🔧 Easily Customizable
- **Prompt engineering**: Modify extraction rules without coding
- **Pipeline configuration**: Adjust models, parameters, output formats
- **Works with AI assistants**: Adapt to your documents using Cursor, GitHub Copilot, or similar tools (see below)

## 🏃‍♂️ Quick Start for Economic Historians

### What You Need
1. **Basic computing setup**: Mac/Windows/Linux with Python 3.10+
2. **Google Gemini API key**: [Get free API key here](https://ai.google.dev/) (~$300 free credits for new users)
3. **Your documents**: Scanned PDFs of historical documents
4. **Optional but recommended**: Cursor or VS Code (AI-assisted coding tools—see below)

### Installation (5 minutes)

1. **Clone this repository**
   ```bash
   git clone https://github.com/niclasgriesshaber/llm_patent_pipeline.git
   cd llm_patent_pipeline
   ```

2. **Install dependencies**
   ```bash
   # Using conda (recommended)
   conda env create -f config/environment.yml
   conda activate llm_patent_pipeline
   
   # Or using pip
   pip install -r config/requirements.txt
   ```

3. **Add your API key**
   ```bash
   # Create config/.env file and add:
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Test with sample data**
   ```bash
   # Place a PDF in data/pdfs/patent_pdfs/
   python src/01_dataset_construction/gemini-2.5-parallel.py --pdf your_file.pdf
   ```

### Running the Full Pipeline

**For a single document:**
   ```bash
# Stage 1: Extract structured data from scans
python src/01_dataset_construction/gemini-2.5-parallel.py --pdf Patentamt_1878.pdf

# Stage 2: Clean and merge incomplete entries
python src/02_dataset_cleaning/run_all_cleaning_sequential.py

# Stage 3: Extract structured variables
python src/03_variable_extraction/run_all_variables_sequential.py

# Stage 4: Validate results
python src/04_dataset_validation/01_validated.py
```

**For multiple documents:**
   ```bash
# Process all PDFs in data/pdfs/patent_pdfs/
bash src/01_dataset_construction/run_all_patents.sh
# Then run stages 2-4 as above
```

### Understanding Costs
- **Small test** (100 pages): ~$0.50
- **Medium project** (1,000 pages): ~$5
- **Large project** (10,000 pages): ~$50
- **This full pipeline** (41 PDFs, ~10,000 pages): ~$50-100 total

Compare this to: Manual transcription ($1-5 per page = $10,000-50,000) or research assistant time (hundreds of hours).

## 🤖 Working with AI-Assisted Coding Tools (Recommended for Non-Programmers)

**You don't need to be a programmer to adapt this pipeline to your documents.** Modern AI-assisted coding tools like Cursor and GitHub Copilot can help you modify the code through natural language instructions.

### Why Use AI-Assisted Tools?

Traditional approach (manual coding):
- Learn Python syntax, libraries, error handling
- Debug API integration issues
- Figure out how to modify prompts and processing logic
- **Time investment**: Weeks to months

AI-assisted approach:
- Describe what you want in plain language
- AI suggests code changes, you review and approve
- Iterate quickly with immediate feedback
- **Time investment**: Hours to days

### Recommended: Cursor IDE

[Cursor](https://cursor.sh/) is a code editor (fork of VS Code) with powerful AI assistance built-in. It's particularly good for this type of project.

**Setup (5 minutes):**
1. Download and install [Cursor](https://cursor.sh/)
2. Open this repository folder in Cursor: `File → Open Folder`
3. Configure your Cursor API key (uses Claude or GPT-4)
4. Start asking questions in the chat panel

**Example prompts to try in Cursor:**

```
"Explain how the dataset construction stage works"

"I have company directories instead of patents. The entries have: 
company name, address, business type, founding year. 
How do I modify the extraction prompt?"

"My documents are in French, not German. What needs to change?"

"The validation shows 5% errors. How can I improve accuracy?"

"Add a new field 'patent_value' to the extraction stage"

"My PDFs have 3 columns instead of 2. Update the prompt to handle this"

"Create a summary report showing: total entries, entries per category, 
and processing time"
```

### Step-by-Step: Adapting This Pipeline to Your Documents

**Example: Converting Company Directories (1880-1920) to Dataset**

1. **Understand the structure** (with Cursor's help)
   ```
   Cursor prompt: "Walk me through the pipeline. I want to process 
   historical company directories instead of patents. What are the 
   main changes I need to make?"
   ```

2. **Modify the extraction prompt**
   - Open `src/01_dataset_construction/prompts/prompt.txt`
   - Ask Cursor: "Help me adapt this prompt for company directories. 
     My documents have: company name, address, business type, 
     founding year, and capital."
   - Review Cursor's suggestions, iterate until it matches your documents

3. **Adjust variable extraction**
   - Open `src/03_variable_extraction/prompts/prompt.txt`
   - Ask Cursor: "Update this to extract: company_name, street, city, 
     business_type, founding_year, capital_amount"
   - Test on a few pages, refine as needed

4. **Test on sample pages**
   ```
   Cursor prompt: "Help me run this on just pages 1-10 of my PDF 
   to test before processing everything"
   ```

5. **Validate and iterate**
   - Check the output CSV
   - Ask Cursor: "The city names are being incorrectly merged with 
     street addresses. How do I fix this?"
   - Refine prompts based on errors

6. **Scale to full dataset**
   ```
   Cursor prompt: "Everything looks good. Help me process all 
   50 PDFs in parallel"
   ```

### Alternative: GitHub Copilot in VS Code

If you prefer VS Code:
1. Install [VS Code](https://code.visualstudio.com/)
2. Add the [GitHub Copilot extension](https://github.com/features/copilot)
3. Open this repository
4. Use Copilot Chat for similar conversational assistance

### What AI Tools Can Help You With

✅ **Works great:**
- Modifying prompts for different document types
- Adjusting extraction rules and output formats
- Debugging errors and improving accuracy
- Adding new features (additional fields, processing steps)
- Understanding how the code works
- Creating visualizations and summary statistics

⚠️ **Requires your judgment:**
- Evaluating data quality (AI can't judge historical accuracy)
- Deciding which errors matter for your research
- Determining if extracted information is correct
- Choosing appropriate models and parameters

### Tips for Effective AI-Assisted Adaptation

1. **Start small**: Test on 10-50 pages before processing thousands
2. **Be specific**: "Extract company names" → "Extract company names, which appear after entry numbers and before addresses"
3. **Iterate**: Perfect prompts require 3-5 rounds of refinement
4. **Validate**: Always check output manually on sample data
5. **Document changes**: Keep notes on what prompts/parameters work best

### Learning Resources

- **Cursor documentation**: [docs.cursor.sh](https://docs.cursor.sh/)
- **Prompt engineering for historical documents**: See `src/benchmarking/prompts/` for examples
- **This project's structure**: Ask Cursor to explain any file

## 📋 Adapting to Your Documents: Key Modification Points

### 1. Extraction Prompts (Most Important)

**File**: `src/01_dataset_construction/prompts/prompt.txt`

This prompt tells the LLM how to read your documents. You'll need to modify:
- **Document structure**: "two-column layout" → your layout
- **Entry format**: How individual records appear
- **What to extract**: Categories, IDs, text blocks
- **What to ignore**: Headers, footers, page numbers

**Example modification** (for company directories):
```
Original: "Extract patent entries from two-column German Patent Office pages"
Modified: "Extract company entries from single-column trade directories"

Original: "Entry starts with patent ID number (e.g., 55711.)"
Modified: "Entry starts with company name in bold, followed by address"
```

### 2. Variable Extraction Prompts

**File**: `src/03_variable_extraction/prompts/prompt.txt`

This prompt tells the LLM which structured fields to extract from unstructured text.

**For patents**: patent_id, name, location, description, date  
**For companies**: company_name, street, city, business_type, founding_year  
**For trade records**: commodity, quantity, origin, destination, value, date

### 3. Pipeline Configuration

**Files**: Each script has configurable parameters at the top

Key parameters to adjust:
- `MODEL_NAME`: "gemini-2.5-pro" (accurate) vs "gemini-2.5-flash" (cheaper)
- `MAX_WORKERS`: Parallel processing (higher = faster but more $)
- `TEMPERATURE`: 0.0 (deterministic) vs 0.5+ (more creative)
- `MAX_OUTPUT_TOKENS`: Increase if entries are very long

### 4. Benchmarking (Optional but Recommended)

Test different models and prompts on a sample before processing everything:

```bash
# Compare gemini-2.5-pro vs gemini-2.5-flash on your documents
python src/benchmarking/scripts/01_dataset_construction_benchmarking.py \
  --model gemini-2.5-pro \
  --prompt your_custom_prompt.txt

# Generates HTML report with accuracy metrics and cost comparison
```

## ⚙️ Advanced Configuration

### Parallel Processing
Control speed vs cost tradeoff:
```python
# In gemini-2.5-parallel.py
MAX_WORKERS = 20  # Process 20 pages simultaneously
# Higher = faster but more API costs (rate limits apply)
```

### Model Selection
Choose based on accuracy needs:
- **gemini-2.5-pro**: Highest accuracy, ~10x cost (~$0.005/page)
- **gemini-2.5-flash**: Balanced, ~2x cost (~$0.0002/page)
- **gemini-2.0-flash**: Fastest/cheapest (~$0.0001/page), lower accuracy

### Thinking Budget (Advanced)
For complex documents with tricky layouts:
```python
THINKING_BUDGET = 32768  # More "thinking tokens" = better reasoning
# Use higher values (24576-32768) for difficult documents
# Use lower values (8192-15000) for simple, regular layouts
```

## 📊 Evaluating Accuracy: Benchmarking Framework

Before processing thousands of pages, you should test accuracy on a sample. This project includes a comprehensive benchmarking system.

### How to Benchmark

1. **Prepare ground truth**: Manually transcribe 50-100 sample entries
2. **Run benchmark**: Test different models/prompts against ground truth
3. **Compare results**: Get detailed accuracy metrics and cost estimates
4. **Choose best approach**: Balance accuracy, speed, and cost

```bash
# Example: Test extraction accuracy
python src/benchmarking/scripts/01_dataset_construction_benchmarking.py \
  --model gemini-2.5-pro \
  --prompt construction_v0.4_prompt.txt

# Output: HTML report with metrics
```

### Key Metrics

**Character Error Rate (CER)**
- Measures transcription accuracy character-by-character
- < 2% = Excellent (better than most OCR)
- 2-5% = Good (acceptable for most research)
- > 5% = Needs improvement (refine prompts or use stronger model)

**Field Match Rates**
- How often extracted variables exactly match ground truth
- Patent ID: Expect > 99% (critical field)
- Names: Expect > 95% (some variation acceptable)
- Dates: Expect > 95% (structured format helps)
- Descriptions: 80-90% (longest/most complex field)

**Cost-Accuracy Tradeoff**
- The benchmarking reports show: accuracy vs cost per page
- Example: gemini-2.5-pro (98% accuracy, $0.005/page) vs gemini-2.5-flash (95% accuracy, $0.0002/page)
- For 10,000 pages: $50 vs $2 → You decide if 3% accuracy improvement is worth $48

### Benchmarking Output

HTML reports include:
- Side-by-side comparison (ground truth vs extracted)
- Error analysis (which fields have issues)
- Cost projections for full dataset
- Recommendations for improvement

## 🐛 Common Issues & Solutions

### "API Rate Limit Exceeded"
**Problem**: Google enforces rate limits on API calls  
**Solution**: The pipeline automatically retries with backoff. If persistent, reduce `MAX_WORKERS` from 20 to 10 or 5.

### "JSON Parsing Error"
**Problem**: LLM output doesn't match expected JSON format  
**Solution**: 
- Check that your prompt clearly specifies output format
- Try increasing `THINKING_BUDGET` for complex pages
- Switch to a more capable model (flash → pro)
- Ask Cursor: "The LLM is returning malformed JSON. Help me debug this."

### Low Accuracy (High CER)
**Problem**: Extracted text doesn't match source documents  
**Solutions**:
1. **Improve prompts**: Be more specific about what to extract
2. **Add examples**: Include example entries in your prompt
3. **Use stronger model**: gemini-2.5-flash → gemini-2.5-pro
4. **Increase thinking budget**: More reasoning tokens for complex layouts
5. **Check document quality**: Ensure scans are high-resolution (300 DPI+)

### Wrong Fields Extracted
**Problem**: LLM puts data in wrong columns (e.g., city in "name" field)  
**Solutions**:
- Refine variable extraction prompt with clearer field definitions
- Add examples showing correct field assignments
- Ask Cursor: "The model confuses city names with person names. Help me fix the prompt."

### Processing Too Slow/Expensive
**Solutions**:
- Use cheaper model: gemini-2.5-pro → gemini-2.5-flash
- Reduce `THINKING_BUDGET` if documents are straightforward
- Increase `MAX_WORKERS` (but watch rate limits)
- Process sample first to test settings before full run

### Need Help?
- Ask Cursor/Copilot: Describe your specific error
- Check the logs: `data/01_dataset_construction/csvs/[filename].log`
- GitHub Issues: Report bugs or request features

## 🔬 Research Applications & Example Projects

### For Economic Historians

**Innovation & Technology Studies**
- Patent data → Innovation patterns, technology diffusion, inventor networks
- This dataset: ~240,000 German patents (1878-1918) for studying Second Industrial Revolution

**Firm & Industry Analysis**
- Company directories → Firm demographics, industry structure, geographic concentration
- Possible project: Extract all Berlin manufacturing firms (1880-1914) from Berliner Adressbücher

**Trade & Commerce**
- Trade statistics → Import/export patterns, commodity flows, price series
- Possible project: Digitize Deutscher Reichsanzeiger trade tables

**Labor & Social Structure**
- City directories → Occupational structure, social mobility, residential patterns
- Notarial records → Property transactions, inheritance patterns, credit networks

**Financial History**
- Stock exchange listings → Firm valuations, capital markets development
- Bank reports → Credit allocation, financial intermediation

### Beyond Economic History

- **Political Science**: Parliamentary records, voter registrations, party membership lists
- **Social History**: Civil registries (births, marriages, deaths), military records
- **Legal History**: Court proceedings, legal codes, notarial documents
- **Cultural Studies**: Newspaper archives, book catalogs, exhibition records

### Why This Approach Works for Economic History

Traditional datasets often require:
- **Years** of manual transcription
- **Thousands of dollars** in research assistant salaries
- **Data entry errors** from fatigue and complexity
- **Limited scale** due to cost constraints

With this pipeline:
- **Days/weeks** to process (mostly computational)
- **Hundreds of dollars** in API costs
- **Consistent quality** (AI doesn't get tired)
- **Scale limited only by budget**, not labor availability

## 💬 Questions & Contact

**For economic historians interested in using this approach:**
- Open a GitHub Discussion to share your use case
- Issues for bugs or technical problems
- Feel free to fork and adapt for your projects (MIT license)

**Collaboration opportunities:**
- Testing on different historical document types
- Comparing results across languages/periods
- Sharing ground truth datasets for benchmarking

## 📚 Citation

If you use this pipeline or approach in your research, please cite:

```bibtex
@software{griesshaber2024llm_patent_pipeline,
  title={Multimodal LLMs for Dataset Construction from Image Scans: German Patents (1877-1918)},
  author={Griesshaber, Niclas},
  year={2024},
  url={https://github.com/niclasgriesshaber/llm_patent_pipeline},
  note={A pipeline for extracting structured datasets from historical document scans using multimodal large language models}
}
```

## 📄 License

MIT License - Free to use, modify, and distribute. See [LICENSE](LICENSE) for details.

Perfect for academic research: cite the work, adapt it freely, share improvements back with the community.

## 🎓 About This Project

Developed as part of DPhil research at Oxford studying innovation and industrialization in Imperial Germany. 

**Key insight**: Modern multimodal LLMs can read and understand historical documents well enough to automate dataset construction at scale—opening new possibilities for data-intensive economic history research that was previously impractical due to transcription costs.

**Goal**: Make this technology accessible to economic historians who need datasets but lack the time/budget for traditional transcription, and don't necessarily have programming expertise (hence the emphasis on AI-assisted coding tools).

---

**Questions? Ideas? Collaboration proposals?** → [Open a GitHub Discussion](https://github.com/niclasgriesshaber/llm_patent_pipeline/discussions)

**Built for economic historians, by an economic historian** 📚 