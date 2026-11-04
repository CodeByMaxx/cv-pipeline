# Automated CV & Application Document Pipeline (`main.py`)

An automated, multi-provider Python application designed to crawl job postings, analyze requirements using LLMs, render clean LaTeX/PDF application packages (CVs and Cover Letters), and dynamically track progress in a structured CSV log.

## Architecture & System Overview

The pipeline processes job listings via URL or local text files, passes cleansed input through an active LLM Provider Cascade, maps extracted structured data into dynamic LaTeX templates (`settings.tex`), compiles PDFs asynchronously, and logs each application.

```
                  +-----------------------------------+
                  |  Input: Job Posting URL / Text    |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |   Web Extraction & Truncation     |
                  |  (Trafilatura / Playwright FF)    |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |   Multi-Provider LLM Cascade      |
                  | (Gemini -> AWS -> Azure -> Local) |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  Data Cleansing & Sanitization    |
                  | (Regex Gender Strip, Math Escape) |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  LaTeX Renderer & Compiler        |
                  | (settings.tex -> PDF Generation)  |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  | CSV Application Tracking Logging  |
                  +-----------------------------------+
```

### Core Workflow Steps

1. **Extraction:** Crawls text from target web pages using standard HTML extraction (`trafilatura`) or headless Firefox rendering (`playwright`) for JavaScript-heavy sites. Content is bounded to 15,000 characters to ensure optimal prompt window usage.

2. **LLM Extraction Cascade:** Queries configured LLM providers in sequence using Pydantic structured output validation. If a model fails or times out, the pipeline seamlessly transitions to the next available provider.

3. **Field Sanitization & Protection:** Strips corporate gender labels (`(w/m/x)`, `(f/m/d)`, `(gn)`) and sanitizes special characters (`&`, `%`, `$`, `_`) to ensure LaTeX compilation safety while nullifying `<UNKNOWN>` fields.

4. **LaTeX Compilation:** Dynamically injects context parameters into `settings.tex` and runs `latexmk` isolated per company target directory.

5. **CSV Logging:** Stores historical application metadata (company name, size, salary/fit estimate, role, timestamp) inside `applications.csv`.

---

## Features

- **Multi-Provider LLM Fallback:** Resilient cascade processing via **Google Gemini** (`gemini-3.8-flash`), **AWS Bedrock** (Claude Sonnet 3.5 / 4.5 EU Inference Profile), **Azure OpenAI** (`gpt-4o-mini`), and local **Ollama** (`qwen2.5`).
- **Dynamic Web Crawling:** Headless browser capabilities using Playwright when standard HTTP scrapers encounter dynamic paywalls or SPAs.
- **Pydantic Validation:** Strict schema validation ensuring extracted job attributes match expected formats.
- **Automated Gender & Special Char Stripping:** Automatically removes gender suffixes (`(m/f/d)`, `(w/m/x)`, `(gn)`) from header roles and job titles while escaping reserved TeX characters.
- **`<UNKNOWN>` Protection:** Prevents broken placeholder text inside header fields during incomplete LLM outputs.
- **CLI Overrides:** Allows manual overriding of any structured output key directly from terminal commands (`--set`).
- **Offline Multi-Variant Generator:** Generates language-specific application sets (German/English) for pre-built role specializations (Backend, DevOps, Machine Learning) when no input parameter is supplied.

---

## 📄 Compiled PDF Results & Direct Output Access

Upon running the local test mode (`python main.py`) or live runs, compiled PDF documents are generated in isolated target folders under the `output/` directory:

### Local Offline Output Paths & Links (`output/Lokale_Rollen_Laeufe/`)

| Specialization / Role | Language | CV PDF (Relative Path) | Cover Letter PDF (Relative Path) |
| :--- | :---: | :--- | :--- |
| **Backend & Systems** | DE | [`cv_backend_de.pdf`](output/Lokale_Rollen_Laeufe/cv_backend_de/cv_backend_de.pdf) | [`cover_letter_backend_de.pdf`](output/Lokale_Rollen_Laeufe/cv_backend_de/cover_letter_backend_de.pdf) |
| **Backend & Systems** | EN | [`cv_backend.pdf`](output/Lokale_Rollen_Laeufe/cv_backend_en/cv_backend.pdf) | [`cover_letter_backend.pdf`](output/Lokale_Rollen_Laeufe/cv_backend_en/cover_letter_backend.pdf) |
| **Cloud & DevOps** | DE | [`cv_devops_de.pdf`](output/Lokale_Rollen_Laeufe/cv_devops_de/cv_devops_de.pdf) | [`cover_letter_devops_de.pdf`](output/Lokale_Rollen_Laeufe/cv_devops_de/cover_letter_devops_de.pdf) |
| **Cloud & DevOps** | EN | [`cv_devops.pdf`](output/Lokale_Rollen_Laeufe/cv_devops_en/cv_devops.pdf) | [`cover_letter_devops.pdf`](output/Lokale_Rollen_Laeufe/cv_devops_en/cover_letter_devops.pdf) |
| **ML & Data Science** | DE | [`cv_mlds_de.pdf`](output/Lokale_Rollen_Laeufe/cv_mlds_de/cv_mlds_de.pdf) | [`cover_letter_mlds_de.pdf`](output/Lokale_Rollen_Laeufe/cv_mlds_de/cover_letter_mlds_de.pdf) |
| **ML & Data Science** | EN | [`cv_mlds.pdf`](output/Lokale_Rollen_Laeufe/cv_mlds_en/cv_mlds.pdf) | [`cover_letter_mlds.pdf`](output/Lokale_Rollen_Laeufe/cv_mlds_en/cover_letter_mlds.pdf) |

### Dynamic Live Target Directories (`output/<Company_Name>/`)

For actual live application pipelines, output files are compiled inside dedicated company folders:
- `output/<Company_Name>/`

---

## Requirements & Setup

### Prerequisites

- **Python:** 3.10 or higher
- **LaTeX Distribution:** TeX Live or MiKTeX with `latexmk` and standard packages (`babel`, `geometry`, `microtype`, `enumitem`, `charter`).

### Dependencies Installation

```bash
# Clone repository and enter project folder
cd cv-pipeline

# Install Python packages
pip install pydantic python-dotenv langchain-ollama google-genai playwright trafilatura openai boto3

# Install Playwright browser dependencies
playwright install firefox
```

---

## Configuration (`.env`)

Create a `.env` file in the root directory with your preferred API keys and model options:

```env
# Google Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key

# AWS Bedrock Credentials & Options
AWS_REGION=eu-central-1
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_BEDROCK_MODEL_ID=eu.anthropic.claude-sonnet-4-5-20250929-v1:0

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

---

## CLI Usage Guide

### 1. Process Live Job Posting (URL or File)

```bash
python3 main.py "https://jobs.mchunters.at/stellenanzeigen/mh618" --set company_name="McHunters" contact_person="DI Thomas Haudum, MBA" location="Linz"
```

### 2. Specify Target Language & Cascade Order

```bash
python3 main.py "job_description.txt" --lang de --cascade gemini aws ollama
```

### 3. Dry-Run Mode (Extract Only)

Analyze and extract metadata without writing files or compiling PDFs:

```bash
python3 main.py "https://example.com/job-posting" --dry-run
```

### 4. Local Offline Test Mode

Executes generation across all technical profiles (Backend, DevOps, ML/DS) in both German and English:

```bash
python3 main.py
```
