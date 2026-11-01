# CV & Application Pipeline

An automated, local-first Python pipeline that ingests job postings via web scraping, extracts structured metadata using a customizable LLM cascade, and automatically compiles tailored LaTeX CVs and cover letters.

## 🚀 Key Features

- **Smart Web Scraping:** Fast content extraction via `trafilatura` with an automated `Playwright` (Firefox) fallback for dynamic or protected JavaScript job boards.
- **Configurable LLM Cascade:** Automatically falls back through multiple providers (`Gemini`, `AWS Bedrock Claude`, `Ollama`, `Azure OpenAI`) to guarantee robust extraction.
- **Strict Schema Validation:** Powered by `Pydantic` to ensure precise, type-safe JSON extraction for company names, roles, requirements, and templates.
- **Automated LaTeX Compilation:** Renders customized `settings.tex` configurations and compiles both CV and cover letter PDFs using `latexmk`.
- **Local Application Tracking:** Automatically logs every processed job into an `applications.csv` file for status and pipeline management.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Parsing & Scraping:** Trafilatura, Playwright (Firefox)
- **Validation:** Pydantic V2
- **LLM Integrations:** 
  - Google GenAI (`gemini-3.8-flash`)
  - AWS Bedrock (`boto3` / Claude Sonnet 4.5 EU Inference Profile)
  - Ollama (`qwen2.5` local)
  - Azure OpenAI (optional)
- **Typesetting:** LaTeX (`latexmk`)

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CodeByMaxx/cv-pipeline.git
   cd cv-pipeline
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install pydantic trafilatura playwright langchain-ollama google-genai boto3 python-dotenv openai
   playwright install firefox
   ```

4. **Configure environment variables:**
   Create a `.env` file in the root directory:
   ```ini
   GEMINI_API_KEY=your_gemini_key_here
   AWS_REGION=eu-central-1
   AWS_BEDROCK_MODEL_ID=eu.anthropic.claude-sonnet-4-5-20250929-v1:0
   ```

---

## 📖 Usage

Run the pipeline by passing a job posting URL:

```bash
python main.py "https://example.com/job-posting"
```

### Advanced Options:
- **Custom Provider Cascade:**
  ```bash
  python main.py "https://example.com/job-posting" --cascade gemini aws ollama
  ```
- **Force a Single Provider:**
  ```bash
  python main.py "https://example.com/job-posting" --provider ollama
  ```
- **Dry-Run (Analysis only, no PDF compilation):**
  ```bash
  python main.py "https://example.com/job-posting" --dry-run
  ```

---

## 📄 License
Private project. All rights reserved.
