# Automated CV & Cover Letter Application Pipeline

An automated pipeline to parse job postings (via URL or local text file), extract structured metadata using a customizable LLM cascade (Gemini, AWS Bedrock, Azure OpenAI, Ollama), generate dynamic LaTeX settings (`settings.tex`), log applications into a CSV tracking file, and compile tailored PDF documents (`.tex` $\rightarrow$ `.pdf`) using `latexmk`.

---

## Features

- **Multi-Source Extraction:** Supports fetching job descriptions from URLs using `trafilatura` with an automated fallback to Playwright (Firefox headless).
- **LLM Cascade Support:** Automatically falls back through multiple AI providers if one fails:
  - Google Gemini (`gemini-3.8-flash`)
  - AWS Bedrock (Claude Sonnet EU Inference Profile via `boto3`)
  - Microsoft Azure OpenAI
  - Local Ollama (`qwen2.5` via LangChain)
- **Pydantic Validation:** Strict schema validation ensuring extracted job attributes match expected formats.
- **Dynamic LaTeX Generation:** Automatically generates a centralized `settings.tex` with tailored roles, subheaders, and localized elements (German / English via Babel).
- **Automated CSV Tracking:** Logs every application run, including source, company size, model used, and status into `applications.csv`.
- **Offline Multi-Variant Testing:** Runs local test runs across all roles and language variants when no source URL/file is provided.

---

## 📄 Compiled PDF Results & Direct Access

Nach Ausführung des Offline-Modus (`python main.py`) kannst du die generierten PDF-Dokumente direkt über die folgenden Verzeichnisse und Links aufrufen:

### Lokale Offline-Ausgabepfade & PDFs (`output/Lokale_Rollen_Laeufe/`)

| Profil / Fachrichtung | Sprache | CV-PDF (Direktlink / Ordner) | Anschreiben-PDF (Direktlink / Ordner) |
| :--- | :---: | :--- | :--- |
| **Backend & Systems** | DE | [`cv_backend_de.pdf`](output/Lokale_Rollen_Laeufe/cv_backend_de/cv_backend_de.pdf) | [`cover_letter_backend_de.pdf`](output/Lokale_Rollen_Laeufe/cv_backend_de/cover_letter_backend_de.pdf) |
| **Backend & Systems** | EN | [`cv_backend.pdf`](output/Lokale_Rollen_Laeufe/cv_backend_en/cv_backend.pdf) | [`cover_letter_backend.pdf`](output/Lokale_Rollen_Laeufe/cv_backend_en/cover_letter_backend.pdf) |
| **Cloud & DevOps** | DE | [`cv_devops_de.pdf`](output/Lokale_Rollen_Laeufe/cv_devops_de/cv_devops_de.pdf) | [`cover_letter_devops_de.pdf`](output/Lokale_Rollen_Laeufe/cv_devops_de/cover_letter_devops_de.pdf) |
| **Cloud & DevOps** | EN | [`cv_devops.pdf`](output/Lokale_Rollen_Laeufe/cv_devops_en/cv_devops.pdf) | [`cover_letter_devops.pdf`](output/Lokale_Rollen_Laeufe/cv_devops_en/cover_letter_devops.pdf) |
| **ML & Data Science** | DE | [`cv_mlds_de.pdf`](output/Lokale_Rollen_Laeufe/cv_mlds_de/cv_mlds_de.pdf) | [`cover_letter_mlds_de.pdf`](output/Lokale_Rollen_Laeufe/cv_mlds_de/cover_letter_mlds_de.pdf) |
| **ML & Data Science** | EN | [`cv_mlds.pdf`](output/Lokale_Rollen_Laeufe/cv_mlds_en/cv_mlds.pdf) | [`cover_letter_mlds.pdf`](output/Lokale_Rollen_Laeufe/cv_mlds_en/cover_letter_mlds.pdf) |

### Dynamische Unternehmensläufe (`output/<Company_Name>/`)
Für echte Stellenanzeigen werden die PDFs jeweils im generierten Firmenordner abgelegt:
- `output/<company_name>/`

---

## Requirements

Ensure you have Python installed along with the required packages:

```bash
pip install pydantic trafilatura playwright langchain-ollama google-genai python-dotenv boto3 openai
playwright install
```

You also need a LaTeX distribution installed on your system (e.g., TeX Live or MikTeX) that includes `latexmk`.

---

## Configuration

Create a `.env` file in the root directory with your API keys:

```env
GEMINI_API_KEY=your_gemini_api_key
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AWS_REGION=eu-central-1
```

---

## Usage

### 1. Process a Job Posting (URL or File)
Run the pipeline by passing a job posting URL or a path to a text file:

```bash
python main.py "https://example.com/job-posting"
```

### 2. Specify Language and Cascade Order
```bash
python main.py "job_description.txt" --lang de --cascade gemini aws ollama
```

### 3. Dry-Run (Extract Only)
Analyze the posting without compiling PDFs or writing logs:
```bash
python main.py "https://example.com/job-posting" --dry-run
```

### 4. Offline / Local Test Run
Execute offline generation across all roles (Backend, DevOps, ML/DS) and languages (DE/EN):
```bash
python main.py
```
