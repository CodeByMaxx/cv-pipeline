import sys
import os
import re
import csv
import time
import argparse
import subprocess
import trafilatura
from datetime import date
from pathlib import Path
from typing import Literal, Optional, List
from pydantic import BaseModel, Field

# Umweltvariablen aus .env laden
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from langchain_ollama import ChatOllama
from google import genai
from google.genai import types, errors
from playwright.sync_api import sync_playwright

# Optional Cloud Clients
try:
    import boto3
except ImportError:
    boto3 = None

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None


# ==========================================
# 1. PYDANTIC SCHEMA
# ==========================================
class JobMetadata(BaseModel):
    company_name: str = Field(description="Name des ausschreibenden Unternehmens")
    company_size: str = Field(
        default="",
        description="Unternehmensgröße weltweit (z.B. '50-200 Mitarbeiter', '1000+', oder Schätzung basierend auf Text)",
    )
    contact_person: str = Field(
        default="Recruiting Team",
        description="Name des Ansprechpartners inkl. Rolle/Titel (z.B. 'Katharina Schwarzl-Moser (HR Manager)'), sonst 'Recruiting Team'",
    )
    job_title: str = Field(description="Exakter Titel der ausgeschriebenen Stelle")
    location: str = Field(
        description="Standort/Hauptsitz der Stelle (z.B. 'Wien, Österreich' oder 'Remote')"
    )

    language: Literal["Deutsch", "Englisch"] = Field(
        default="Englisch",
        description="Sprache der Stellenanzeige bzw. in welcher Sprache das Anschreiben/CV formuliert werden soll ('Deutsch' oder 'Englisch')",
    )

    cv_variant: Literal["cv_mlds.tex", "cv_devops.tex", "cv_backend.tex", "cv.tex"] = (
        Field(
            description="Wähle 'cv_mlds.tex' für Machine Learning, Data Science, AI/GenAI oder RAG. Wähle 'cv_devops.tex' für Cloud, Kubernetes, CI/CD, Infrastructure. Wähle 'cv_backend.tex' für Core C++, Python Backend oder High-Performance Engineering."
        )
    )

    cover_letter_variant: Literal[
        "cover_letter_mlds.tex", "cover_letter_devops.tex", "cover_letter_backend.tex"
    ] = Field(
        description="Wähle das am besten passende Anschreiben-Template entsprechend der Fachrichtung (mlds, devops oder backend)."
    )

    cv_rolle: str = Field(
        description="Prägnanter Rollentitel für den Header (z.B. 'AI & Software Engineer -- M&A Automation')"
    )
    cv_subheader: str = Field(
        description="Tech-Stack & Kernkompetenzen getrennt durch '\\cdot' (z.B. 'Python \\cdot LLM APIs \\cdot Financial Data Pipelines')"
    )

    salutation: str = Field(
        description="Passende Anrede (z.B. 'Dear Ms. Schwarzl-Moser,' oder 'Sehr geehrte Frau Schwarzl-Moser,')"
    )
    company_highlight: str = Field(
        description="Ein prägnanter Satz auf Englisch oder Deutsch über den Fokus des Unternehmens/der Rolle für das Anschreiben"
    )


# ==========================================
# 2. CRAWLER (TRAFILATURA + PLAYWRIGHT FIREFOX)
# ==========================================
def extract_text_from_url(url: str) -> str:
    print(f"[*] Fetching content from: {url}")
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(
            downloaded, include_links=False, include_comments=False
        )
        if text and len(text) > 300:
            print("[✓] Quick extraction via Trafilatura successful.")
            return text

    print("[!] Trafilatura fallback triggered. Launching Playwright Firefox browser...")
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        body_text = page.inner_text("body")
        browser.close()
        print("[✓] Browser extraction successful.")
        return body_text


# ==========================================
# 3. LLM EXTRACTORS (OLLAMA, GEMINI, AZURE, AWS)
# ==========================================
OLLAMA_MODEL_NAME = "qwen2.5"
GEMINI_MODEL_NAME = "gemini-3.8-flash"


def analyze_with_ollama(job_text: str) -> JobMetadata:
    print(f"[*] Analyzing job posting locally with Ollama ({OLLAMA_MODEL_NAME})...")
    llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0)
    structured_llm = llm.with_structured_output(JobMetadata)

    prompt = f"Analysiere die folgende Stellenanzeige und extrahiere die Daten exakt im vorgegebenen Schema:\n\n{job_text}"
    return structured_llm.invoke(prompt)


def analyze_with_gemini(job_text: str) -> JobMetadata:
    print(f"[*] Analyzing job posting with Gemini API ({GEMINI_MODEL_NAME})...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY ist nicht in den Umgebungsvariablen gesetzt!")

    client = genai.Client(
        api_key=api_key, http_options=types.HttpOptions(timeout=60000)
    )

    prompt = f"Analysiere die folgende Stellenanzeige und extrahiere die Daten exakt im vorgegebenen Schema:\n\n{job_text}"

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=JobMetadata,
        temperature=0.0,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
                config=config,
            )
            print("[✓] Gemini analysis completed successfully.")
            return JobMetadata.model_validate_json(response.text)
        except errors.APIError as e:
            if e.code in (503, 504) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(
                    f"[!] Gemini {e.code} ({e.message}). Retry {attempt + 1}/{max_retries} in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise e
        except Exception as e:
            print(f"[!] Error during Gemini API call: {e}")
            raise e


def analyze_with_azure(job_text: str) -> JobMetadata:
    if not AzureOpenAI:
        raise ImportError("openai package nicht installiert.")

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key or "dein_" in api_key:
        raise ValueError("AZURE_OPENAI_API_KEY ist nicht oder ungültig konfiguriert!")

    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    print(
        f"[*] Analyzing job posting with Microsoft Azure OpenAI ({deployment_name})..."
    )

    client = AzureOpenAI(
        api_key=api_key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    completion = client.beta.chat.completions.parse(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": "Analysiere die Stellenanzeige und extrahiere die Metadaten.",
            },
            {"role": "user", "content": job_text},
        ],
        response_format=JobMetadata,
        temperature=0.0,
    )
    print("[✓] Azure OpenAI analysis completed successfully.")
    return completion.choices.message.parsed


def analyze_with_aws(job_text: str) -> JobMetadata:
    if not boto3:
        raise ImportError("boto3 package nicht installiert.")

    aws_region = os.getenv("AWS_REGION", "eu-central-1")
    print(
        f"[*] Analyzing job posting with AWS Bedrock (Claude Sonnet 4.5 EU Inference Profile) in {aws_region}..."
    )

    bedrock = boto3.client(service_name="bedrock-runtime", region_name=aws_region)
    prompt = f"Analysiere die folgende Stellenanzeige und extrahiere die Daten exakt im Schema:\n\n{job_text}"

    model_id = os.getenv(
        "AWS_BEDROCK_MODEL_ID", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )

    response = bedrock.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"temperature": 0.0},
        toolConfig={
            "tools": [
                {
                    "toolSpec": {
                        "name": "extract_job_metadata",
                        "description": "Extract structured job metadata",
                        "inputSchema": {"json": JobMetadata.model_json_schema()},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": "extract_job_metadata"}},
        },
    )
    tool_use = response["output"]["message"]["content"][0]["toolUse"]
    print("[✓] AWS Bedrock analysis completed successfully.")
    return JobMetadata.model_validate(tool_use["input"])


# ==========================================
# 4. SETTINGS TEMPLATE & UTILS
# ==========================================
SETTINGS_TEMPLATE = r"""% ==========================================
% SETTINGS.TEX - GENERATED BY CV-PIPELINE
% ==========================================
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[american]{babel}
\usepackage[top=1.8cm, bottom=1.8cm, left=1.8cm, right=1.8cm]{geometry}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{graphicx}
\usepackage{calc}
\usepackage{charter}
\usepackage{microtype}
\usepackage{etoolbox}

% --- DESIGN FARBEN ---
\definecolor{primary}{RGB}{15, 23, 42}      % Deep Slate / Tech Dark
\definecolor{accent}{RGB}{37, 99, 235}     % Electric Blue
\definecolor{textgray}{RGB}{51, 65, 85}    % Slate Gray

% --- PERMANENTE ABSENDER-DATEN ---
\newcommand{\meinName}{Markus Gruber, M.Sc.}
\newcommand{\meineStrasse}{Maximilianstraße 55 Stg. B 20}
\newcommand{\meinOrt}{3100 St. Pölten}
\newcommand{\meineEmail}{Markus.Gruber4@gmx.net}
\newcommand{\meineTel}{+43 650 5871145}
\newcommand{\meinLinkedin}{https://www.linkedin.com/in/markus-gruber-70851428/}
\newcommand{\meinGithub}{https://github.com/CodeByMaxx}

% --- ROLLEN-DEFAULTS ---
\providecommand{\cvRolle}{}
\renewcommand{\cvRolle}{__CV_ROLLE__}
\providecommand{\cvSubheader}{}
\renewcommand{\cvSubheader}{__CV_SUBHEADER__}

% --- EMPFÄNGER-DATEN ---
\newcommand{\firmenName}{__COMPANY_NAME__}
\newcommand{\ansprechPartner}{__CONTACT_PERSON__}
\newcommand{\firmenStrasse}{__LOCATION__}
\newcommand{\firmenOrt}{__LOCATION__}

\newcommand{\stellenTitel}{__JOB_TITLE__}
\newcommand{\referenzNummer}{}

\newcommand{\anrede}{__SALUTATION__}
\newcommand{\firmenHighlight}{__COMPANY_HIGHLIGHT__}

% --- HEADER BEFEHL ---
\newcommand{\cvHeader}{%%
\begin{center}
    {\Huge\bfseries\color{primary} \meinName}\\[0.1cm]
    {\large\bfseries \cvRolle}\\[0.05cm]
    {\color{accent}\textbf{\cvSubheader}}\\[0.1cm]
    {\small \meinOrt, Austria \quad$\cdot$\quad \meineTel \quad$\cdot$\quad \href{mailto:\meineEmail}{\meineEmail} \quad$\cdot$\quad \href{\meinLinkedin}{LinkedIn} \quad$\cdot$\quad \href{\meinGithub}{GitHub}}
\end{center}
\vspace{0.1cm}
}

\newcommand{\cvsection}[1]{%%
    \vspace{0.25cm}%%
    {\Large\bfseries\color{primary} #1}\par%%
    \vspace{-0.1cm}{\color{primary}\rule{\textwidth}{0.8pt}}\par%%
    \vspace{0.15cm}%%
}
\newcommand{\daterange}[2]{\hfill{\small #1 -- #2}}

\hypersetup{
    colorlinks=true,
    linkcolor=primary,
    urlcolor=primary,
    pdfauthor={\meinName},
    pdftitle={Application Documents - \meinName}
}
"""


def render_settings(meta: JobMetadata) -> str:
    return (
        SETTINGS_TEMPLATE.replace("__CV_ROLLE__", meta.cv_rolle)
        .replace("__CV_SUBHEADER__", meta.cv_subheader)
        .replace("__COMPANY_NAME__", meta.company_name)
        .replace("__CONTACT_PERSON__", meta.contact_person)
        .replace("__LOCATION__", meta.location)
        .replace("__JOB_TITLE__", meta.job_title)
        .replace("__SALUTATION__", meta.salutation)
        .replace("__COMPANY_HIGHLIGHT__", meta.company_highlight)
    )


# ==========================================
# 5. CSV TRACKING
# ==========================================
CSV_FILE = "applications.csv"
CSV_HEADERS = [
    "Gefunden am / Monat",
    "Link / Quelle",
    "Priorität",
    "Status",
    "Bewerbungsdatum",
    "Company",
    "Unternehmensgröße (weltweit)",
    "Sprache",
    "Template",
    "Role",
    "LLM Model",
    "Grund Absage",
    "Hauptsitz",
]


def append_to_csv(meta: JobMetadata, url: str, used_model: str):
    file_exists = Path(CSV_FILE).exists()
    today_str = date.today().strftime("%Y-%m-%d")

    row = {
        "Gefunden am / Monat": today_str,
        "Link / Quelle": url or "Manual / Compile-Only",
        "Priorität": "Hoch",
        "Status": "Erstellt",
        "Bewerbungsdatum": today_str,
        "Company": meta.company_name,
        "Unternehmensgröße (weltweit)": meta.company_size,
        "Sprache": meta.language,
        "Template": meta.cv_variant,
        "Role": meta.job_title,
        "LLM Model": used_model,
        "Grund Absage": "",
        "Hauptsitz": meta.location,
    }

    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[✓] Application logged to {CSV_FILE} (Model: {used_model})")


# ==========================================
# 6. PIPELINE & CONFIGURABLE CASCADE
# ==========================================
# Standard-Reihenfolge der Kaskade (kann angepasst werden)
DEFAULT_CASCADE_ORDER = ["gemini", "aws", "azure", "ollama"]


def run_pipeline(url: str, cascade_order: List[str], dry_run: bool = False):
    job_text = extract_text_from_url(url)

    meta = None
    used_model = None

    # Provider-Mapping
    providers = {
        "gemini": lambda: analyze_with_gemini(job_text),
        "aws": lambda: analyze_with_aws(job_text),
        "azure": lambda: analyze_with_azure(job_text),
        "ollama": lambda: analyze_with_ollama(job_text),
    }

    print(f"[*] Starte Kaskade mit Reihenfolge: {cascade_order}")

    for provider_name in cascade_order:
        if provider_name not in providers:
            print(f"[!] Unbekannter Provider in Kaskade übersprungen: {provider_name}")
            continue

        try:
            print(f"[*] Versuche Provider: {provider_name.upper()}...")
            meta = providers[provider_name]()
            used_model = provider_name
            print(f"[✓] Erfolgreich analysiert mit Provider: {provider_name.upper()}")
            break
        except Exception as e:
            print(f"[!] Provider {provider_name.upper()} fehlgeschlagen: {e}")
            print("[!] Versuche nächsten Provider in der Kaskade...")
            continue

    if not meta:
        raise RuntimeError("[X] Alle Provider in der Kaskade sind fehlgeschlagen!")

    print("\n--- EXTRAHIERTE METADATEN ---")
    print(meta.model_dump_json(indent=2))

    if dry_run:
        print("[*] Dry-run Modus aktiv. Keine Dateien geschrieben.")
        return

    # settings.tex schreiben
    settings_content = render_settings(meta)
    Path("settings.tex").write_text(settings_content, encoding="utf-8")
    print("[✓] settings.tex erfolgreich generiert.")

    # CSV Loggen
    append_to_csv(meta, url, used_model)

    # LaTeX Kompilieren
    target_dir = Path("output") / sanitize_name(meta.company_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        cv_cmd = [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            f"-output-directory={target_dir}",
            meta.cv_variant,
        ]
        print(f"[*] Compiling CV ({meta.cv_variant})...")
        subprocess.run(cv_cmd, check=True)

        cl_cmd = [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            f"-output-directory={target_dir}",
            meta.cover_letter_variant,
        ]
        print(f"[*] Compiling Cover Letter ({meta.cover_letter_variant})...")
        subprocess.run(cl_cmd, check=True)
        print(f"[✓] PDFs erfolgreich erstellt unter: {target_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[!] Fehler beim LaTeX-Kompilieren: {e}")


def sanitize_name(name: str) -> str:
    clean = re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")
    return re.sub(r"[-_]+", "_", clean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV Pipeline CLI")
    parser.add_argument("url", nargs="?", help="URL der Stellenanzeige")
    parser.add_argument(
        "--provider", default="cascade", help="Einzelner Provider oder 'cascade'"
    )
    parser.add_argument(
        "--cascade",
        nargs="+",
        default=DEFAULT_CASCADE_ORDER,
        help="Reihenfolge der Kaskade (z.B. gemini aws ollama)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur analysieren, nicht schreiben/kompilieren",
    )

    args = parser.parse_args()

    if not args.url:
        print("Fehler: Bitte gib eine URL an.")
        sys.exit(1)

    order = args.cascade
    if args.provider != "cascade":
        # Wenn ein spezifischer Provider gewählt wurde, erzwingen wir diesen als einzigen in der Kaskade
        order = [args.provider]

    run_pipeline(args.url, cascade_order=order, dry_run=args.dry_run)
