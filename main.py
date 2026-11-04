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
# 1. PYDANTIC SCHEMA & CLEANING
# ==========================================
class JobMetadata(BaseModel):
    company_name: str = Field(description="Name des ausschreibenden Unternehmens")
    company_size: str = Field(
        default="",
        description="Unternehmensgröße weltweit (z.B. '50-200 Mitarbeiter', '1000+', oder Schätzung basierend auf Text)",
    )
    contact_person: str = Field(
        default="Recruiting Team",
        description="Name des Ansprechpartners inkl. Rolle/Titel (z.B. 'Katharina Schwarzl-Moser (HR Manager)'). Falls kein Name genannt wird, verwende 'Recruiting Team'.",
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
        description="Tech-Stack & Kernkompetenzen getrennt durch '\\cdot' (z.B. 'Python \\cdot LLM APIs \\cdot Financial Data Pipelines \\cdot Agent Workflows')"
    )

    salutation: str = Field(
        description="Passende Anrede (z.B. 'Sehr geehrte Frau Müller,' oder 'Dear Mr. Smith,'). Falls kein Name bekannt ist, verwende 'Sehr geehrte Damen und Herren,' bzw. 'Dear Recruiting Team,'."
    )
    company_highlight: str = Field(
        description="Ein sehr kurzer, prägnanter Fokus auf Deutsch oder Englisch (MAXIMAL 3-6 Wörter, z.B. 'der modernen Cloud-Architektur und KI-Integration' oder 'your focus on automated cloud platforms'). KEINE ganzen Sätze, KEINE Nennung des Firmennamens!"
    )

    fit_score: str = Field(
        default="85%",
        description="Einschätzung des prozentualen Fits (Übereinstimmung deines Profils mit den Anforderungen der Stelle, z.B. '90%').",
    )


def clean_job_title(title: str) -> str:
    if not title:
        return ""
    patterns = [
        r"\s*\(all genders?\)",
        r"\s*\(f/m/d\)",
        r"\s*\(m/f/d\)",
        r"\s*\(m/w/d\)",
        r"\s*\(w/m/d\)",
        r"\s*\(f/m/x\)",
        r"\s*\(m/w/x\)",
        r"\s*\(gn\)",
        r"\s*-\s*f/m/d",
        r"\s*-\s*m/f/d",
        r"\s*-\s*m/w/d",
        r"\s*-\s*w/m/d",
        r"\s*-\s*f/m/x",
    ]
    cleaned = title
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


# ==========================================
# 2. CRAWLER / TEXT-EXTRACTOR
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

    prompt = (
        "Analysiere die folgende Stellenanzeige. "
        "WICHTIG: Falls im Text ein echter Name eines Ansprechpartners steht, extrahiere diesen bei 'contact_person' "
        "und setze eine persönliche Anrede unter 'salutation'. "
        "Bewerte zudem den fachlichen Fit deines Profils als Prozentwert (z.B. '85%') unter 'fit_score'.\n\n"
        f"Stellenanzeige:\n{job_text}"
    )
    return structured_llm.invoke(prompt)


def analyze_with_gemini(job_text: str) -> JobMetadata:
    print(f"[*] Analyzing job posting with Gemini API ({GEMINI_MODEL_NAME})...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY ist nicht in den Umgebungsvariablen gesetzt!")

    client = genai.Client(
        api_key=api_key, http_options=types.HttpOptions(timeout=60000)
    )

    prompt = (
        "Analysiere die folgende Stellenanzeige. "
        "WICHTIG: Falls im Text ein echter Name eines Ansprechpartners steht, extrahiere diesen bei 'contact_person' "
        "und setze eine persönliche Anrede unter 'salutation'. "
        "Bewerte zudem den fachlichen Fit deines Profils als Prozentwert (z.B. '85%') unter 'fit_score'.\n\n"
        f"Stellenanzeige:\n{job_text}"
    )

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
                "content": "Analysiere die Stellenanzeige, extrahiere Metadaten und bestimme den Fit als Prozentwert.",
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
    prompt = (
        "Analysiere die folgende Stellenanzeige. Extrahiere Metadaten und bestimme den passenden Fit-Prozentwert.\n\n"
        f"{job_text}"
    )

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
% SETTINGS.TEX - ZENTRALE KONFIGURATION
% ==========================================
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[__BABEL_LANG__]{babel}
\usepackage[top=1.8cm, bottom=1.8cm, left=1.8cm, right=1.8cm]{geometry}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{graphicx}
\usepackage{calc}
\usepackage{charter}
\usepackage{microtype}
\usepackage{etoolbox}
\usepackage{ifthen}
\usepackage{pdftexcmds}

% --- DESIGN FARBEN ---
\definecolor{primary}{RGB}{15, 23, 42}      % Deep Slate / Tech Dark
\definecolor{accent}{RGB}{37, 99, 235}     % MP Blue Accent
\definecolor{textgray}{RGB}{51, 65, 85}    % Charcoal Body Text

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

% --- INTELLIGENTE ADRESS-AUSGABE ---
\newcommand{\firmenAdresse}{%
    \def\tempStrasse{__LOCATION__}%
    \def\tempOrt{__LOCATION__}%
    \ifx\tempStrasse\tempOrt
        \tempOrt%
    \else
        __LOCATION__ \\[0.15em] __LOCATION__%
    \fi
}

\newcommand{\stellenTitel}{__JOB_TITLE__}
\newcommand{\referenzNummer}{}

\newcommand{\anrede}{__SALUTATION__}
\newcommand{\firmenHighlight}{__COMPANY_HIGHLIGHT__}

% --- DYNAMISCHER FLIESSTEXT-BEZUG ---
\newcommand{\firmenFlussbezug}{__FIRMEN_FLUSSBEZUG__}

% --- HEADER BEFEHL ---
\newcommand{\cvHeader}{%
\begin{center}
    {\Huge\bfseries\color{primary} \meinName}\\[0.1cm]
    {\large\bfseries \cvRolle}\\[0.05cm]
    {\small\textsl{\cvSubheader}}\\[0.1cm]
    {\small \meinOrt, __COUNTRY__ \quad\textperiodcentered\quad \meineTel \quad\textperiodcentered\quad \href{mailto:\meineEmail}{\meineEmail} \quad\textperiodcentered\quad \href{\meinLinkedin}{LinkedIn} \quad\textperiodcentered\quad \href{\meinGithub}{GitHub}}
\end{center}
\vspace{0.1cm}
}

% --- BEFEHLE FÜR LEBENSLÄUFE & ANSCHREIBEN ---
\newcommand{\cvsection}[1]{%
    \vspace{0.25cm}%
    {\Large\bfseries\color{primary} #1}\par%
    \vspace{-0.1cm}{\color{primary}\rule{\textwidth}{0.8pt}}\par%
    \vspace{0.15cm}%
}

\newcommand{\daterange}[2]{\hfill{\small #1 -- #2}}

% --- ENGINEERING APPROACH & MINDSET DEFINITION ---
\newcommand{\engineeringApproach}{%
    \cvsection{\ifnum\pdfstrcmp{\languagename}{german}=0 Arbeitsweise \& Mindset\else Engineering Approach \& Mindset\fi}%
    \begin{itemize}[leftmargin=*, itemsep=2pt]%
        \ifnum\pdfstrcmp{\languagename}{german}=0
            \item \textbf{Systematisches Problemlösen:} Analytischer First-Principles-Ansatz für systematisches Debugging und Stabilitätsoptimierung.%
            \item \textbf{Fokus auf Qualität & Effizienz:} Pragmatisches Engineering-Mindset, das schnelles Deployment mit langfristiger Wartbarkeit und hoher Performance verbindet.%
            \item \textbf{Funktionsübergreifende Integration:} Fungiert als technisches Bindeglied zwischen Entwicklung, QA und Architektur.%
        \else
            \item \textbf{Systematic Problem Solving:} First-principles analytical approach to systematic debugging and stability optimization.%
            \item \textbf{Quality & Efficiency Focus:} Pragmatic engineering mindset balancing rapid deployment with long-term maintainability and high performance.%
            \item \textbf{Cross-Functional Integration:} Acts as the technical glue bridging development, QA, and architecture.%
        \fi
    \end{itemize}%
}

% --- HYPERREF SETUP ---
\hypersetup{
    colorlinks=true,
    linkcolor=primary,
    urlcolor=primary,
    pdfauthor={\meinName},
    pdftitle={Application Documents - \meinName}
}
"""


def escape_latex(text: str) -> str:
    if not text:
        return ""
    chars = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    pattern = re.compile(
        "|".join(
            re.escape(key) for key in sorted(chars.keys(), key=lambda item: -len(item))
        )
    )
    return pattern.sub(lambda match: chars[match.group(0)], text)


def render_settings(meta: JobMetadata, lang: str) -> str:
    babel_lang = "german" if lang.lower() in ["de", "deutsch", "german"] else "american"
    country = "Österreich" if lang.lower() in ["de", "deutsch", "german"] else "Austria"

    flussbezug = (
        r"\textbf{Ihr Unternehmen}~"
        if lang.lower() in ["de", "deutsch", "german"]
        else r"\textbf{Your company}~"
    )

    def sanitize_field(val: str) -> str:
        if not val or "<unknown>" in val.lower():
            return ""
        return val

    meta.job_title = sanitize_field(meta.job_title)
    meta.cv_rolle = sanitize_field(meta.cv_rolle) or meta.job_title
    meta.cv_subheader = sanitize_field(meta.cv_subheader)
    meta.company_name = sanitize_field(meta.company_name)
    meta.contact_person = sanitize_field(meta.contact_person)
    meta.location = sanitize_field(meta.location)

    safe_cv_rolle = escape_latex(meta.cv_rolle)
    clean_subheader = escape_latex(meta.cv_subheader)
    clean_subheader = re.sub(
        r"\\textbackslash\{\\}cdot|·",
        r"\\quad\\textperiodcentered\\quad",
        clean_subheader,
    )

    return (
        SETTINGS_TEMPLATE.replace("__BABEL_LANG__", babel_lang)
        .replace("__COUNTRY__", country)
        .replace("__FIRMEN_FLUSSBEZUG__", flussbezug)
        .replace("__CV_ROLLE__", safe_cv_rolle)
        .replace("__CV_SUBHEADER__", clean_subheader)
        .replace("__COMPANY_NAME__", escape_latex(meta.company_name))
        .replace("__CONTACT_PERSON__", escape_latex(meta.contact_person))
        .replace("__LOCATION__", escape_latex(meta.location))
        .replace("__JOB_TITLE__", escape_latex(meta.job_title))
        .replace("__SALUTATION__", escape_latex(meta.salutation))
        .replace("__COMPANY_HIGHLIGHT__", escape_latex(meta.company_highlight))
    )


def resolve_file_variants(meta: JobMetadata, lang_setting: str) -> tuple[str, str, str]:
    if lang_setting in ["de", "en"]:
        target_lang = lang_setting
    else:
        target_lang = "de" if meta.language == "Deutsch" else "en"

    def get_variant_filename(filename: str) -> str:
        stem = Path(filename).stem
        if stem.endswith("_de"):
            stem = stem[:-3]

        if target_lang == "de":
            candidate = f"{stem}_de.tex"
            if Path(candidate).exists():
                return candidate

        if Path(filename).exists():
            return filename

        candidate_std = f"{stem}.tex"
        if Path(candidate_std).exists():
            return candidate_std

        return filename

    cv_file = get_variant_filename(meta.cv_variant)
    cl_file = get_variant_filename(meta.cover_letter_variant)

    return target_lang, cv_file, cl_file


# ==========================================
# 5. CLEANUP HELPER
# ==========================================
def cleanup_target_dir(target_dir: Path):
    for temp_file in target_dir.glob("*"):
        if temp_file.is_file():
            if temp_file.suffix.lower() not in [".pdf", ".log"]:
                temp_file.unlink()


# ==========================================
# 6. CSV TRACKING
# ==========================================
CSV_FILE = "applications.csv"
CSV_HEADERS = [
    "E",
    "Gefunden am / Monat",
    "Link / Quelle",
    "Priorität",
    "Status",
    "Bewerbungsdatum",
    "Company",
    "Firmengröße (weltweit, ca.)",
    "Template",
    "Role",
    "Grund Absage",
    "Hauptsitz",
    "Sprache",
    "Fit",
]


def append_to_csv(meta: JobMetadata, source: str, used_model: str, resolved_lang: str):
    file_exists = Path(CSV_FILE).exists()
    today_str = date.today().strftime("%d.%m.%Y")

    row = {
        "E": "",
        "Gefunden am / Monat": today_str,
        "Link / Quelle": source or "Manual / Compile-Only",
        "Priorität": "Hoch",
        "Status": "Erstellt",
        "Bewerbungsdatum": today_str,
        "Company": meta.company_name,
        "Firmengröße (weltweit, ca.)": meta.company_size,
        "Template": meta.cv_variant,
        "Role": meta.job_title,
        "Grund Absage": "",
        "Hauptsitz": meta.location,
        "Sprache": "Deutsch" if resolved_lang == "de" else "Englisch",
        "Fit": meta.fit_score,
    }

    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(
        f"[✓] Application logged to {CSV_FILE} (Model: {used_model}, Fit: {meta.fit_score})"
    )


def sanitize_name(name: str) -> str:
    clean = re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")
    return re.sub(r"[-_]+", "_", clean)


# ==========================================
# 7. PIPELINE & CONFIGURABLE CASCADE
# ==========================================
DEFAULT_CASCADE_ORDER = ["gemini", "aws", "azure", "ollama"]


def run_pipeline(
    source: str,
    cascade_order: List[str],
    lang_setting: str = "auto",
    dry_run: bool = False,
    overrides: Optional[List[str]] = None,
):
    if os.path.exists(source):
        print(f"[*] Reading job description from file: {source}")
        job_text = Path(source).read_text(encoding="utf-8")
    else:
        job_text = extract_text_from_url(source)

    max_chars = 15000
    if len(job_text) > max_chars:
        print(
            f"[!] Hinweis: Text ist {len(job_text)} Zeichen lang. Kürze exakt auf die ersten {max_chars} Zeichen..."
        )
        job_text = job_text[:max_chars]

    meta = None
    used_model = None

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

    if overrides:
        for item in overrides:
            if "=" in item:
                key, val = item.split("=", 1)
                key = key.strip()
                val = val.strip()
                if hasattr(meta, key):
                    old_val = getattr(meta, key)
                    setattr(meta, key, val)
                    print(f"[*] Generischer Override [{key}]: '{old_val}' -> '{val}'")
                else:
                    print(
                        f"[!] Warnung: Feld '{key}' existiert nicht im JobMetadata-Schema."
                    )

    meta.job_title = clean_job_title(meta.job_title)
    meta.cv_rolle = clean_job_title(meta.cv_rolle.replace("\n", " "))
    meta.cv_subheader = clean_job_title(meta.cv_subheader.replace("\n", " "))

    resolved_lang, cv_file, cl_file = resolve_file_variants(meta, lang_setting)

    print("\n--- EXTRAHIERTE & ANGEPASSTE METADATEN ---")
    print(meta.model_dump_json(indent=2))

    if dry_run:
        print("[*] Dry-run Modus aktiv. Keine Dateien geschrieben.")
        return

    settings_content = render_settings(meta, resolved_lang)
    Path("settings.tex").write_text(settings_content, encoding="utf-8")
    print("[✓] settings.tex erfolgreich generiert.")

    append_to_csv(meta, source, used_model, resolved_lang)

    target_dir = Path("output") / sanitize_name(meta.company_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        cv_cmd = [
            "latexmk",
            "-pdf",
            "-f",
            "-interaction=nonstopmode",
            f"-output-directory={target_dir}",
            cv_file,
        ]
        print(f"[*] Compiling CV ({cv_file})...")
        subprocess.run(cv_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Fehler beim Kompilieren des CV ({cv_file}): {e}")

    try:
        cl_cmd = [
            "latexmk",
            "-pdf",
            "-f",
            "-interaction=nonstopmode",
            f"-output-directory={target_dir}",
            cl_file,
        ]
        print(f"[*] Compiling Cover Letter ({cl_file})...")
        subprocess.run(cl_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Fehler beim Kompilieren des Anschreibens ({cl_file}): {e}")

    cleanup_target_dir(target_dir)
    print(f"[✓] PDFs und Log-Dateien erfolgreich verarbeitet unter: {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV Pipeline CLI")
    parser.add_argument(
        "source", nargs="?", help="URL der Stellenanzeige oder Pfad zur Textdatei"
    )
    parser.add_argument(
        "--lang",
        choices=["auto", "de", "en"],
        default="auto",
        help="Sprache festlegen: 'auto' (automatisch), 'de' (Deutsch), 'en' (Englisch)",
    )
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
        "--set",
        nargs="+",
        help='Generische Overrides im Format feld=wert (z.B. --set company_name="Familienland GmbH" location="Wien")',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur analysieren, nicht schreiben/kompilieren",
    )

    args = parser.parse_args()

    if not args.source:
        print(
            "[*] Kein Quellpfad angegeben. Starte lokalen Offline-Modus für ALLE Sprach- und Rollen-Varianten..."
        )

        variants = [
            {
                "cv_file": "cv_backend.tex",
                "cl_file": "cover_letter_backend.tex",
                "rolle": r"Senior C++/Python Backend \& Systems Engineer",
                "subheader": r"High-Performance Architectures · Pragmatic Problem Solver · Domain Expert",
            },
            {
                "cv_file": "cv_devops.tex",
                "cl_file": "cover_letter_devops.tex",
                "rolle": r"Senior Cloud Infrastructure \& DevOps Engineer",
                "subheader": r"CI/CD Automation · Release Optimization · Platform Engineering Specialist",
            },
            {
                "cv_file": "cv_mlds.tex",
                "cl_file": "cover_letter_mlds.tex",
                "rolle": r"Senior Machine Learning Engineer \& Data Scientist",
                "subheader": r"AI Infrastructure · RAG Pipelines · Scientific Computing Specialist",
            },
        ]

        target_dir = Path("output") / "Lokale_Rollen_Laeufe"
        target_dir.mkdir(parents=True, exist_ok=True)

        languages_to_run = ["de", "en"] if args.lang == "auto" else [args.lang]

        for current_lang in languages_to_run:
            for v in variants:
                salutation = (
                    "Sehr geehrte Damen und Herren,"
                    if current_lang == "de"
                    else "Dear Recruiting Team,"
                )
                highlight = (
                    "der Modernisierung von Cloud-Infrastrukturen"
                    if current_lang == "de"
                    else "modern cloud infrastructure"
                )

                meta = JobMetadata(
                    company_name="Lokaler_Testlauf",
                    company_size="N/A",
                    contact_person="Recruiting Team",
                    job_title=v["rolle"],
                    location="Österreich",
                    language="Deutsch" if current_lang == "de" else "Englisch",
                    cv_variant=v["cv_file"],
                    cover_letter_variant=v["cl_file"],
                    cv_rolle=v["rolle"],
                    cv_subheader=v["subheader"],
                    salutation=salutation,
                    company_highlight=highlight,
                    fit_score="85%",
                )

                resolved_lang, cv_file, cl_file = resolve_file_variants(
                    meta, current_lang
                )

                print(
                    f"\n[*] Generiere lokalen Lauf [{current_lang.upper()}] für: {v['rolle']}"
                )

                settings_content = render_settings(meta, resolved_lang)
                Path("settings.tex").write_text(settings_content, encoding="utf-8")

                sub_dir = (
                    target_dir
                    / f"{sanitize_name(v['cv_file'].replace('.tex', ''))}_{current_lang}"
                )
                sub_dir.mkdir(parents=True, exist_ok=True)

                if Path(cv_file).exists():
                    try:
                        cv_cmd = [
                            "latexmk",
                            "-pdf",
                            "-f",
                            "-interaction=nonstopmode",
                            f"-output-directory={sub_dir}",
                            cv_file,
                        ]
                        print(f"[*] Kompiliere CV ({cv_file})...")
                        subprocess.run(cv_cmd, check=True)
                        print(f"[✓] Erfolgreich: {cv_file} -> {sub_dir}")
                    except subprocess.CalledProcessError as e:
                        print(f"[!] Fehler beim Kompilieren von {cv_file}: {e}")

                if Path(cl_file).exists():
                    try:
                        cl_cmd = [
                            "latexmk",
                            "-pdf",
                            "-f",
                            "-interaction=nonstopmode",
                            f"-output-directory={sub_dir}",
                            cl_file,
                        ]
                        print(f"[*] Kompiliere Anschreiben ({cl_file})...")
                        subprocess.run(cl_cmd, check=True)
                        print(f"[✓] Erfolgreich: {cl_file} -> {sub_dir}")
                    except subprocess.CalledProcessError as e:
                        print(f"[!] Fehler beim Kompilieren von {cl_file}: {e}")

                cleanup_target_dir(sub_dir)

        print(
            f"\n[✓] Alle Offline-Läufe (DE/EN) abgeschlossen. Ausgabeordner: {target_dir}"
        )
        sys.exit(0)

    order = args.cascade
    if args.provider != "cascade":
        order = [args.provider]

    run_pipeline(
        args.source,
        cascade_order=order,
        lang_setting=args.lang,
        dry_run=args.dry_run,
        overrides=args.set,
    )
