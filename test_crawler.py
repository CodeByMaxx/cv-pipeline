import sys
import trafilatura
from playwright.sync_api import sync_playwright


def extract_with_trafilatura(url: str) -> str:
    """Schneller Weg für statische Seiten."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(
            downloaded, include_links=False, include_comments=False
        )
        if text:
            return text
    return ""


def extract_with_playwright(url: str) -> str:
    """Fallback für dynamische Seiten (JS-Rendering)."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        # Warte bis die Seite geladen ist
        page.goto(url, wait_until="networkidle", timeout=30000)

        # Extrahiere den sichtbaren Text des Body-Tags
        body_text = page.inner_text("body")
        browser.close()
        return body_text


def test_fetch_job(url: str):
    print(f"=== Teste URL: {url} ===")

    # Versuch 1: Trafilatura
    print("[1/2] Versuche schnelle Extraktion via Trafilatura...")
    text = extract_with_trafilatura(url)

    # Versuch 2: Playwright Fallback
    if not text or len(text) < 200:
        print(
            "[!] Trafilatura lieferte zu wenig Text. Wechsle zu Playwright (Headless Browser)..."
        )
        text = extract_with_playwright(url)

    print("\n--- ERGEBNIS (Erste 1000 Zeichen) ---")
    print(text[:1000])
    print("\n-------------------------------------")
    print(f"Gesamtlänge des extrahierten Texts: {len(text)} Zeichen")

    return text


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_url = sys.argv[1]
    else:
        # Beispiel-URL zum Testen
        target_url = "https://www.metacareers.com/"
        print("Keine URL übergeben. Bitte rufe das Skript so auf:")
        print("python test_crawler.py <JOB_URL>")
        sys.exit(1)

    test_fetch_job(target_url)
