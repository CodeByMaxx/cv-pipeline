#!/bin/bash

# Zielordner definieren (z. B. "build" oder "pdf_output")
TARGET_DIR="build"

# Zielordner erstellen, falls er noch nicht existiert
mkdir -p "$TARGET_DIR"

# Alle .tex-Dateien im aktuellen Ordner durchgehen
for file in *.tex; do
    # Prüfen, ob .tex-Dateien existieren
    [ -e "$file" ] || continue

    echo "=== Kompiliere $file nach $TARGET_DIR/ ==="
    pdflatex -interaction=nonstopmode -output-directory="$TARGET_DIR" "$file"
done

echo "=== Fertig! ==="

# Hilfsdateien sauber im Zielordner aufräumen
rm -f "$TARGET_DIR"/*.log "$TARGET_DIR"/*.aux "$TARGET_DIR"/*.out "$TARGET_DIR"/*.toc
