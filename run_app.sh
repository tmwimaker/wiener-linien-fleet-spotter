#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -f venv/bin/python ]; then
    echo "Erstelle virtuelle Umgebung in ./venv ..."
    python3 -m venv venv
    echo "Installiere Abhaengigkeiten aus requirements.txt ..."
    venv/bin/pip install --upgrade pip
    venv/bin/pip install -r requirements.txt
fi

venv/bin/python -m streamlit run app/app.py
