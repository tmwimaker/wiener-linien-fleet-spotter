#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -f venv/bin/python ]; then
    # torch<2.6 (siehe requirements.txt) hat nur Wheels fuer Python 3.10-3.12.
    PYVER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    case "$PYVER" in
        3.10|3.11|3.12) ;;
        *)
            echo "[Fehler] Python $PYVER wird nicht unterstuetzt - bitte Python 3.10, 3.11 oder 3.12 verwenden."
            echo "         (torch<2.6 hat fuer Python 3.13+ kein passendes Wheel.)"
            echo "         Tipp: 'python3.12 -m venv venv' o. ae., dann dieses Skript erneut starten."
            exit 1
            ;;
    esac
    echo "Erstelle virtuelle Umgebung in ./venv (Python $PYVER) ..."
    python3 -m venv venv
    echo "Installiere Abhaengigkeiten aus requirements.txt ..."
    venv/bin/pip install --upgrade pip
    venv/bin/pip install -r requirements.txt
fi

venv/bin/python -m streamlit run app/app.py
