#!/usr/bin/env bash
# Uruchamia cały system od zera: środowisko → baza → model → API z panelem.
#
# Użycie (z dowolnego miejsca):
#   ./run.sh              # panel pod http://127.0.0.1:8000/
#   PORT=8080 ./run.sh    # inny port
set -euo pipefail

cd "$(dirname "$0")"
PORT="${PORT:-8000}"

echo "== 1/4 Środowisko Pythona (.venv)"
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -r requirements.txt
    echo "Utworzono .venv i zainstalowano biblioteki z requirements.txt"
else
    echo ".venv już istnieje (po zmianie requirements.txt: .venv/bin/pip install -r requirements.txt)"
fi
PY=.venv/bin/python

echo "== 2/4 Import wynajmów z CSV do bazy SQLite"
"$PY" scripts/import_csv.py

echo "== 3/4 Uczenie modelu i porównanie z metodą bazową"
"$PY" -m src.model.train

echo "== 4/4 Start API i panelu: http://127.0.0.1:${PORT}/  (dokumentacja API: /docs, zatrzymanie: Ctrl+C)"
exec "$PY" -m uvicorn src.api:app --port "$PORT"
