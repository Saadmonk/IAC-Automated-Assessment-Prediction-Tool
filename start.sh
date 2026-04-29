#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

. venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Open http://localhost:8000 in your browser."
echo

python app.py
