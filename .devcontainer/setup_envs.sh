#!/bin/bash

# Exit on error
set -e

echo "Initializing .venv (Main Library)..."
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ./src

echo "Initializing .venv-alibi (Alibi Explainer)..."
python3 -m venv .venv-alibi
.venv-alibi/bin/pip install --upgrade pip
.venv-alibi/bin/pip install -e ./src
.venv-alibi/bin/pip install -r explainers/alibi/requirements.minimal.txt

echo "Initializing .venv-dice (DiCE Explainer)..."
python3 -m venv .venv-dice
.venv-dice/bin/pip install --upgrade pip
.venv-dice/bin/pip install -e ./src
.venv-dice/bin/pip install -r explainers/dice/requirements.minimal.txt

echo "All environments initialized successfully."
