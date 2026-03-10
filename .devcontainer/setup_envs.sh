#!/bin/bash

# Exit on error
set -e

echo "Initializing .venv (Natives Only)..."
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ./src
.venv/bin/pip install -r explainers/native/requirements.minimal.txt

echo "Environment initialized successfully."
