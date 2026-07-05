#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run ML pipeline to generate model files (.pkl)
python src/prediction.py

# Initialize database
python -c "from app import initialize_database; initialize_database()"
