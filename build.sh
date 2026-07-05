#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run Machine Learning Pipeline to generate model artifacts (.pkl files)
python src/data_preprocessing.py
python src/feature_engineering.py
python src/prediction.py

# Initialize database
python -c "from app import initialize_database; initialize_database()"
