# Official lightweight Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Set working directory inside the container
WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project directories and files
COPY config.py app.py /app/
COPY data/ /app/data/
COPY notebooks/ /app/notebooks/
COPY src/ /app/src/
COPY templates/ /app/templates/
COPY static/ /app/static/
COPY saved_models/ /app/saved_models/

# Pre-run the data loading and training pipeline during image build
# This bakes the trained champion model and initialized sqlite database into the container
RUN python src/data_preprocessing.py && \
    python src/feature_engineering.py && \
    python src/prediction.py && \
    python -c "import app; app.initialize_database()"

# Expose port 5000 for the web application
EXPOSE 5000

# Start the application using waitress production WSGI server
CMD ["python", "app.py"]
