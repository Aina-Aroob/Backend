FROM python:3.10.0-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Verify model exists and can be loaded
RUN python -c "import tensorflow as tf; model=tf.keras.models.load_model('glasses_model.h5'); print('Model verified successfully')"

# Start the application
CMD gunicorn app:app --bind 0.0.0.0:$PORT 