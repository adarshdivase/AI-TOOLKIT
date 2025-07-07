# Use an official Python runtime as a parent image.
# Python 3.10 is recommended for compatibility with AI libraries.
FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set Hugging Face cache directory to a temporary, writable location
# /tmp is usually always writable in Docker containers.
ENV HF_HOME=/tmp/hf_cache
# Disable symlinks in HF_HOME, which can sometimes cause issues in certain environments
ENV HF_HUB_DISABLE_SYMLINKS_HF_HOME=1

# Explicitly create the cache directory and set full permissions
RUN mkdir -p /tmp/hf_cache && chmod -R 777 /tmp/hf_cache

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Set environment variables
# Use port 7860 which is the default for Hugging Face Spaces
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Expose the port that the application will listen on
EXPOSE $PORT

RUN pip install --no-cache-dir scipy


# Run the application using uvicorn
# Use the PORT environment variable that Hugging Face Spaces provides
CMD ["sh", "-c", "uvicorn merged_backend:app --host 0.0.0.0 --port ${PORT:-7860}"]
