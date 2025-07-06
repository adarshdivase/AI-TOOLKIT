    # Use an official Python runtime as a parent image.
    # Python 3.10 or 3.9 is recommended for compatibility with AI libraries.
    FROM python:3.10-slim-buster

    # Set the working directory in the container
    WORKDIR /app

    # Set Hugging Face cache directory to a writable location within /app
    ENV HF_HOME=/app/hf_cache

    # Copy the requirements file into the container at /app
    COPY requirements.txt .

    # Install any needed packages specified in requirements.txt
    # Use --no-cache-dir to save space
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy the rest of the application code to the container
    COPY . .

    # Expose the port that the application will listen on.
    ENV PORT=8000
    EXPOSE 8000

    # Run the application using uvicorn.
    CMD ["uvicorn", "merged_backend:app", "--host", "0.0.0.0", "--port", "8000"]
    
