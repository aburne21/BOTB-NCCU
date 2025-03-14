# Use official Python runtime
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy dependencies first (better for caching layers)
fastapi
uvicorn
boto3
scikit-learn


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including `app/` folder)


# Expose the API port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
