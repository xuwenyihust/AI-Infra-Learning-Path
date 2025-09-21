# Stage 1: Use the official Python image as a base
FROM python:3.10-slim

# Stage 2: Set the working directory inside the container
WORKDIR /app

# Stage 3: Copy the requirements file into the container
# We copy this first to leverage Docker's layer caching.
# Dependencies change less often than source code.
COPY requirements.txt .

# Stage 4: Install the Python dependencies
# --no-cache-dir reduces image size.
RUN pip install --no-cache-dir -r requirements.txt

# Stage 5: Copy the application code and the trained model into the container
COPY app.py .
COPY sentiment_transformer.pth .

# Stage 6: Expose the port that the FastAPI application will run on
EXPOSE 8000

# Stage 7: Define the command to run the application using uvicorn
# This is the command that will be executed when the container starts.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

