# Commands
- Build Docker Image
  - `docker build -t sentiment-api .`
- Run the Docker Container
  - `docker run -d -p 8000:8000 --name sentiment-container sentiment-api`
- Test the Live API
  - `python test_client.py`
