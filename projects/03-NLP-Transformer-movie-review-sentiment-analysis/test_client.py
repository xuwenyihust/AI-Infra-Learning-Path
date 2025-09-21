import requests
import json

# The URL where your FastAPI application is running
API_URL = "http://127.0.0.1:8000/predict"

# A few example movie reviews to test
reviews = [
    "This is one of the best films I have ever seen. The acting was incredible and the story was gripping.",
    "A complete waste of time. The plot was nonsensical and the characters were boring.",
    "It was an okay movie, not great but not terrible either.",
    "I was on the edge of my seat the entire time! A must-watch thriller."
]


def test_review(review_text):
    """Sends a single review to the API and prints the response."""
    try:
        response = requests.post(API_URL, json={"text": review_text})
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        print(f"Review: '{review_text}'")
        print(f"Prediction: {response.json()}")
        print("-" * 30)

    except requests.exceptions.RequestException as e:
        print(f"Could not connect to the API at {API_URL}")
        print(f"Error: {e}")
        print("Please ensure the Docker container is running.")


if __name__ == "__main__":
    for review in reviews:
        test_review(review)

