import os
import json
import base64
import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from dotenv import load_dotenv

# Load .env environment variables
load_dotenv()


class FinetunedGeminiClient:
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        self.region = os.getenv("VERTEX_AI_REGION")
        self.endpoint_id = os.getenv("VERTEX_AI_ENDPOINT_ID")

        self.credentials = self._setup_credentials()
        self.endpoint_url = self._build_endpoint_url()

    def _setup_credentials(self):
        """Load service account creds from file path."""
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set")

        return service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    def _build_endpoint_url(self):
        """Vertex AI tuned-model endpoint."""
        return (
            f"https://{self.region}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.region}/endpoints/{self.endpoint_id}:generateContent"
        )

    # -----------------------------
    #  TEXT ONLY MODEL CALL
    # -----------------------------
    def generate_content(
        self,
        prompt: str = "",
        image_path: str = "",
        temperature: float = 0.4,
        max_tokens: int = 1000
    ):
        """Send text + optional image to your Gemini endpoint."""

        parts = []

        # If an image is provided → add fileData
        if image_path:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")

            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": image_b64
                }
            })

        # Add text prompt
        if prompt:
            parts.append({"text": prompt})

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts
                }
            ],
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        }

        # Refresh OAuth2 access token
        self.credentials.refresh(Request())

        headers = {
            "Authorization": f"Bearer {self.credentials.token}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.endpoint_url, json=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                return json.dumps(data, indent=2)

        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

if __name__ == "__main__":
    client = FinetunedGeminiClient()

    # Put ANY image path here
    image_path = r"C:\Users\gsart\Downloads\plantvillage\color\Tomato___Spider_mites Two-spotted_spider_mite\fe7f0ec0-35b0-49b0-836e-1fa902ee7723___Com.G_SpM_FL 8739.JPG"
    output = client.generate_content(
        image_path=image_path,
        prompt="What disease is shown in this image? Answer with only the class name. Give a deatiled explanation after the class name., how to cure it and also certain videos for that.",
    )

    print("\nMODEL OUTPUT:\n")
    print(output)
