import os
import json
import requests
from typing import Optional
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()


class FinetunedGeminiClient:
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        self.region = os.getenv("VERTEX_AI_REGION")
        self.endpoint_id = os.getenv("VERTEX_AI_ENDPOINT_ID")

        self.credentials = self._setup_credentials()
        self.endpoint_url = self._build_endpoint_url()

    def _setup_credentials(self):
        """Load service account credentials from a file path."""
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set")

        return service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    def _build_endpoint_url(self):
        """Construct the Vertex AI endpoint URL for generateContent."""
        return (
            f"https://{self.region}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.region}/endpoints/{self.endpoint_id}:generateContent"
        )

    def generate_content(self, prompt: str, temperature: float = 0.4, max_tokens: int = 1000):
        """Generate text from your Vertex AI fine-tuned Gemini model."""
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
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
            response_data = response.json()

            try:
                return response_data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                return json.dumps(response_data, indent=2)

        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")


# Example usage
if __name__ == "__main__":
    client = FinetunedGeminiClient()
    output = client.generate_content("Explain the theory of relativity in simple terms.")
    print(output)
