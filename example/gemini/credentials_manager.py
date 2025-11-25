import os
import json
import logging
from typing import Optional

from google import genai
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import google.auth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# Credentials Manager
# =========================
class CredentialsManager:
    """Manages GCP credentials and authentication using service account"""

    def __init__(self, credentials_path: Optional[str] = None):
        self.credentials_path = credentials_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        self.project_id = None
        self.credentials = None
        self.setup_credentials()

    def setup_credentials(self):
        """Setup GCP credentials using service account JSON"""
        if self.credentials_path and os.path.exists(self.credentials_path):
            try:
                self.credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )

                with open(self.credentials_path, 'r') as f:
                    creds_info = json.load(f)
                    self.project_id = creds_info.get('project_id')

                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path

                logger.info(f"✅ Service account authentication successful")
                logger.info(f"✅ Project ID: {self.project_id}")
                logger.info(f"✅ Service account: {creds_info.get('client_email', 'N/A')}")

            except Exception as e:
                logger.error(f"❌ Error loading service account credentials: {e}")
                raise ValueError(f"Failed to load service account from {self.credentials_path}: {e}")
        else:
            try:
                self.credentials, project = google.auth.default(
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                self.project_id = project or os.getenv('GOOGLE_CLOUD_PROJECT')

                if self.project_id:
                    logger.info(f"✅ Using default credentials for project: {self.project_id}")
                else:
                    raise ValueError("No project ID found in default credentials")

            except Exception as e:
                logger.error(f"❌ Error with default credentials: {e}")
                raise ValueError("No valid GCP credentials found. Provide service account JSON or set default credentials")

    def get_authenticated_client(self):
        """Get properly authenticated genai client"""
        if self.credentials:
            if not self.credentials.valid:
                self.credentials.refresh(Request())

            client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=os.getenv('GOOGLE_CLOUD_REGION', 'us-central1'),
                credentials=self.credentials
            )
        else:
            client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')
            )

        return client

