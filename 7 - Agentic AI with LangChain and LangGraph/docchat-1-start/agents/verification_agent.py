import json  # Import for JSON serialization
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from typing import Dict, List
from langchain.schema import Document

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                  )
client = APIClient(credentials)

class VerificationAgent:
    def __init__(self):
        """
        Initialize the verification agent with the IBM WatsonX ModelInference.
        """
        # Initialize the WatsonX modelInference
        print("Initializing VerificationAgent with IBM WatsonX ModelInference...")
        self.model = ModelInference(
            model_id="ibm/granite-4-h-small",
            credentials=credentials,
            project_id="skills-network",
            params={
                "max_token": 200,
                "temperatura": 0.0,
            }
        )
        print("ModelInference initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        """
        Satinize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, answer: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to verify the answer against the context.
        """
        prompt = f"""
        you are an AI assistant designed to verify the accuracy and relevance of answers based on the provided context.
        