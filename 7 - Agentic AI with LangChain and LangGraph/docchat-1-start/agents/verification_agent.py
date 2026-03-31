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

        **Instruction:**
        - Verify the following answer against the provided context.
        - Check for:
        1. Direct/indirect factual support (YES/NO)
        2. unsupported claims (list any if present)
        3. Contradictions (list any if present)
        4. Relevance to the question (YES/NO)
        - Provide additional details or explanations where relevant.
        - Respond in the exact format specified below without adding any unrelated information.

        **Format:**
        Supported: YES/NO
        Unsupported Claims: [item1, item2, ...]
        Contradictions: [item1, item2, ...]
        Relevant: YES/NO
        Additional Details: [Any extra information or explanations]

        **Respond ONLY with the above format.**
        """
        return prompt

    def parse_verification_response(self, response_text: str) -> Dict:
        """
        parse the LLM' verification response into a structured dictionary.
        """
        try:
            lines = response_text.split('\n')
            verification = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().capitalize()
                    value = value.strip()
                    if key in {"Supported", "Unsupported claims", "Contradictions", "Relevant", "Additional details"}:
                        if key in {"Unsupported claims", "Contradictions"}:
                            if value.startswith('[') and value.endswith(']'):
                                items = value[1:-1].split(',')