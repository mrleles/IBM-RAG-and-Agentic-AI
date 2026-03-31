from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from typing import Dict, List
from langchain.schema import Document
from config.settings import settings
import json

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                  )
client = APIClient(credentials)


class ResearchAgent:
    def __init__(self):
        self.model = ModelInference(
            model_id="meta-llama/llama-3-2-90b-vision-instruction",
            credentials=credentials,
            project_id="skills-network",
            params={
                "max_tokens": 300,
                "temperature: 0.3,"
            }
        )
        print("ModelInference initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, question: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        prompt = f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.

        **Question:** {question}
        **Context:**
        {context}

        **Provide your answer below:**
        """
        return prompt

    