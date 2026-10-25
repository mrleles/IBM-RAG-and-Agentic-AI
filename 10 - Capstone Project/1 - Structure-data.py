'''
capture
pip install numpy==2.3.4
pip install matplotlib==3.10.7
pip install ibm-watsonx-ai==1.4.7
'''

import numpy as np
import matplotlib.pyplot as plt
import json
import os

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import (
    ModelTypes,
    DecodingMethods,
)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# !wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/1r_mM6ZPYNxcFv65QkzubA/California-Culinary-Map.txt

file_path = "California-Culinary-Map.txt"

with open(file_path, 'r') as file:
    data = file.read()

restaurant_list = data.split('\n\n')

restaurant_list = restaurant_list[1:]

def llm_model(system_msg, prompt_txt):
    model_id = "ibm/granite-4-h-small"
    project_id="skills-network"
    credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")

    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt}
    ]

    response = model.chat(messages=messages)
    output_text = response["choices"][0]["message"]["content"]

    return output_text

import time

def safe_llm_call(system_msg, prompt_txt, retries=3):
    for i in range(retries):
        try:
            return llm_model(system_msg, prompt_txt)
        except Exception:
            time.sleep(2)
    return "Failed after retries"

EXAMPLE_RESTAURANT_PARAGRAPH = restaurant_list[1]
EXAMPLE_OUTPUT = """
    {{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier."
    "shortcomings": []
    }}
"""

def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg =f"""
    You are a data-extraction assistant. You read restaurant descriptions and return ONLY a valid JSON object. No explanations, no markdown, no code fences.
    Use null for unknown rating, price_range or vibe, and [] for empy lists.
"""
    base_user_prompt = f"""
    Task:
    Extract the following fields from the restaurant description and return them as JSON:
    name (str), location (str), type (str), food_style (str), rating (float or null), price_range (int 1-4 or null),
    signatures (list of str), vibe (str or null),
    environment (str), shortcomings (list of str).
    Do not invent information that is not in the text.

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {EXAMPLE_OUTPUT}

"""
    return base_system_msg, base_user_prompt

restaurant_paragraph = restaurant_list[0]
base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(restaurant_paragraph=restaurant_paragraph)

test_response = llm_model(system_msg=base_system_msg, prompt_txt=base_user_prompt)
print(test_response)

from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)

try:
    restaurant_data = Restaurant.model_validate_json(test_response)
    print(f"Success! Validated: {restaurant_data.name}")
except ValidationError as e:
    print(f"Validaion failed: {e.json()}")

def JSON_auto_repair_prompts(candidate_json_output, error_message):
    auto_repair_system_msg = """
    You fix invalid JSON. Return ONLY the corrected JSON object, with no explanations, no markdown and no code fences.
"""

    auto_repair_prompt = f"""
    The JSON below failed validation against the required schema:
    name, location, type, food_style (str); rating (float|null); price_range (int|null); signatures (list[str]); vibe (str|null); environment (str); shortcomings (list[str]).

    Invalid JSON:
    {candidate_json_output}

    Validation error:
    {error_message}

    Return the corrected JSON.
"""
    return auto_repair_system_msg, auto_repair_prompt

structured_restaurant_lists = []
for i, restaurant_paragraph in enumerate(restaurant_list):
    base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(restaurant_paragraph=restaurant_paragraph)
    candidate_json_output = safe_llm_call(system_msg=base_system_msg, prompt_txt=base_user_prompt)

    while(i < len(restaurant_list)):
        try:
            candidate_json_output = Restaurant.model_validate_json(test_response)
        except ValidationError as e:
            auto_repair_system_msg, auto_repair_prompt = JSON_auto_repair_prompts(candidate_json_output=candidate_json_output, error_message=e)
    if (i+1)%20 == 0:
        print(f'{i+1} out of {len(restaurant_list)} is done')

print('ALL DONE!')