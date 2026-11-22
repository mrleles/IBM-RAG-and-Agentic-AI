'''
pip install numpy==2.3.4
pip install matplotlib==3.10.7
pip install ibm-watsonx-ai==1.4.7
'''

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from PIL import Image

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

'''
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hpTjb6liKBLVHQK0UgMi5A/Recipes.json
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fQUs9wQ6aB6ts6fmkD2V2w/Synthetic-User-Reviews.json
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip
'''

import zipfile
with zipfile.ZipFile("synthetic-recipe-images.zip", 'r') as zip_ref:
    zip_ref.extractall()

with open('Recipes.json', 'r') as json_file:
    recipe_data = json.load(json_file)
for key, value in recipe_data[0].items:
    print(f"{key} ({type(value)}): {value}")

path = 'synthetic_recipe_images/recipe1.png'
Image.open(path)

# 2 - Define the vision LLM with LLaMA
import base64

def vision_llm(system_msg, prompt_txt, image_path):
    model_id = 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8'
    project_id = "skills-network"
    credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
    generate_params = {"max_tokens": 300}

    model = ModelInference(
        project_id=project_id,
        model_id=model_id,
        credentials=credentials,
        params=generate_params
    )

    with open(image_path, "rb") as image:
        string_image = base64.b64encode(image.read()).decode('utf-8')

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt}
    ]

    response = model.chat(messages=messages)
    return response

# 3 - Design and Validate prompts for the vision LLM

def image_caption_prompt_template(food_name):
    image_caption_system_msg = "You are an assistant that generates useful and consistent image caption"
    image_caption_prompt_txt = f"Describe the food {food_name}. Focus on relevant visual details, such as ingredients, cooking style, and presentation, while avoiding unnecessary or speculative information."

    return image_caption_system_msg, image_caption_prompt_txt

system_msg, prompt_txt = image_caption_prompt_template(recipe_data[0]['name'])

response = vision_llm(system_msg, prompt_txt, path)
print(response)

for i in range(len(recipe_data)):
    if (i+1)%20 == 0:
        print(f'{i+1} out of {len(recipe_data)} is done')

    system_msg, prompt_txt = image_caption_prompt_template(recipe_data[i]['name'])

    path = f'synthetic_recipe_images/recipe{i+1}.png'
    response = vision_llm(system_msg, prompt_txt, path)

    recipe_data[i]['image_description'] = response
print('ALL DONE!')

filename = 'augmented_food_recipe.json'
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(recipe_data, f, indent=4)