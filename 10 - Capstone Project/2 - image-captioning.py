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

    return pass