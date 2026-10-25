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