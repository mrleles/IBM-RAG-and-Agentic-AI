!pip install --user "ibm-watsonx-ai==0.2.6"
!pip install --user "langchain==0.1.16"
!pip install --user "langchain-ibm==0.1.14"
!pip install --user "langchain-experimental==0.0.57"
!pip install --user "matplotlib==3.8.4"
!pip install --user "seaborn==0.13.2"

def warn(*args, **kargs):
	pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZNoKMJ9rssJn-QbJ49kOzA/student-mat.csv")