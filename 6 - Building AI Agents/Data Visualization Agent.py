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

df.head(5)
df.info()

credentials = {
	"url": "https://us-south.ml.cloud.ibm.com",
}
model_id = 'ibm/granite-3-2-8b-instruct'

params = {
	GenParams.MAX_NEW_TOKENS: 256,
	GenParams.TEMPERATURE: 0,
}

project_id = "skills-network"

space_id = None
verify = False

model = Model(
	model_id=model_id,
	credentials=credentials,
	params=params,
	project_id=project_id,
	space_id=space_id,
	verify=verify
)

llm = WatsonxLLM(model=model)

agent = create_pandas_dataframe_agent(
	llm, df, verbose=False,
	return_intermediate_steps=True,
	handle_parsing_errors="true"
)

response = agent.invoke("how many rows of data are in this file?")
response['output']
len(df)
response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n') # reveal the underlying commands

response = agent.invoke("Give me all the data where student's age is over 18 years old.")
response = agent.invoke("Generate a bar chart to plot the gender count.")
response = agent.invoke("Generate a pie chart to display average value of Walc for each gender.")
