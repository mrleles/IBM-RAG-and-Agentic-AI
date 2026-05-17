'''
%pip install crewai==0.186.1
%pip install crewai-tools==0.71.0
%pip install langchain-community==0.3.29
%pip install langchain-huggingface==0.3.1
%pip install sentence-transformers==5.1.0
%pip install litellm
'''

from crewai import Agent, Task, Crew, Process
from crewai import LLM
from crewai_tools import PDFSearchTool, SerperDevTool
import litellm
litellm.ssl_verify = False

llm = LLM(
    model="watsonx/ibm/granite-4-h-small",
    base_url="url",
    project_id="project_id",
)

import os
os.environ['SERPER_API_KEY'] = "api_key"
web_search_tool = SerperDevTool()

