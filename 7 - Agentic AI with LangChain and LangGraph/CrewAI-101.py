'''
%pip install langchain==0.3.20 | tail -n 1
%pip install crewai==0.80.0 | tail -n 1
%pip install langchain-community==0.3.19 | tail -n 1 
%pip install crewai-tools==0.38.0 | tail -n 1
%pip install databricks-sdk==0.57.0| tail -n 1
'''

import os
os.environ['SERPER_API_KEY'] = "paste-api-here"

from crewai_tools import SerperDevTool

search_tool = SerperDevTool()
print(type(search_tool))

search_query = "Últimas sobre São Mateus ES"
search_results = search_tool.run(query=search_query)

print(f"Search Results for '{search_results}':\n")

print("Keys of search results", search_results.keys())

from crewai import LLM

llm = LLM(
    model="watsonx/meta-llama/llama-3-3-70b-instruct",
    base_url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    max_tokens=2000,
)

