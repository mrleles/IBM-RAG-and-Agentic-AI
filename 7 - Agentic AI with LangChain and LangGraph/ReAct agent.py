"""
pip install -U langgraph langchain-openai
pip install langgraph==0.3.34 langchain-openai==0.3.14 langchainhub==0.1.21 langchain==0.3.24 pygraphviz==1.14 langchain-community==0.3.23
"""

import warnings
warnings.filterwarnings('ignore')

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
import os
import json

os.environ["TAVILY_API_KEY"] = "your-api-key"

search = TavilySearchResults()

@tool
def search_tool(query: str):
	"""
	Search the web for information using Tavily API.

	:param query The search query string
	:return: Search results related to the query
	"""
	return search.invoke(query)

search_tool.invoke("What's the weather like in Tokyo today?")

@tool
def recommend_clothing(weather: str) -> str:
	"""
	Returns a clothing recommendation based on the provided weather description.

	This function examines the input string for specific keywords or temperature indicators (e.g., "snow", "freezing", "rain", "85°F") to suggest appropriate attire. It handles common weather conditions like snow, rain, heat, and cold by providing simple and practical clothing advice.

	:param weather: A brief description of the weather (e.g., "overcast, 64.9°F")
	:return: A string with clothing recommendations suitable for the weather
	"""
	weather = weather.lower()
	if "snow" in weather or "freezing" in weather:
		return "Wear a heavy coat, gloves, and boots."
	elif "rain" in weather or "wet" in weather:
		return "Bring a raincoat and waterproof shoes."
	elif "hot" in weather or "85" in weather:
		return "T-shirt, shorts, and sunscreen recommended."
	elif "cold" in weather or "50" in weather:
		return "Wear a warm jacket or sweater."
	else:
		