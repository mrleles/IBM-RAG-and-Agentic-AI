'''
%%capture
!pip install langchain-openai==0.3.27
!pip install langgraph==0.6.6
!pip install pygraphviz==1.14
'''

from langgraph.graph import StateGraph, END, START
from langgraph.types import Send

from typing import TypeDict, Annotated, list, Literal
from pydantic import BaseModel, Field
import operator
from pprint import pprint
from IPython.display import Image, display

from langgchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# !pip install litellm

import litellm
litellm.ssl_verify = False