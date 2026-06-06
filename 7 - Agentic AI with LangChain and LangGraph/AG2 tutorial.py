# !pip install ag2[openai] python-dotenv | tail -n 1

import os
from dotenv import load_dotenv
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.llm_config import LLMConfig
import json
import time
import random

load_dotenv()

print("AG2 modules imported successfully!")

import logging

logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)

config_list = [
    {
        "model": "gpt-4",
        "api_key": ""
    }
]

config_list = config_list_from_json("config.json")

