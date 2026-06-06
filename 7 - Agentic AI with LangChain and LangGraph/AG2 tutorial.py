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

# Conversable Agent
from autogen import ConversableAgent

llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini"
        }
    ]
}

student = ConversableAgent(
    name="student",
    system_message="You are a curious student. You ask clear, specific questions to learn new concepts.",
    human_input_mode="NEVER",
    llm_config=llm_config
)

tutor = ConversableAgent(
    name="tutor",
    system_message="You are a helpful tutor who provides clear and concise explanations suitable for a beginner.",
    human_input_mode="NEVER",
    llm_config=llm_config
)

chat_result = student.initiate_chat(
    recipient=tutor,
    message="Can you explain what a neural network is?",
    max_turns=2,
    summary_method="reflection_with_llm"
)

print("\nFinal Summary:")
print(chat_result.summary)

# Built-in Agent Types
# AssistantAgent and UserProxyAgent

# !pip install matplotlib numpy pyautogen

from autogen import AssistantAgent, User ProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini"
        }
    ]
}

assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant who writes and explains Python code clearly.",
    llm_config=llm_config
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(work_dir="coding", timeout=30),
    },
)

chat_result = user_proxy.initiate_chat(
    recipient=assistant,
    message="Plot a sine wave using matplotlib from -2π to 2π and save the plot as sine wave.png",
    max_turns=4,
    summary_method="reflection_with_llm"
)

from IPython.display import Image, display
import os

image_path = "coding/sine_wave.png"
if os.path.exists(image_path):
    display(Image(filename=image_path))
else:
    print("Plot not found.")

print("\nFinal Summary:")
print(chat_result.summary)

# Human-in-the-Loop

from autogen import ConversableAgent
import random

llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini"
        }
    ]
}

triage_system_message = """
You are a bug triage assistant. You will be given bug report summaries.

For each bug:
- If it is urgent (e.g., 'crash', 'security', or 'data loss' is mentioned), escalate it and ask the human agent for confirmation.
- If it seems minor (e.g., cosmetic, typo), suggest closing it but still ask for human review.
- Otherwise, classify it as medium priority and ask the human for review.

Once all bugs are processed, summarize what was escalated, closed, or marked as medium priority.
End by saying: "You can type exit to finish."
"""

triage_bot = ConversableAgent(
    name="triage_bot",
    system_message=triage_system_message,
    llm_config=llm_config
)

human = ConversableAgent(
    name="human",
    human_input_mode="ALWAYS",
)

BUGS = [
    "App crashes when opening user profile.",
    "Minor UI misalignment on settings page.",
    "Password reset email not sent consistently.",
    "Typo in the About Us footer text.",
    "Database connection timeout under heavy load.",
    "Login form allows SQL injection attack.",
]

random.shuffle(BUGS)
selected_bugs = BUGS[:3]

initial_prompt = (
    "Please triage the following bug reports one by one:\n\n" +
    "\n".join([f"{i+1}. {bug}" for i, bug in enumerate(selected_bugs)])
)

human.initiate_chat(
    recipient=triage_bot,
    message=initial_prompt,
)