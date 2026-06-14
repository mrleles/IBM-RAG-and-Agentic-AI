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

# Multi Agent with GroupChat
from autogen import ConversableAgent, GroupChat, GroupChatManager

llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini"
        }
    ]
}

lesson_planner = ConversableAgent(
    name="planner_agent",
    system_message="Create a short lesson plan for 4th graders.",
    description="Makes lesson plans.",
    llm_config=llm_config
)

lesson_reviewer = ConversableAgent(
    name="reviewer_agent",
    system_message="Review a plan and suggest up to 3 brief edits.",
    description="Reviews lesson plans and suggests edits.",
    llm_config=llm_config
)

teacher = ConversableAgent(
    name="teacher_agent",
    system_message="Suggest a topic and reply DONE when satisfied.",
    llm_config=llm_config,
    is_termination_msg=lambda x: "DONE" in (x.get("content", "") or "").upper()
)

groupchat = GroupChat(
    agents=[teacher, lesson_planner, lesson_reviewer],
    speaker_selection_method="auto"
)

manager = GroupChatManager(
    name="group_manager",
    groupchat=groupchat,
    llm_config=llm_config
)

teacher.initiate_chat(
    recipient=manager,
    message="Make a simple lesson about the moon.",
    max_turns=6,
    summary_method="reflection_with_llm"
)

from autogen import ConversableAgent, register_function
from typing import Annotated

llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini"
        }
    ]
}

def is_prime(n: Annotated[int, "Positive integer"]) -> str:
    if n < 2:
        return "No"
    for i in range(2, int(n**0.5) + 1):
        if n % i ==0:
            return "No"
    return "Yes"

math_asker = ConversableAgent(
    name="math_asker",
    system_message="Ask whether a number is prime.",
    llm_config=llm_config
)

math_checker = ConversableAgent(
    name="math_checker",
    human_input_mode="NEVER",
    llm_config=llm_config
)

register_function(
    is_prime,
    caller=math_asker,
    executor=math_checker,
    description="Check if a number is prime. Returns Yes or No."
)

math_checker.initiate_chat(
    recipient=math_asker,
    message="Is 72 a prime number?",
    max_turns=2
)

from pydantic import BaseModel
from autogen import ConversableAgent

class TicketSummary(BaseModel):
    customer_name: str
    issue_type: str
    urgency_level: str
    recommended_action: str

llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini"
        }
    ],
    "response_format": TicketSummary
}

support_agent = ConversableAgent(
    name="support_agent",
    system_message=(
        "You are a support assistant. Summarize a customer ticket using:"
        "\n- customer_name"
        "\n- issue_type (e.g. login issue, billing problem, bug report)"
        "\n- urgency_level (Low, Medium, High)"
        "\n- recommended_action"
    ),
    llm_config=llm_config
)

support_agent.initiate_chat(
    recipient=support_agent,
    message="Ticket: John Doe is unable to reset his password and has an important meeting in 30 minutes.",
    max_turns=1
)