# !pip install autogen==0.7 openai==1.64 python-dotenv==1.1.0 | tail -n 1

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from autogen import ConversableAgent, GroupChat, GroupChatManager
from openai import OpenAI
import logging

logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)

client = OpenAI()

code_execution_config = {"use_docker": False}

llm_config = {"config_list": [{"model": "gpt-4", "api_key": None}]}

patient_agent = ConversableAgent(
    name="patient",
    system_message="You describe symptoms and ask for medical help.",
    llm_config=llm_config
)

diagnosis_agent = ConversableAgent(
    name="diagnosis",
    system_message="You analyze symptoms and provide a possible diagnosis. Summarize key points in one response.",
    llm_config=llm_config
)

pharmacy_agent = ConversableAgent(
    name="pharmacy",
    system_message="You recommend medications based on diagnosis. Only respond once.",
    llm_config=llm_config
)

consultation_agent = ConversableAgent(
    name="consultation",
    system_message="You determine if a doctor's visit is required. Provide a final summary with clear next steps. IMPORTANT: End your response with 'CONSULTATION COMPLETE' to signal the end of the conversation.",
    llm_config=llm_config
)

groupchat = GroupChat(
    agents=[diagnosis_agent, pharmacy_agent, consultation_agent],
    messages=[],
    max_round=5,
    speaker_selection_method="round_robin"
)

manager = GroupChatManager(name="manager", groupchat=groupchat)

print("\n Welcome to the AI Healthcare Consultation System!")
symptoms = input("Please describe yours symptoms: ")
print("\n Diagnosing symptoms...")
response = patient_agent.initiate_chat(
    manager,
    message=f"I am feeling {symptoms}. Can you help?"
)