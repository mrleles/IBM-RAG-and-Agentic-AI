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

pdf_search_tool = PDFSearchTool(
    pdf="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/7vgNfis17dQfjHAiIKkBOg/The-Daily-Dish-FAQ.pdf",
    config=dict(
        embedder=dict(
            provider="huggingface",
            config=dict(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
        )
    )
)

agent_centric_agent = Agent(
    role="The Daily Dish Inquiry Specialist",
    goal="""Accurately answer customer questions about The Daily Dish restaurant.
    You must decide whether to use the restaurant's FAQ PDF or a web search to find the best answer.""",
    backstory="""You are an AI assistant for 'The Daily Dish'.
    You have access to two tools: one for searching the restaurant's FAQ document and another for searching the web.
    Your job is to analyze the user's question and choose the most appropriate tool to find the information needed to provide a helpful response.""",
    tools=[pdf_search_tool, web_search_tool],
    verbose=True,
    allow_deletation=False,
    llm=llm
)

agent_centric_task = Task(
    description="Answer the following customer query: '{customer_query}'. "
    "Analyze the question and use the tools at your disposal (PDF search or web search) to find the most relevant information. "
    "Synthesize the findings into a clear and friendly response.",
    expected_output="A comprehensive and well-formatted answer to the customer's query.",
    agent=agent_centric_agent
)

agent_centric_crew = Crew(
    agents=[agent_centric_agent],
    tasks=[agent_centric_task],
    process=Process.sequential,
    verbose=False
)

print("\nWelcome to The Daily Dish Chatbot!")
print("What would like to know? (Type 'exit' to quit)")

while True:
    user_input = input("\nYour question: ").lower()
    if user_input == 'exit':
        print("Thank you for chatting. Have a great day!")
        break

    if not user_input:
        print("Please type a question.")
        continue

    try:
        result_agent_centric = agent_centric_crew.kickoff(inputs={'customer_query': user_input})
        print("\n--- The Daily Dish Assistant ---")
        print(result_agent_centric)
        print("--------------------------------")
    except Exception as e:
        print(f"An error occurred: {e}")

task_centric_agent = Agent(
    role="Customer Service Specialist",
    goal="Provide exceptional customer service by following a multi-step process to answer customer questions accurately.",
    backstory="""You are an AI assistant for 'The Daily Dish'.
    You are an expert at following instructions. You will be given a sequence of tasks to complete.
    For each task, you will be provided with the specific tool needed to accomplish it.
    Your job is to execute each task diligently and pass the results to the next step.""",
    tools=[],
    verbose=True,
    allow_deletation=False,
    llm=llm
)

faq_search_task = Task(
    description="Search the restaurant's FAQ PDF for information relate to the customer's query: '{customer_query}'.",
    expected_output="A snippet of the most relevant information from the PDF, or a statement that the information was not found.",
    tools=[pdf_search_tool],
    agent=task_centric_agent
)

response_drafting_task = Task(
    description="Using the information gathered from the FAQ search, draft a friendly and comprehensive response to the customer's query: '{customer_query}'",
    expected_output="The final, customer-facing response.",
    agent=task_centric_agent,
    context=[faq_search_task]
)

task_centric_crew = Crew(
    agents=[task_centric_agent],
    tasks=[faq_search_task, response_drafting_task],
    process=Process.sequential,
    verbose=True
)

print("\nWelcome to The Daily Dish Chatbot!")
print("What would you like to know? (Type 'exit' to quit)")

while True:
    user_input = input("\nYour question: ").lower()
    if user_input == 'exit':
        print("Thank you for chatting. Have a great day!")
        break

    if not user_input:
        print("Please type a question.")
        continue

    try:
        result_task_centric = task_centric_crew.kickoff(inputs={'customer_query': user_input})
        print("\n--- The Daily Dish Assistant ---")
        print(result_task_centric)
    except Exception as e:
        print(f"An error occurred: {e}")

from crewai.tools import tool
import re

@tool("Add Two Numbers Tool")
def add_numbers(data: str) -> int:
    """
    Extracts and adds integers from the input string.
    Example input: 'add 1 and 2' or '[1,2,3,4]'
    Output: sum of the numbers
    """
    numbers = list(map(int, re.findall(r'-?\d+', data)))
    return sum(numbers)

from functools import reduce
@tool("Multiply Numbers Tool")
def multiply_numbers(data: str) -> int:
    """
    Extracts and multiplies integers from the input string.
    Example input: 'multiply 2 and 3' or '[2,3,4]'
    Output: the product of all numbers found
    """
    numbers = list(maps(int, re.findall(r'-?\d+', data)))
    return reduce(lambda x, y: x * y, numbers, 1)

calculator_agent = Agent(
    role="Calculator",
    goal="Extracts, add, or multiplies numbers when asked, using the Add Two Numbers and Multiply Numbers tools.",
    backstory="An expert at parsing numeric instructions and computing sums or products.",
    tools=[add_numbers, multiply_numbers],
    llm=llm,
    allow_delegation=False
)

calculation_task = Task(
    description="Extract numbers from '{numbers} and either add or multiply them, depending on the natural-language instruction.",
    expected_output="An integer result (sum or product) based on the user's request.",
    agent=calculator_agent
)

crew = Crew(
    agents=[calculator_agent],
    tasks=[calculation_task],
)

result = crew.kickoff(inputs={'numbers': 'please add 4, 5, and 6'})
print("Sum result:", result)