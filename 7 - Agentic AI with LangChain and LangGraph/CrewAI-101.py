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

from crewai import Agent

research_agent = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge information and insights on any subject with comprehensive anlysis',
    backstory="""You are an expert researcher with extensive experience in gathering, analyzing, and synthesizing information across multiple domains.
    Your analytical skills allow you to quickly identify key trends, separate fact from opinion, and produce insightful reports on any topic.
    You excel at finding reliable sources and extracting valuable information efficiently.""",
    verbose=True,
    allow_delegation=False,
    llm = llm,
    tools=[SerperDevTool()]
)

writer_agent = Agent(
    role='Tech Content Strategist',
    goal='Craft well-structured and engaging content based on research findings',
    backstory="""You are a skilled content strategist known for translating
    complex topics into clear and compelling narratives. Your writing makes
    information accessible and engaging for a wide audience.""",
    verbose=True,
    llm = llm,
    allow_delegation=True,
)

research_task = Task(
    description="Analyze the major {topic}, identifying key trends and technologies. Provide a detailed report on their potential impact.",
    agent-research_agent,
    expected_output="A detailed report on {topic}, including trends, emergin technologies and their impact."
)

writer_task = Task(
    description="Create an engaging blog post based on the research findings about {topic}. Tailor the content for a tech-savvy audience, ensuring clarity and interest."
    agent=writer_agent,
    expected_output="A 4-paragraph blog post on {topic}, written clearly and engagingly for tech enthusiasts."
)

from crewai import Crew, Process

crew = Crew(
    agents=[research_agent, writer_agent],
    tasks=[research_task, writer_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff(inputs={"topic": "Latest Generative AI breakthroughs"})

print(type(result))
print(result)

final_output = result.raw
tasks_outputs = result.tasks_output

print("Task Description", tasks_outputs[0].description)
print("Output of research task ",tasks_outputs[0])

print("We can get the agent for researcher task:  ",tasks_outputs[0].agent)
print("We can get the agent for the writer task: ",tasks_outputs[1].agent)

token_count = result.token_usage.total_tokens
prompt_tokens = result.token_usage.prompt_tokens
completion_tokens = result.token_usage.completion_tokens

print(f"Total tokens used: {token_count}")
print(f"Prompt tokens: {prompt_tokens} (used for instructions to the model)")
print(f"Completion tokens: {completion_tokens} (generated in response)")

post_agent = Agent(
    role="Social Media Agent",
    goal="Curates a summary and a short-form version (such as tweets or LinkedIn posts)",
    backstory="You are a seasoned social media professional proeminent in writing and analysing trends",
    verbose=True,
    llm=llm,
    allow_delegation=True,
)

post_task = Task(
    description=(
        "Summarize the blog post about {topic} into 2–3 engaging social media posts "
        "suitable for platforms like LinkedIn or Twitter. Make sure the tone is informative, "
        "professional, and encourages further reading."
    ),
    agent=social_agent,
    expected_output="A series of 2–3 well-written social posts highlighting the key insights from the blog content."
)

post_crew = Crew(
    agents=[research_agent, writer_agent, post_agent],
    tasks=[research_task, writer_task, post_task],
    process=Process.sequential,
    verbose=True
)