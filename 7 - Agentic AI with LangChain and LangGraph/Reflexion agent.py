"""
pip install langchain-openai==0.3.10
pip install langchain==0.3.21
pip install openai==1.68.2
pip install langchain-community==0.2.1
pip install --upgrade langgraph
pip install langchain_community==0.3.24
"""

import os
import json
import getpass
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatpromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph

def _set_if_undefined(var: str) -> None:
	if os.environ.get(var):
		return
	os.environ[var] = getpass.getpass(var)
_set_if_undefined("TAVILY_API_KEY")

tavily_tool=TavilySearchResult(max_results=1)
sample_query = "healthy breakfast recipes"
search_results = tavily_tool.invoke(sample_query)
print(search_results)

llm = ChatOpenAI(model="gpt-4.1-nano")
question="Any ideas for a healthy breakfast"
response=llm.invoke(question).content
print(response)

prompt_template = ChatPromptTemplate.from_messages([
	("system",
		"""you are Dr. Paul Saladino, "Carnivore MD", advocating for animal-based nutrition and challenging plant-centric dietary dogma. Focus on the superior bioavailability of animal foods and he potential toxicity of plant compounds such as oxalates, lectins, and phytates.
	
		Your response must follow these steps:
		1. {first_instruction}
		2. Present the evolutionary and bichemical rationale for animal-based nutrition, emphasizing nutrient density, bioavailability, and the absence of plant antinutrients.
		3. Challenge conventional "plants are healthy" narratives with mechanistic evidence about plant defense compounds and their impact on human physiology.
		4. Reflect and critique your answer. Be rigorous about evolutionary consistency and nutrient science.
		5. After the reflection, **list 1-3 search queries separately** for researching antinutrients, bioavailability studies, or evolutionary nutrition. Do not include them inside the reflection.
	
		Focus on: organ meats as superfoods, the carnivore elimination protocol, and why "fiber is unnecessary" based on human evolutionary biology.
		"""),
	MessagesPlaceholder(variable_name="messages"),
	(
		"system",
		"Answer the user's question above using the required format, emphasizing the superiority of animal-based nutrition."
		),
	])

first_responder_prompt = prompt_template.partial(first_instruction="Provide a detailed ~250 word answer")
temp_chain = first_responder_prompt | llm
response = temp_chain.invoke({"messages": [HumanMessage(content=question)]})
print(response.content)

class Reflection(BaseModel):
	missing: str = Field(description="What information is missing")
	superfluous: str = Field(description="What information is unnecessary")

class AnswerQuestion(BaseModel):
	answer: str = Field(description="Main response to the question")
	reflection: Reflection = Field(description="Self-critique of the answer")
	search_queries: List[str] = Field(description="Queries for additional research")

initial_chain = first_responder_prompt | llm.bind_tools(tools=[AnswerQuestion])
response = initial_chain.invoke({"messages": [HumanMessage(question)]})

answer_content = response.tools_calls[0]['args']['answer']
reflection_content = response.tools_calls[0]['args']['reflection']

response_list=[]
response_list.append(HumanMessage(content=question))
response_list.append(response)

tool_call = response.tools_calls[0]
search_queries = tool_call["args"].get("search_queries", [])
print(search_queries)

tavily_tool=TavilySearchResults(max_results=3)

def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
	last_ai_message = state[-1]
	tool_messages = []
	for tool_call in last_ai_message.tool_calls:
		if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
			call_id = tool_call["id"]
			search_queries = tool_call["args"].get("search_queries", [])
			query_results = {}
			for query in search_queries:
				result = tavily_tool.invoke(query)
				query_results[query] = result
			tool_messages.append(ToolMessage(
				content=json.dumps(query_results),
				tool_call_id=call_id
				))
	return tool_messages

tool_response = execute_tools(response_list)
response_list.extend(tool_response)

revise_instructions = """Revise your previsous answer using the new information, applying the rigor and evidence-based approach of Dr. David Attia.
- Incorporate the previous critique to add clinically relevant information, focusing on mechanistic understanding and individual variability.
- You MUST include numerical citations referencing peer-reviewed research, randomized controlled trials, or meta-analyses to ensure medical accuracy.
- Distinguish between correlation and causation, and acknowledge limitations in current research.
- Address potential biomarker considerations (lipid panels, inflammatory markers, an so on) when relevant.
- Add a "References" section to the bottom of your answer (which does not count towards the word limit) in the form of:
- [1] https://example.com
- [2] https://example.com
- Use the previous critique to remove specutation and ensure claims are supported by high-quality evidence. Keep response under 250 words with precision over volume.
- when discussing nutritional interventions, consider metabolic flexibility, insulin sensitivity, and individual response variability.
"""
revisor_prompt = prompt_template.partial(first_instruction=revise_instructions)

class ReviseAnswer(AnswerQuestion):
	references: List[str] = Field(description="Citations motivating your updated answer.")
revisor_chain = revisor_prompt | llm.bind_tools(tools=[ReviseAnswer])

response = revisor_chain.invoke({"messages": response_list})
response_list.append(response)

# Building the Graph
MAX_ITERATIONS = 4

def event_loop(state: List[BaseMessage]) -> str:
	count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
	num_iterations = count_tool_visits
	if num_iterations >= MAX_ITERATIONS:
		return END
	return "execute_tools"

graph=MessageGraph()

graph.add_node("respond", initial_chain)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revisor_chain)

graph.add_edge("respond", "execute_tools")
graph.add_edge("execute_tools", "revisor")

graph.add_conditional_edges("revisor", event_loop)
graph.set_entry_point("respond")

app = graph.compile()
responses = app.invoke(
	"""I'm pre-diabetic and need to lower my blood sugar, and i have heart issues.
	What breakfast food should I eat and avoid"""
	)

print("--- Initial Draft Answer ---")
initial_answer = responses[1].tool_calls[0]['args']['answer']
print(initial_answer)
print("\n")

print("--- Intermediate and Final Revised Answers ---")
answers = []

for msg in reversed(responses):
	if getattr(msg, 'tool_calls', None):
		for tool_call in msg.tool_calls:
			answer = tool_call.get('args', {}).get('answer')
			if answer:
				answers.append(answer)

for i, ans in enumerate(answers):
	label = "Final Revised Answer" if i == 0 else f"Intermediate Step {len(answers) - 1}"
	print(f"{label}:\n{ans}\n")