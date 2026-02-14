# pip install langchain==0.3.25 | tail -n 1
# pip install langchain-openai==0.3.19 | tail -n 1

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

from langchain.chat_models import init_chat_model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

@tool
def add(a: int, b: int) -> int:
	"""
	Add a and b.

	Args:
		a (int): first integer to be added
		b (int): second integer to be added

	Return:
		int: sum of a and b
	"""
	return a + b

@tool
def subtract(a: int, b: int) -> int:
	"""Subtract b from a."""
	return a - b

@tool
def multiply(a: int, b: int) -> int:
	"""Multiply a and b."""
	return a * b

tool_map = {
	"add": add,
	"subtract": subtract,
	"multiply": multiply
}

input_ = {
	"a": 1,
	"b": 2
}

tool_map["add"].invoke(input_)

tools = [add, subtract, multiply]

llm_with_tools = llm.bind_tools(tools)

query = "What is 3 + 2?"
chat_history = [HumanMessage(content=query)]

response_1 = llm_with_tools.invoke(chat_history)
chat_history.append(response_1)

tool_calls_1 = response_1.tool_calls

tool_1_name = tool_calls_1[0]["name"]
tool_1_args = tool_calls_1[0]["args"]
tool_call_1_id = tool_calls_1[0]["id"]

tool_response = tool_map[tool_1_name].invoke(tool_1_args)
tool_message = ToolMessage(content=tool_response, tool_call_id=tool_call_1_id)

chat_history.append(tool_message)

answer = llm_with_tools.invoke(chat_history)
print(answer.content)

# Building an Agent

class ToolCallingAgent:
	def __init__(self, llm):
		self.llm_with_tools = llm.bind_tools(tools)
		self.tool_map = tool_map

	def run(self, query: str) -> str:
		# Step 1: Initial user message
		chat_history = [HumanMessage(content=query)]

		# Step 2: LLM chooses tool
		response = self.llm_with_tools.invoke(chat_history)
		if not response.tool_calls:
			return response.content
		# Step 3: Handle first tool call
		tool_call = response.tool_calls[0]
		tool_name = tool_call["name"]
		tool_args = tool_call["args"]
		tool_call_id = tool_call["id"]

		# Step 4: Call tool manually
		tool_result = self.tool_map[tool_name].invoke(tool_args)

		# Step 5: Send result back to LLM
		tool_message = ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
		chat_history.extend([response, tool_message])

		final_response = self.llm_with_tools.invoke(chat_history)
		return final_response.content