from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

class Add(BaseModel):
	""" Add two numbers together"""
	a: int = Fiel(description="First number")
	b: int = Field(description="Second number")

llm = ChatOpenAI(model="gpt-4.1-nano")
initial_chain = llm.bind_tools(tools=[Add])

question = "add 1 and 10"

response = initial_chain.invoke([HumanMessage(content=question)])

def extract_and_add(response):
	tool_call = response.tool_calls[0]
	a = tool_call["args"]['a']
	b = tool_call["args"]['b']
	return a + b

result = extract_and_add(response)
print(f"LLM extracted: a={response.tool_calls[0]['args']}, b={response.tool_calls[0]['args']['b']}")
print(f"Result: {result}")

--------------------------------------
from pydantic import BaseModel
from typing import Literal

class TwoOperands(BaseModel):
	a: float
	b: float
class AddInput(TwoOperands):
	operation: Literal['add']

class SubtractInput(TwoOperands):
	operation: Literal['subtract']

class MathOutput(BaseModel):
	result: float

def add_tool(data: AddInput) -> MathOutput:
	return MathOutput(result=data.a + data.b)

incoming_json = '{"a": 7, "b": 3, "operation": "subtract"}'

def dispatch_tool(json_payload: str) -> str:
	base = SubtractInput.parse_raw(json_payload)
	if base.operation == "add":
		output = add_tool(AddInput.parse_raw(json_payload))
	elif base.operation == "subtract":
		output = subtract_tool(SubtractInput.parse_raw(json_payload))
	else:
		raise ValueError("Unsupported operation")
	return output.json

result_json = dispatch_tool(incoming_json)
print(result_json)