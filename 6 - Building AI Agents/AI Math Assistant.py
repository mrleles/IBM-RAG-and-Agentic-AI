'''
%pip install langchain==0.3.23 | tail -n 1
%pip install langchain-ibm==0.3.10 | tail -n 1
%pip install langchain-community==0.3.16 | tail -n 1
%pip install wikipedia==1.4.0 | tail -n 1
%pip install openai==1.77.0 | tail -n 1
%pip install langchain-openai==0.3.16 | tail -n 1
'''

from langchain_ibm import ChatWatsonx
import re

llm = ChatWatsonx(
    model_id="ibm/granite-4-h-small",
    url="url",
    project_id="skills-network"
    )

response = llm.invoke("What is tool calling in langchain?")
print(response.content)

def add_numbers(inputs:str) -> dict:
    """
    Adds a list of numbers provided in the input dictionary or extracts numbers from a string.

    Parameters:
    - inputs (str):
    string, it should contain numbers that can be extracted and summed.

    Returns:
    - dict: A dictionary with a single key "result" containing the sum of the numbers.

    Example Input (Dictionary):
    {"numbers": [10, 20, 30]}

    Example Input (String):
    "Add the numbers 10, 20, and 30."

    Example Output:
    {"result": 60}
    """
    numbers = [int(x) for x in inputs.replace(",", "").split() if x.isdigit()]

    result = sum(numbers)
    return {"result": result}

from langchain.agents import Tool
add_tool=Tool(
    name="AddTool",
    func=add_numbers,
    description="Adds a list of numbers and returns the result.")

# Tool Function: add_tool.invoke


from langchain_core.tools import tool
import re

@tool
def add_numbers(inputs:str) -> dict:
    """
    Adds a list of numbers provided in the input string.
    Parameters:
    - inputs (str):
    string, it should contain numbers that can be extracted and summed.
    Returns:
    - dict: A dictionary with a single key "result" containing the sum of the numbers.
    Example Input:
    "Add the numbers 10, 20, and 30."
    Example Output:
    {"result": 60}
    """
    # Use regular expressions to extract all numbers from the input
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]

    result = sum(numbers)
    return {"result": result}
