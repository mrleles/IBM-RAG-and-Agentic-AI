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

from typing import List

@tool
def add_numbers_with_options(numbers: List[float], absolute: bool = False) -> float:
    """
    Adds a list of numbers provided as input.

    Parameters:
    - numbers (List[float]): A list of numbers to be summed.
    - absolute (bool): If True, use the absolute values of the numbers before summing.

    Returns:
    - float: The total sum of the numbers.
    """
    if absolute:
        numbers = [abs(n) for n in numbers]
    return sum(numbers)

from typing import Dict, Union

@tool
def sum_numbers_with_complex_output(inputs: str) -> Dict[str, Union[float, str]]:
    """
    Extracts and sums all integers and decimal numbers from the input string.

    Parameters:
    - inputs (str): A string that may contain numeric values.

    Returns:
    - dict: A dictionary with the key "result". If numbers are found, the value is their sum (float).
        If no numbers are found or an error occurs, the value is a corresponding message (str).

    Example Input:
    "Add 10, 20.5, and -3."

    Example Output:
    {"result": 27.5}
    """
    matches = re.findall(r'-?\d+(?:\.\d+)?', inputs)
    if not matches:
        return {"result": "No numbers found in input."}
    try:
        numbers = [float(num) for num in matches]
        total = sum(numbers)
    except Exception as e:
        return {"result": f"Error during summation: {str(e)}"}

@tool
def sum_numbers_from_text(inputs: str) -> float:
    """
    Adds a list of numbers provided in the input string.

    Args:
    text: A string containing numbers that should be extracted and summed.

    Returns:
    The sum of all numbers found in the input.
    """
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    result = sum(numbers)
    return result

from langchain.agents import initialize_agent

agent = initialize_agent([add_tool], llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True)

response = agent.run("In 2023, the US GDP was approximately $27.72 trillion, while Canada's was around $2.14 trillion and Mexico's was about $1.79 trillion what is the total.")

agent2 = initialize_agent([sum_numbers_from_text], llm, agent="structured-chat-zero-shot-react-description", verbose=True, handle_parsing_errors=True)

from langchain_openai import ChatOpenAI
llm_ai = ChatOpenAI(model="gpt-4.1-nano")

agent_3 = initialize_agent([sum_numbers_with_complex_output], llm_ai, agent="openai-functions", verbose=True, handle_parsing_errors=True)

agent_4 = initialize_agent(
    [add_numbers_with_options],
    llm,
    agent="structured-chat-zero-shot-react-description",
    verbose=True
    )

response = agent_4.invoke({
    "input": "Add -10, -20, and -30 using absolute values."
    })

agent_openai = initialize_agent(
    [add_numbers_with_options],
    llm_ai,
    agent="openai-functions",
    verbose=True
    )

# pip install langgraph==0.6.1 | tail -n 1
from langgraph.prebuilt import create_react_agent

agent_exec = create_react_agent(model=llm, tools=[sum_numbers_from_text])
msgs = agent_exec.invoke({"messages":[("human", "Add the numbers -10, -20, -30")]})

# Mathematical toolkit
@tool
def subtract_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string and performs subtraction sequentially, starting with the first number.

    This function is designed to handle input in string format, where numbers are separated
    by spaces, commas, or other delimiters. It parses the string, extracts valid numeric values,
    and calculates the result by subtracting each subsequent number from the first. inputs[0]-inputs[1]-inputs[2]

    Parameters:
    - inputs (str):
      A string containing numbers to subtract. The string may include spaces, commas, or
      other delimiters between the numbers.

    Returns:
    - dict:
      A dictionary containing the key "result" with the calculated difference as its value.
      If no valid numbers are found in the input string, the result defaults to 0.

    Example Input:
    "100, 20, 10"

    Example Output:
    {"result": 70}

    Notes:
    - Non-numeric characters in the input are ignored.
    - If the input string contains only one valid number, the result will be that number negated.
    - Handles a variety of delimiters (e.g., spaces, commas) but does not validate input formats
      beyond extracting numeric values.
    """
    # Extract numbers from the string
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]

    # If no numbers are found, return 0
    if not numbers:
        return {"result": 0}

    # Start with the first number negated
    result = numbers[0]

    # Subtract all subsequent numbers
    for num in numbers[1:]:
        result -= num

    return {"result": result}

# Multiplication Tool
@tool
def multiply_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string and calculates their product.

    Parameters:
    - inputs (str): A string containing numbers separated by spaces, commas, or other delimiters.

    Returns:
    - dict: A dictionary with the key "result" containing the product of the numbers.

    Example Input:
    "2, 3, 4"

    Example Output:
    {"result": 24}

    Notes:
    - If no numbers are found, the result defaults to 1 (neutral element for multiplication).
    """
    # Extract numbers from the string
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    print(numbers)

    # If no numbers are found, return 1
    if not numbers:
        return {"result": 1}

    # Calculate the product of the numbers
    result = 1
    for num in numbers:
        result *= num
        print(num)

    return {"result": result}

# Division Tool
@tool
def divide_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string and calculates the result of dividing the first number
    by the subsequent numbers in sequence.

    Parameters:
    - inputs (str): A string containing numbers separated by spaces, commas, or other delimiters.

    Returns:
    - dict: A dictionary with the key "result" containing the quotient.

    Example Input:
    "100, 5, 2"

    Example Output:
    {"result": 10.0}

    Notes:
    - If no numbers are found, the result defaults to 0.
    - Division by zero will raise an error.
    """
    # Extract numbers from the string
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]


    # If no numbers are found, return 0
    if not numbers:
        return {"result": 0}

    # Calculate the result of dividing the first number by subsequent numbers
    result = numbers[0]
    for num in numbers[1:]:
        result /= num

    return {"result": result}

tools = [add_numbers, subtract_numbers, multiply_numbers, divide_numbers]

from langgraph.prebuilt import create_react_agent

math_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a helpful mathematical assistant that can perform various operations. Use the tools precisely and explain your reasoning clearly.")

response = math_agent.invoke({
    "messages":[("human", "What is 25 divided by 4?")]
    })

final_answer = response["messages"][-1].content

correct_tasks = []
for index, test in enumerate(test_cases, start=1):
    query = test["query"]
    expected_result = test["expected"]["result"]

    print(f"\n--- Test Case {index}: {test['description']} ---")
    print(f"Query: {query}")

    response = math_agent_new.invoke({"messages": [("human", query)]})

    tool_message = None
    for msg in response["messages"]:
        if hasattr(msg, 'name') and msg.name in ['add_numbers', 'new_subtract_numbers', 'multiply_numbers', 'divide_numbers']:
            tool_message = msg
            break

    if tool_message:
        import json
        tool_result = json.loads(tool_message.content)["result"]
        print(f"Tool Result: {tool_result}")
        print(f"Expected Result: {expected_result}")

        if tool_result == expected_result:
            print(f"Test Passed: {test['description']}")
            correct_tasks.append(test["description"])
        else:
            print(f"Test Failed: {test['description']}")

    else:
        print("No tool was called by the agent")

print("\nCorrectly passed tests:", correct_tasks)

# LangCahin's built-in tools

from langchain_community.utilities import WikipediaAPIWrapper

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for factual information about a topic.

    Parameters:
    - query (str): The topic or question to search for on Wikipedia

    Returns:
    - str: A summary of relevant information from Wikipedia
    """
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)

search_wikipedia.invoke("What is tool calling?")

