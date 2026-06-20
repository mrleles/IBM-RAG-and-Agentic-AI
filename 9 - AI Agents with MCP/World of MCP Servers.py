'''
pip install fastmcp==2.12.2
pip install langchain==0.3.27
%pip install langchain_mcp_adapters==0.1.9
%pip install langgraph==0.6.7
%pip install langchain_openai==0.3.33
'''

import socket
import asyncio
from fastmcp import FastMCP, Client

import os
def make_dir():
    if os.path.exists("path"):
        print("Path directory already exists")
    else:
        print("Pathdirectory doesn't exist - creating it...")
        os.makedirs("path")
        print("Path directory created")

PORT = 8000

def test_port(port=PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return False
        except socket.error:
            return True

print(f"Port {PORT} is available: {not test_port()}")

def print_stream_info(read, write, _sid, verbose=False):
    """Print information about the read/write streams and session ID."""
    if verbose:
        print("READ (receives FROM server):")
        print(read)
        print()

        print("WRITE (sends TO server):")
        print(write)
        print()

        print("SESSION ID:")
        print(_sid())

from langchain_core.tools import tool

# Tool in Langchain
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

print(multiply.name)
print(multiply.description)
print(multiply.args)

print("What is 2 x 3?")
print("Answer: " + str(multiply.invoke({"a": 2, "b": 3})))

# Creating a Calculator MCP Server
from fastmcp import FastMCP

mcp = FastMCP(
    name="CalculatorMCPServer",
    instructions="""
    This server provides mathematical calculation tools. Call add() and subtract() to perform basic arithmetic operations
    """
)

@mcp.tool
def add(a: int, b: int) -> int:
    """
    Add two integers together.
    Args:
    a (int): The first integer.
    b (int): The second integer.
    Returns:
    int: The sum of 'a' and 'b'.
    Example:
    >>> add(3,5)
    8
    """
    return a +b

@mcp.tool
def subtract(a: int, b: int) -> int:
    """
    Subtract one integer from another.
    Args:
    a (int): The number to subtract from.
    b (int): The number to subtract.
    Returns:
    int: The result of 'a - b'.
    Example:
    >>> subtract(10, 4)
    6
    """
    return a - b

@mcp.resource("file:///endpoint/{name}")
def return_template_document(name: str) -> str:
    """Read a document by name"""
    return f"Document contents of {name}"

make_dir()

%%capture
!wget -P path/ https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/aNE__JjH4DLNEibuNpfDlg/examples.txt
!wget -P path/ https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/tfoeGPInNoajVS0DSohdVg/README.txt

@mcp.resource("file:///endpoint2/{name}")
def read_document(name: str) -> str:
    """Read a document by name from the path directory"""
    try:
        with open(f"path/{name}", "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Document '{name}' not found in path directory"
    except Exception as e:
        return f"Error reading document: {str(e)}"

# Prompts
@mcp.prompt(title="Code Review")
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"

# Creating a Client - In-Memory Transport
from fastmcp import FastMCP

client = Client(mcp)

async def call_add_tool(a: int, b: int):
    async with client:
        result = await client.call_tool("add", {"a": a, "b": b})
        return result

response = await call_add_tool(4, 5)

print("\nResult Data .data :")
print(response.data)

print("\nContent (as text):")
print(response.content[0].text)

print("\nStructured Content:")
print(response.structured_content)

async with client:
    tools = await client.list_tools()
    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

input_schema = tool.inputSchema
output_schema = tool.outputSchema

# Resources
async def call_resource(name):
    async with client:
        result = await client.read_resource(f"file:///endpoint/{name}")
        return result

response = await call_resource("README.txt")
resource = response[0]

print(f"uri: {resource.uri}")
print(f"mimeType: {resource.mimeType}")
print(f"meta: {resource.meta}")
print(f"text: {resource.text}")

# Prompts
async def call_prompt(code):
    async with client:
        result = await client.get_prompt("review_code", {"code": code})
        return result

response = await call_prompt("code to be reviewed")
message = response.messages[0]
print(f"Prompt Role: {message.role}")
print(f"Prompt Content: {message.content.text}")

# HTTP Transport MCP Servers