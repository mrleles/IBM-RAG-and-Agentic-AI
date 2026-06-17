'''
pip install virtualenv
virtualenv .venv
source .venv/bin/activate

pip install langgraph==0.6.6
pip install langchain==0.3.27
pip install langchain-openai==0.3.32
pip install langchain-ibm==0.3.18
pip install langchain-mcp-adapters==0.1.9

# main.py
touch main.py
'''

# Standard library imports
import asyncio
# Third-party imports for MCP (Model Context Protocol) and LangGraph
from langchain_mcp_adapters.client import MultiServerMCPClient # Connects to MCP servers
from langgraph.prebuilt import create_react_agent # Creates ReAct-style agents
from langgraph.checkpoint.memory import InMemorySaver # Provides conversation memory
from langchain_openai import ChatOpenAI # OpenAI chat model integration
from langchain_ibm import ChatWatsonx # Watsonx chat model if OpenAI gets rate limited


async def main():
    """
    Main function that sets up and runs an AI agent with access to multiple MCP servers.
    The agent can access Context7 library documentation and Met Museum collections.
    """

    client = MultiServerMCPClient(
        {"context7": {
            "url": "https://mcp.context7.com/mcp",
            "transport": "streamable_http",
        },
        "met-museum": {
            "command": "npx",
            "args": ["-y", "metmuseum-mcp"],
            "transport": "stdio",
        }}
    )

    openai_model = ChatOpenAI(
        model="gpt-5-nano",
    )

    tools = await client.get_tools()

    checkpointer = InMemorySaver()

    config = {"configurable": {"thread_id": "conversation_id"}}

    agent = create_react_agent(
        model=openai_model,
        tools=tools,
        checkpointer=checkpointer
    )

# Low-level approach using LangChain MCP adapter's documentation
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

server_params = StdioServerParameters(
    command="python",
    args=["/path/to/math_server.py"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        tools = await load_mcp_tools(session)

        agent = create_react_agent("openai:gpt-4.1", tools)
        agent_response = await agent.ainvoke({"messages": "What's (3 + 5) x 12?"})

# Streamable HTTP version
 from mcp import ClientSession
 from mcp.client.streamable_http import streamablehttp_client

 from langgraph.prebuilt import create_react_agent
 from langchain_mcp_adapters.tools import load_mcp_tools

 async with streamablehttp_client("http://localhost:3000/mcp") as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()

        tools = await load_mcp_tools(session)
        agent = create_react_agent("openai:gpt-4.1", tools)
        math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})

# -------------------------------------------------------------------------------------------

    response = await agent.ainvoke(
        {"messages": [
            {"role": "system", "content": "You are a smart, useful agent with tools to access code library documentation and the Met Museum collection."},
            {"role": "user", "content": "Give a brief introduction of what you do and the tools you can access."},
        ]},
        config=config
    )
    print(response['messages'][-1].content)

    while True:
        choice = input("""
        Menu:
        1. Ask the agent a question
        2. Quit
        Enter your choice (1 or 2):
        """)

        if choice == "1":
            print("Your question")
            query = input(">")

            response = await agent.ainvoke(
                {"message": query},
                config=config
            )

            print(response['messages'][-1].content)
        else:
            print("Goodbye!")
            break

    if __name__ == "__main__":
        asyncio.run(main())