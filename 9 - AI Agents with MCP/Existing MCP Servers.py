# !pip install fastmcp

sys.stdout.write("Hello")
sys.exit("Some message")

print("This is an error message", file=sys.stderr)

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport, StdioTransport

# STDIO Transport
stdio_transport = StdioTransport(
    command = "npx",
    args=["-y", "@upstash/context7-mcp"]
)

stdio_client = Client(stdio_transport)

async with stdio_client as client:
    tools = await client.list_tools()

print(len(tools))
print(tools[0].name)
print(tools[0].description)
print(tools[0].inputSchema)

async with stdio_client as client:
    response = await client.call_tool("resolve-library-id", {
        "libraryName": "fastmcp",
        "query": "I want to create a new MCP server using the fastmcp Python framework"
    })

print(response.content[0].text)

async with stdio_client as client:
    docs = await client.call_tool("query_docs", {
        "libraryId": "/llmstxt/gofastmcp_llms-full_txt",
        "query": "I want to fetch the code snippets and the documentation",
        "tokens": 5000
    })
    print(docs.content[0].text[:1000])

# HTTP
import requests

url = "www.https://ibm.com"
r = requests.get(r)

print(r.status_code)
print(r.request.header)
print(r.request.body)

http_transport = StreamableHttpTransport(
    url="https://mcp.context7.com/mcp"
)

http_client = Client(http_transport)

async with http_client as client:
    tools = await client.list_tools()

    response = await client.call_tool("resolve-library-id", {
        "libraryName": "fastmcp",
        "query": "I want to create a new MCP server using the fastmcp Python framework"
    })

    docs = await client.call_tool("query_docs", {
        "libraryId": "/llmstxt/gofastmcp_llms-full_txt",
        "query": "I want to fetch the code snippets and the documentation",
        "tokens": 5000
    })

for tool in tools:
    print(
        f"""{tool.name}: \n
        {tool.description} \n
        {tool.inputSchema} """)
    print(response.content[0].text[:1000])
    print(docs.content[0].text[:500])

    