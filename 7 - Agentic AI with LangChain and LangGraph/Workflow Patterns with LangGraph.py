'''
%%capture
!pip install langchain-openai==0.3.27
!pip install langgraph==0.6.6
!pip install pygraphviz==1.14
'''

from langgraph.graph import StateGraph, END,START
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
import httpx

def print_workflow_info(workflow, app=None):
    """Prints comprehensive information about a LangGraph workflow."""
    print("WORKFLOW INFORMATION")
    print("====================")
    print(f"Nodes: {workflow.nodes}")
    print(f"Edges: {workflow.edges}")

    try:
        finish_points = workflow.finish_points
        print(f"Finish points: {finish_points}")
    except:
        try:
            print(f"Finish point: {workflow._finish_point}")
        except:
            print("Finish points attribut not directly accessible")

    if app:
        print("\nWorkflow Visualization:")
        from IPython.display import display
        display(app.get_graph().draw_png())

client = httpx.Client(verify=False)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    http_client=client
)

class ChainState(TypedDict):
    job_description: str
    resume_summary: str
    cover_letter: str

def generate_resume_summary(state: ChainState) -> ChainState:
    prompt = f"""
You're a resume assistant. Read the following job descrition and summarize the key qualifications and experience the ideal candidate should have, phrased as if from the perspective of a strong applicant's resume summary.

Job Description:
    {state['job_description']}
    """

    response = llm.invoke(prompt)

    return {**state, "resume_summary": response.content}

def generate_cover_letter(state: ChainState) -> ChainState:
    prompt = f"""
You're a cover letter writing assistant. Using the resume summary below, write a professional and personalized cover letter for the following job.

Resume Summary:
    {state['resume_summary']}

Job Description:
    {state['job_description']}
    """

    response = llm.invoke(prompt)

    return {**state, "cover_letter": response.content}

workflow = StateGraph(ChainState)

workflow.add_node("generate_resume_summary", generate_resume_summary)
workflow.add_node("generate_cover_letter", generate_cover_letter)

workflow.set_entry_point("generate_resume_summary")
workflow.add_edge("generate_resume_summary", "generate_cover_letter")
workflow.set_finish_point("generate_cover_letter")

app = workflow.compile()

from IPython.display import Image, display

display(Image(app.get_graph().draw_png()))

input_state = {
        "job_description": "We are looking for a data scientist with experience in machine learning, NLP, and Python. Prior work with large datasets and experience deploying models into production is required."
}

result = app.invoke(input_state)

# Workflow Pattern: Routing

class RouterState(TypedDict):
    user_input: str
    task_type: str
    output: str

class Router(BaseModel):
    role: str = Field(..., description="Decide whether the user wants to summarize a passage output 'summarize' or translate text into French output translate.")
llm_router = llm.bind_tools([Router])

def router_node(state: RouterState) -> RouterState:
    routing_prompt = f"""
    You are an AI task classfier.

    Decide whether the user wants to:
    - "summarize" a passage
    - or "translate" text into French

    Respond with just one word: 'summarize' or 'translate'.

    User Input "{state['user_input']}"
"""
    
    response = llm_router.invoke(routing_prompt)

    return {**state, "task_type": response.tool_calls[0]['args']['role']}

def router(state: RouterState) -> str:
    return state['task_type']

def summarize_node(state: RouterState) -> RouterState:
    prompt = f"Please summarize the following passage:\n\n{state['user_input']}"
    response = llm.invoke(prompt)

    return {**state, "task_type": "summarize", "output": response.content}

def translate_node(state: RouterState) -> RouterState:
    prompt = f"Translate the following text to French:\n\n{state['user_input']}"
    response = llm.invoke(prompt)

    return {**state, "task_type": "translate", "output": response.content}

workflow = StateGraph(RouterState)

workflow.add_node("router", router_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("translate", translate_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", router, {
    "summarize": "summarize",
    "translate": "translate"
})

workflow.set_finish_point("summarize")
workflow.set_finish_point("translate")

app = workflow.compile()

from IPython.display import Image, display

display(Image(app.get_graph().draw_png()))

input_text = {
    "user_input": "Can you translate this sentence: I love programming?"
}
result = app.invoke(input_text)
print(result['output'])

input_text = {
    "user_input": "Can you summarize this sentence: I love programming so much it is the best thing ever. All I want to do is programming?"
}
result = app.invoke(input_text)

# Parallelization - Multilingual Translation Assistant

class State(TypedDict):
    text: str
    french: str
    spanish: str
    japanese: str
    combined_output: str

def translate_french(state: State) -> dict:
    response = llm.invoke(f"Translate the following text to French:\n\n{state['text']}")
    return {"french": response.content.strip()}

def translate_spanish(state: State) -> dict:
    response = llm.invoke(f"Translate the following text to Spanish:\n\n{state['text']}")
    return {"spanish": response.content.strip()}

def translate_japanese(state: State) -> dict:
    response = llm.invoke(f"Translate the following text to Japanese:\n\n{state['text']}")
    return {"japanese": response.content.strip()}

def aggregator(state: State) -> dict:
    combined = f"Original Text: {state['text']}\n\n"
    combined += f"French: {state['french']}\n\n"
    combined += f"Spanish: {state['spanish']}\n\n"
    combined += f"Japanese: {state['japanese']}\n\n"
    return {"combined_output": combined}

graph = StateGraph(State)

graph.add_node("Translate_french", translate_french)
graph.add_node("Translate_spanish", translate_spanish)
graph.add_node("Translate_japanese", translate_japanese)
graph.add_node("aggregator", aggregator)

graph.add_edge(START, "translate_french")
graph.add_edge(START, "translate_spanish")
graph.add_edge(START, "translate_japanese")

graph.add_edge("Translate_french", "aggregator")
graph.add_edge("Translate_spanish", "aggregator")
graph.add_edge("Translate_japanese", "aggregator")

graph.add_edge("aggregator", END)

app = graph.compile()

input_text = {
    "text": "Good morning! I hope you have a wonderful day."
}

result = app.invoke(input_text)

# Building a Multi-Agent Routing System
class RouterState(TypedDict):
    user_input: str
    task_type: str
    output: str

class Router(BaseModel):
    role: str = Field(
        ..., 
        description="Classify the user request. Return exactly one of: 'ride_hailing_call', 'restaurant_order', 'groceries' and if you do not know output 'default_handler'"
    )

llm_router = llm.bind_tools([Router])

def router_node(state: RouterState) -> RouterState:
    response = llm_router.invoke(state['user_input'])
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]['args']['role']
        return {**state, "task_type": tool_call}
    else:
        return {**state, "task_type": "default_handler"}

def router(state: RouterState) -> str:
    return state['task_type']

def ride_hailing_node(state: RouterState) -> RouterState:
    """
    Processes ride hailing requests by extracting pickup/dropoff locations and preferences
    """
    prompt = f"""
    You are a ride hailing assistant. Based on the user's request, extract and organize the following information:
    
    - Pickup location
    - Destination/dropoff location  
    - Preferred ride type (if mentioned)
    - Any special requirements
    - Estimated timing preferences
    
    User Request: "{state['user_input']}"
    
    Provide a clear summary of the ride request with all available details.
    """
    
    response = llm.invoke(prompt)
    
    return {
        **state, 
        "task_type": "ride_hailing_call", 
        "output": response.content.strip()
    }

def restaurant_order_node(state: RouterState) -> RouterState:
    """
    Processes restaurant orders by organizing menu items, quantities, and preferences
    """
    prompt = f"""
    You are a restaurant ordering assistant. Based on the user's request, organize the following information:
    
    - Menu items requested
    - Quantities for each item
    - Special modifications or dietary restrictions
    - Delivery or pickup preference
    - Any timing requirements
    
    User Request: "{state['user_input']}"
    
    Provide a clear, organized summary of the restaurant order with all details.
    """
    
    response = llm.invoke(prompt)
    
    return {
        **state, 
        "task_type": "restaurant_order", 
        "output": response.content.strip()
    }

def groceries_node(state: RouterState) -> RouterState:
    """
    Processes grocery delivery requests with driver pickup service
    """
    prompt = f"""
    You are a grocery delivery assistant for a service where our drivers pick up groceries for customers.
    
    Based on the user's request, organize the following information:
    
    Shopping List:
    - List of grocery items needed
    - Quantities or amounts for each item
    - Brand preferences (if mentioned)
    - Any dietary restrictions or organic preferences
    
    Store Information:
    - Preferred store or location
    - Budget considerations
    - Special instructions for finding items
    
    Delivery Details:
    - Delivery address (if provided)
    - Preferred delivery time window
    - Any special delivery instructions
    - Contact information for driver coordination
    
    Driver Instructions:
    - Substitution preferences (if item unavailable)
    - How to handle out-of-stock items
    - Any items requiring special handling (fragile, cold items)
    - Payment method (if mentioned)
    
    User Request: "{state['user_input']}"
    
    Provide a comprehensive delivery order summary that our driver can use to efficiently shop and deliver groceries. 
    Include estimated pickup time and any special notes for the shopping trip.
    
    Format the response as a clear, organized delivery order that includes all necessary details for our driver service.
    """
    
    response = llm.invoke(prompt)
    
    return {
        **state, 
        "task_type": "groceries", 
        "output": response.content.strip()
    }
def default_handler_node(state: RouterState) -> RouterState:
    prompt = f"""
    I couldn't classify your request into a specific category. 
    Let me provide general assistance for: "{state['user_input']}"
    
    I can help you with:
    - Ride hailing services
    -  Restaurant orders  
    -  Grocery shopping
    
    Please rephrase your request to match one of these services, or if you need assistance with something else, I will connect you with our customer support team who can provide personalized help.
    
    Would you like me to:
    1. Help you rephrase your request for one of our services
    2. Connect you with customer support for additional assistance
    """
    response = llm.invoke(prompt)
    return {**state, "task_type": "default_handler", "output": response.content.strip()}

# TODO: Create StateGraph with RouterState
# TODO: Add all five nodes (router + 4 processing nodes)
# TODO: Set router as entry point
# TODO: Add conditional edges with all four routing options
# TODO: Set finish points for all processing nodes
# TODO: Compile the application
workflow = StateGraph(RouterState)
# Add all nodes
workflow.add_node("router", router_node)
workflow.add_node("ride_hailing_call", ride_hailing_node)
workflow.add_node("restaurant_order", restaurant_order_node)
workflow.add_node("groceries", groceries_node)
workflow.add_node("default_handler", default_handler_node)

# Set entry point
workflow.set_entry_point("router")

# Add conditional routing
workflow.add_conditional_edges("router", router, {
    "groceries": "groceries", 
    "restaurant_order": "restaurant_order",
    "ride_hailing_call": "ride_hailing_call",
    "default_handler": "default_handler"
})

# Set finish points
workflow.set_finish_point("ride_hailing_call")
workflow.set_finish_point("restaurant_order")
workflow.set_finish_point("groceries")
workflow.set_finish_point("default_handler")

# Compile the application
app = workflow.compile()

class RouterState(TypedDict):
    user_input: str
    task_type: str
    output: str


class Router(BaseModel):
    role: str = Field(
        ..., 
        description="Classify the user request. Return exactly one of: 'ride_hailing_call', 'restaurant_order',  'groceries' and if you do not know output  'default_handler'"
    )

llm_router = llm.bind_tools([Router])

def router_node(state: RouterState) -> RouterState:
    response = llm_router.invoke(state['user_input'])
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]['args']['role']
        return {**state, "task_type": tool_call}
    else:
        return {**state, "task_type": "default_handler"}


def router(state: RouterState) -> str:
    return state['task_type']
    
def ride_hailing_node(state: RouterState) -> RouterState:
    """
    Processes ride hailing requests by extracting pickup/dropoff locations and preferences
    """
    prompt = f"""
    You are a ride hailing assistant. Based on the user's request, extract and organize the following information:
    
    - Pickup location
    - Destination/dropoff location  
    - Preferred ride type (if mentioned)
    - Any special requirements
    - Estimated timing preferences
    
    User Request: "{state['user_input']}"
    
    Provide a clear summary of the ride request with all available details.
    """
    
    response = llm.invoke(prompt)
    
    return {
        **state, 
        "task_type": "ride_hailing_call", 
        "output": response.content.strip()
    }

def restaurant_order_node(state: RouterState) -> RouterState:
    """
    Processes restaurant orders by organizing menu items, quantities, and preferences
    """
    prompt = f"""
    You are a restaurant ordering assistant. Based on the user's request, organize the following information:
    
    - Menu items requested
    - Quantities for each item
    - Special modifications or dietary restrictions
    - Delivery or pickup preference
    - Any timing requirements
    
    User Request: "{state['user_input']}"
    
    Provide a clear, organized summary of the restaurant order with all details.
    """
    
    response = llm.invoke(prompt)
    
    return {
        **state, 
        "task_type": "restaurant_order", 
        "output": response.content.strip()
    }

def groceries_node(state: RouterState) -> RouterState:
    """
    Processes grocery delivery requests with driver pickup service
    """
    prompt = f"""
    You are a grocery delivery assistant for a service where our drivers pick up groceries for customers.
    
    Based on the user's request, organize the following information:
    
    Shopping List:
    - List of grocery items needed
    - Quantities or amounts for each item
    - Brand preferences (if mentioned)
    - Any dietary restrictions or organic preferences
    
    Store Information:
    - Preferred store or location
    - Budget considerations
    - Special instructions for finding items
    
    Delivery Details:
    - Delivery address (if provided)
    - Preferred delivery time window
    - Any special delivery instructions
    - Contact information for driver coordination
    
    Driver Instructions:
    - Substitution preferences (if item unavailable)
    - How to handle out-of-stock items
    - Any items requiring special handling (fragile, cold items)
    - Payment method (if mentioned)
    
    User Request: "{state['user_input']}"
    
    Provide a comprehensive delivery order summary that our driver can use to efficiently shop and deliver groceries. 
    Include estimated pickup time and any special notes for the shopping trip.
    
    Format the response as a clear, organized delivery order that includes all necessary details for our driver service.
    """
    
    response = llm.invoke(prompt)
    
    return {
        **state, 
        "task_type": "groceries", 
        "output": response.content.strip()
    }
def default_handler_node(state: RouterState) -> RouterState:
    prompt = f"""
    I couldn't classify your request into a specific category. 
    Let me provide general assistance for: "{state['user_input']}"
    
    I can help you with:
    - Ride hailing services
    -  Restaurant orders  
    -  Grocery shopping
    
    Please rephrase your request to match one of these services, or if you need assistance with something else, I will connect you with our customer support team who can provide personalized help.
    
    Would you like me to:
    1. Help you rephrase your request for one of our services
    2. Connect you with customer support for additional assistance
    """
    response = llm.invoke(prompt)
    return {**state, "task_type": "default_handler", "output": response.content.strip()}

workflow = StateGraph(RouterState)
workflow.add_node("ride_hailing_call", ride_hailing_node)
workflow.add_node("restaurant_order", restaurant_order_node)
workflow.add_node("groceries", groceries_node)
workflow.add_node("default_handler", default_handler_node)
workflow.add_node("router", router_node)
workflow.set_entry_point("router")
# Update your conditional edges to route to all three options
workflow.add_conditional_edges("router", router, {"groceries":"groceries", "restaurant_order":"restaurant_order","ride_hailing_call":"ride_hailing_call","default_handler":"default_handler"})


# Set all three as possible finish points

workflow.set_finish_point("ride_hailing_call")
workflow.set_finish_point("restaurant_order")
workflow.set_finish_point("groceries")

app = workflow.compile()

test_cases = [
    {"user_input": "I need a ride from downtown to the airport at 3pm"},
    {"user_input": "I want to order 2 large pepperoni pizzas for delivery"},
    {"user_input": "I need milk, bread, eggs, and vegetables for the week"},
    {"user_input": "What's the weather like today?"},  # Default/unclassified example
]

for i, test_input in enumerate(test_cases, 1):
    result=app.invoke(test_input)


    print(f"question {test_input["user_input"]}\n")
    print(f"task_type {result['task_type']}\n")
    print(f"output: {result['output']}\n")
    print('-----------------------------------')