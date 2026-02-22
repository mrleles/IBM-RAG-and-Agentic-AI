# pip install -q langgraph==0.2.57 langchain-ibm==0.3.10

from langgraph.graph import StateGraph

from typing import TypedDict, Optional

class AuthState(TypedDict):
    username: Optional[str]
    password: Optional[str]
    is_authenticated: Optional[bool]
    output: Optional[str]

def input_node(state):
    if state.get('username', "") == "":
        state['username'] = input("What is your username?")

    password = input("Enter your password: ")

    return {"password":password}

def validate_credentials_node(state):
    username = state.get("username", "")
    password = state.get("password", "")

    if username == "test_user" and password == "secure_password":
        is_authenticated = True
    else:
        is_authenticated = False

    return {"is_authenticated": is_authenticated}

def success_node(state):
    return {"output": "Authentication successful! Welcome."}

def failure_node(state):
    return {"output": "Not Successfull, please try again!"}

def router(state):
    if state['is_authenticated']:
        return "success_node"
    else:
        return "failure_node"
    
# Creating the Graph
from langgraph.graph import END

