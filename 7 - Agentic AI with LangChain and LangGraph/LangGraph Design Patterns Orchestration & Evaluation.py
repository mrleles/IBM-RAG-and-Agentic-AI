'''
%%capture
!pip install langchain-openai==0.3.27
!pip install langgraph==0.6.6
!pip install pygraphviz==1.14
'''

from langgraph.graph import StateGraph, END, START
from langgraph.types import Send

from typing import TypeDict, Annotated, list, Literal
from pydantic import BaseModel, Field
import operator
from pprint import pprint
from IPython.display import Image, display

from langgchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# !pip install litellm

import litellm
litellm.ssl_verify = False

# Dish schema for a single dish
class Dish(BaseModel):
    name: str = Field(
        description="Name of the dish (for example, Spaghetti Bolognese, Chicken Curry)."
    )
    ingredients: List[str] = Field(
        description="List of ingredients needed for this dish, separated by commas."
    )
    location: str = Field(
        description="The cuisine or cultural origin of the dish (for example, Italian, Indian, Mexican)."
    )

# Dishes schema for a list of Dish objects
class Dishes(BaseModel):
    sections: List[Dish] = Field(
        description="A list of grocery sections, one for each dish, with ingredients."
    )

# construct a prompt template
dish_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an assistant that generates a structured grocery list.\n\n"
        "The user wants to prepare the following meals: {meals}\n\n"
        "For each meal, return a section with:\n"
        "- the name of the dish\n"
        "- a comma-separated list of ingredients needed for that dish.\n"
        "- the cuisine or cultural origin of the food"
    )
])

planner_pipe = dish_prompt | llm.with_structured_output(Dishes)
planner_pipe.invoke({"meals": ["banana smoothie", "carrot cake"]})

class State(TypedDict):
    meals: str
    sections: List[Dish]
    completed_menu: Annotated[List[str], operator.add]
    final_meal_guide: str

dummy_state: State = {
    "meals": "Spaghetti Bolognese and Chicken Stir Fry",
    "sections": [],
    "completed_menu": [],
    "final_meal_guide": ""
}

report_sections = planner_pipe.invoke({"meals": dummy_state['meals']})

for i, section in enumerate(report_sections.sections):
    print(f"Dish {i+1}\n")
    dummy_state["sections"].append(section)
    print(f"Item Name: {section.name}")
    print(f"Location/Cuisine: {section.location}")
    print(f"Ingredients: {section.ingredients}")

def orchestrator(state: State):
    """Orchestrator that generates a structured dish list from the given meals."""
    dish_descriptions = planner_pipe.invoke({"meals": state["meals"]})
    return {"sections": dish_descriptions}

chef_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a world-class chef from {location}.\n\n"
        "please introduce yourself briefly and present a detailed walkthrough for preparing the dish: {name}.\n"
        "your response should include:\n"
        "- Start with hello, with your name and culinary background\n"
        "- A clear list of preparation steps\n"
        "- A full explanation of the cooking process\n\n"
        "Use the following ingredients: {ingredients}."
    )
])

chef_pipe = chef_prompt | llm

class WorkerState(TypedDict):
    section: Dish
    completed_menu: Annotated[list, operator.add]

def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    # Kick off section writing in parallel via Send() API
    return [Send("chef_worker", {"section": s}) for s in state["sections"]]

def chef_worker(state: WorkerState):
    """Worker node that generates the cooking instructions for one meal section."""
    meal_plan = chef_pipe.invoke({
        "name": state["section"].name,
        "location": state["section"].location,
        "ingredients": state["section"].ingredients,
    })

    return {"completed_menu": [meal_plan.content]}

dummy_dishes: List[Dish] = dummy_state["sections"]

for section in dummy_dishes:
    worker_state: WorkerState = {
        "section": section,
        "recipe": []
    }
    result = chef_worker(worker_state)
    dummy_state["completed_menu"] += result["completed_menu"]

completed_menu_sections = "\n".join(dummy_state["completed_menu"])
print(completed_menu_sections[:1000])

def synthesizer(state: State):
    """Synthesize full report from sections"""
    completed_sections = state["completed_menu"]
    completed_menu = "\n\n---\n\n".join(completed_sections)
    return {"final_meal_guide": completed_menu}

# Building the Workflow (Orchestration)

