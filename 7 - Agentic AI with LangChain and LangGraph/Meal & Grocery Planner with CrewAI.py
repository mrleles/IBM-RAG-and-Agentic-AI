'''
%pip install langchain==0.3.20 | tail -n 1
%pip install crewai==0.141.0 | tail -n 1
%pip install langchain-community==0.3.19 | tail -n 1
%pip install langchain-openai==0.3.25 | tail -n 1
%pip install duckduckgo-search==7.5.2 | tail -n 1
%pip install crewai-tools==0.51.1 | tail -n 1
%pip install databricks-sdk==0.46.0 | tail -n 1
'''

# !pip install litellm
import litellm
litellm.ssl_verify = False

# !wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/3xGOgzMOv5jhRsA3A8N9fQ/leftover.py"

import sys
sys.path.append(".")

from leftover import LeftoversCrew

import os
files = os.listdir('.')

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from IPython.display  import display, JSON, Markdown
from datetime import datetime

class GroceryItem(BaseModel):
    """Individual grocery item"""
    name: str = Field(description="Name of the grocery item")
    quantity: str = Field(description="Quantity needed (for example, '2 lbs', '1 gallon')")
    estimated_price: str = Field(description="Estimated price (for example, '$3-5')")
    category: str = Field(description="Store section (for example, 'Produce', 'Dairy')")

sample_item = GroceryItem(
    name="Chicken Breast",
    quantity="2 lbs",
    estimated_price="$8-12",
    category="Meat"
)

print("Sample Grocery Item Structure:")
display(JSON(sample_item.model_dump()))

class MealPlan(BaseModel):
    """Simple meal plan"""
    meal_name: str Field(description="Name of the meal")
    difficulty_level: str = Field(description="'Easy', 'Medium', 'Hard'")
    servings: int = Field(description="Number of people it serves")
    researched_ingredients: List[str] = Field(description="Ingredients found through research")

class ShoppingCategory(BaseModel):
    """Store section with items"""
    section_name: str = Field(description="Store section (for example, 'Produce', 'Dairy')")
    items: List[GroceryItem] = Field(description="Items in this section")
    estimated_total: str = Field(description="Estimated cost for this section")

class GroceryShoppingPlan(BaseModel):
    """Complete simplified shopping plan"""
    total_budget: str = Field(description="Total planned budget")
    meal_plans: List[MealPlan] = Field(description="Total planned budget")
    shopping_sections: List[ShoppingCategory] = Field(description="Organized by store sections")
    shopping_tips: List[str] = Field(description="Money-saving and efficiency tips")

from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process
from crewai import LLM

# Set Watsonx environment variables
os.environ["WATSONX_API_BASE"] = "https://us-south.ml.cloud.ibm.com"
os.environ["WX_PROJECT_ID"] = "skills-network"

llm = LLM(model="watsonx/ibm/granite-3-3-8b-instruct")

# Set up search tool (you'll need to add your API key)
os.environ['SERPER_API_KEY'] = 'API_KEY'  # Replace with actual key

meal_planner = Agent(
    role="Meal Planner & Recipe Researcher",
    goal="Search for optimal recipes and create detailed meal plans",
    backstory="A skilled meal planner who researches the best recipes online, considering dietary needs, cooking skill lovels, and budget constraints."
    tools=[SerperDevTool()],
    llm=llm,
    verbose=False
)

meal_planning_task = Task(
    description=(
        "Search for the best '{meal_name}' recipe for {servings} people within a {budget} budget. "
        "Consider dietary restrictions: {dietary_restrictions} and cooking skill level: {cooking_skill}. "
        "Find recipes that match the skill level and provide complete ingredient lists with quantities."
    ),
    expected_output="A detailed meal plan with researched ingredients, quantities, and cooking instructions appropriate for the skill level.",
    agent=meal_planner,
    output_pydantic=MealPlan,
    output_file="meals.json"
)

meal_planner_crew = Crew(
    agents=[meal_planner],
    tasks=[meal_planning_task],
    process=Process.sequential,
    verbose=True
)

meal_planner_result = meal_planner_crew.kickoff(
    inputs={
        "meal_name": "Chicken Stir Fry",
        "servings": 4,
        "budget": "$25",
        "dietary_restrictions": ["no nuts"],
        "cooking_skill": "beginner"
    }
)

print("✅ Single meal planning completed!")
print("📋 Single Meal Results:")
print(meal_planner_result)

shopping_organizer = Agent(
    role="Shopping Organizer",
    goal="Organize grocery lists by store sections efficiently",
    backstory="An experience shopper who knows how to organize list for quick store trips and considers dietary restrictions.",
    tools=[],
    llm=llm,
    verbose=False
)

shopping_task = Task(
    description=(
        "organize the ingredients from the '{meal_name}' meal plan into a grocery shopping list. "
        "Group items by store sections and estimate quantities for {servings} people. "
        "Consider deitary restrictions: {dietary_restrictions} and cooking skill: {cooking_skill}. "
        "Stay within budget: {bugdet}."
    ),
    expected_output="An organized shopping list grouped by store sections with quantities and prices.",
    agent=shopping_organizer,
    context=[meal_planning_task],
    output_pydantic=GroceryShoppingPlan,
    output_file="shopping_list.json"
)

