# t2

import asyncio
import logging
from beeai_framework.backend import ChatModel, ChatModelParameters, UserMessage, SystemMessage
# Initialize the chat model
async def basic_chat_example():
    # Create a chat model instance (works with OpenAI, WatsonX, etc.)
    llm = ChatModel.from_name("watsonx:ibm/granite-3-3-8b-instruct", ChatModelParameters(temperature=0))
    
    # Create a conversation about something everyone finds interesting
    messages = [
        SystemMessage(content="You are a helpful AI assistant and creative writing expert."),
        UserMessage(content="Help me brainstorm a unique business idea for a food delivery service that doesn't exist yet.")
    ]
    # Generate response using create() method
    response = await llm.create(messages=messages)
    print("User: Help me brainstorm a unique business idea for a food delivery service that doesn't exist yet.")
    print(f"Assistant: {response.get_text_content()}")
    return response


async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL) # Suppress unwanted warnings
    response = await basic_chat_example()
if __name__ == "__main__":
    asyncio.run(main())

# t3

import asyncio
import logging
import string
from beeai_framework.backend import ChatModel, ChatModelParameters, UserMessage

class SimplePromptTemplate:
    """Simple prompt template using Python string formatting."""
    
    def __init__(self, template: str):
        self.template = template
    
    def render(self, variables: dict) -> str:
        """Render the template with provided variables."""
        # Replace mustache-style {{variable}} with Python format {variable}
        formatted_template = self.template
        for key, value in variables.items():
            formatted_template = formatted_template.replace(f"{{{{{key}}}}}", f"{{{key}}}")
        
        # Format the template with the variables
        return formatted_template.format(**variables)

async def prompt_template_example():
    llm = ChatModel.from_name("watsonx:ibm/granite-4-h-small", ChatModelParameters(temperature=0))
    
    # Create a prompt template for data science project evaluation
    template_content = """
    You are a senior data scientist evaluating a machine learning project proposal.
    
    Project Details:
    - Project Name: {{project_name}}
    - Business Problem: {{business_problem}}
    - Available Data: {{data_description}}
    - Timeline: {{timeline}}
    - Success Metrics: {{success_metrics}}
    
    Please provide:
    1. Feasibility assessment (1-10 scale)
    2. Key technical challenges
    3. Recommended approach
    4. Risk mitigation strategies
    5. Expected outcomes
    
    Be specific and actionable in your recommendations.
    """
    
    # Create the prompt template
    prompt_template = SimplePromptTemplate(template_content)
    
    # Test with different project scenarios
    project_scenarios = [
        {
            "project_name": "Smart Inventory Optimization",
            "business_problem": "Reduce inventory costs while maintaining 95% product availability",
            "data_description": "2 years of sales data, supplier lead times, seasonal patterns, 500K records",
            "timeline": "3 months development, 1 month testing",
            "success_metrics": "15% cost reduction, maintain 95% availability, <2% forecast error"
        },
        {
            "project_name": "Fraud Detection System",
            "business_problem": "Detect fraudulent transactions in real-time with minimal false positives",
            "data_description": "1M transaction records, user behavior data, device fingerprints",
            "timeline": "6 months development, 2 months validation",
            "success_metrics": "95% fraud detection rate, <1% false positive rate, <100ms response time"
        }
    ]
    
    for i, scenario in enumerate(project_scenarios, 1):
        print(f"\n=== Project Evaluation {i}: {scenario['project_name']} ===")
        
        # Render the template with scenario data
        rendered_prompt = prompt_template.render(scenario)
        print("\n  Rendered prompt:")
        print(rendered_prompt)
        
        # Generate evaluation using create() method
        messages = [UserMessage(content=rendered_prompt)]
        response = await llm.create(messages=messages)
        
        print("### LLM response: ###\n")
        print(response.get_text_content())
        
async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL) # Suppress unwanted warnings
    await prompt_template_example()

if __name__ == "__main__":
    asyncio.run(main())

# t4

import asyncio
import logging
from pydantic import BaseModel, Field
from typing import List
from beeai_framework.backend import ChatModel, ChatModelParameters, UserMessage, SystemMessage

class BusinessPlan(BaseModel):
    business_name: str = Field(description="Catchy name for the business")
    elevator_pitch: str = Field(description="30-second description of the business")
    target_market: str = Field(description="Primary target audience")
    unique_value_proposition: str = Field(description="What makes this business special")
    revenue_streams: List[str] = Field(description="Ways the business will make money")
    startup_costs: str = Field(description="Estimated initial investment needed")
    key_success_factors: List[str] = Field(description="Critical elements for success")

async def structured_output_example():
    llm = ChatModel.from_name("openai:gpt-5-nano", ChatModelParameters(temperature=0))

    messages = [
        SystemMessage(content="You are an expert business consultant and entrepreneur."),
        UserMessage(content="Create a business plan for a mobile app that helps people find and book unique local experiences in their city.")
    ]

    response = await llm.create_structure(
        schema=BusinessPlan,
        messages=messages
    )

    print("User: Create a business plan for a mobile app that helps people find and book unique local experiences in their city.")
    print("\n AI-Generated Business Plan:")
    print(f" Business Name: {response.object['business_name']}")
    print(f" Elevator Pitch: {response.object['elevator_pitch']}")
    print(f"Target Market: {response.object['target_market']}")
    print(f"⭐ Unique Value Proposition: {response.object['unique_value_proposition']}")
    print(f"💰 Revenue Streams: {', '.join(response.object['revenue_streams'])}")
    print(f"💵 Startup Costs: {response.object['startup_costs']}")
    print(f" Key Success Factors:")
    for factor in response.ovject['key_success_factors']:
        print(f" - {factor}")

async def main() -> None:
    logging.getLogger('asyncion').setLevel(logging.CRITICAL)
    await structured_output_example()

if __name__ == "__main__":
    asyncio.run(main())

# t5

import asyncio
import logging
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.backend import ChatModel, ChatModelParameters

async def minimal_tracked_agent_example():
    llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8", ChatModelParameters(temperature=0))

    SYSTEM_INSTRUCTIONS = """You are an expert cybersecurity analyst specializing in threat assessment and risk analysis.

    Your methodology:
    1. Analyze the threat landscape systematically
    2. Research authoritative sources when available
    3. Provide comprehensive risk assessment with actionable recommendations
    4. Focus on practical, emplementable security measures"""

    minimal_agent = RequirementAgent(
        llm=llm,
        tools=[],
        memory=UnconstrainedMemory(),
        instructions=SYSTEM_INSTRUCTIONS
        )

    ANALYSIS_QUERY = """Analyze the cybersecutiry risks of quantum computing for financial institutions.
    What are the main threats, timeline for concern, and recommended preparation strategies?"""

    result = await minimal_agent.run(ANALYSIS_QUERY)
    print(f"\n Pure LLM Analysis:\n{result.answer.text}")

async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    await minimal_tracked_agent_example()

if __name__ == "__main__":
    asyncio.run(main())

# t6

import asyncio
import logging
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework. tools import Tool

async def wikipedia_enhanced_agent_example():
    """
    RequirementAgent with Wikipedia - Research Enhancement and tracking

    Adding WikipediaTool provides access to Wikipedia summaries for contextual research.
    Same query - but now with research capability.
    Moreover, middleware is used to track all tool usage.
    """
    llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8", ChatModelParameters(temperature=0))

    SYSTEM_INSTRUCTIONS = """You are an expert cybersecutiry analyst specializing in threat assessment and risk analysis.

    Your methodology:
    1. Analyze the threat landscape systematically
    2. Research authoritative sources when available
    3. Provide comprehensive risk assessment with actionable recommendations
    4. Focus on practical, implementable security measures"""

    wikipedia_agent = RequirementAgent(
        llm=llm,
        tools=[WikipediaTool()],
        memory=UnconstrainedMemory(),
        instructions=SYSTEM_INSTRUCTIONS,
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        requirements=[ConditionalRequirement(WikipediaTool, max_invocations=2)]
    )

    ANALYSIS_QUERY = """Analyze the cybersecurity risks of quantum computing for financial institutions.
    What are the main threats, timeline for concern, and recommended preparation strategies?"""

    result = await wikipedia_agent.run(ANALYSIS_QUERY)
    print(f"\n Research-Enhanced Analysis:\n{result.answer.text}")

async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    await wikipedia_enhanced_agent_example()

if __name__ == "__main__":
    asyncio.run(main())

# t7

import asyncio
import logging
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool

async def reasoning_enhanced_agent_example():
    """
    RequirementAgent with Systematic Reasoning - ThinkTool + WikipediaTool

    Adding ThinkTool enables structured reasoning alongside research.
    Same query, same tracking - now with visible thinking process.
    """
    llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8", ChatModelParameters(temperature=0))

    SYSTEM_INSTRUCTIONS = """You are an expert cybersecurity analyst specializing in threat assessment and risk analysis.

    Your methodology:
    1. Analyze the threat landscape systematically
    2. Research authoritative sources when available
    3. Provide comprehensive risk assessment with actionable recommendations
    4. Focus on practical, implementable security measures"""

    reasoning_agent = RequirementAgent(
        llm=llm,
        tools=[ThinkTool(), WikipediaTool()],
        memory=UnconstrainedMemory(),
        instructions=SYSTEM_INSTRUCTIONS,
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        requirements=[
            ConditionalRequirement(ThinkTool, max_invocations=2),
            ConditionalRequirement(WikipediaTool, max_invocations=2)
            ]
        )

    ANALYSIS_QUERY = """Analyze the cybersecurity risks of quantum computing for financial institutions.
    What are the main threats, timeline for concern, and recommended preparation strategies?"""

    result = await reasoning_agent.run(ANALYSIS_QUERY)
    print(f"\n Reasoning + Research Analysis:\n{result.answer.text}")

async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    await reasoning_enhanced_agent_example()

if __name__ == "__main__":
    asyncio.run(main())

# t8

import asyncio
import logging
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool

async def controlled_execution_example():
    """
    RequirementAgent with Controlled Execution - Requirements System

    Requirements provide precise controle over tool execution order and behavior.
    Same query, same tracking - but now with strict execution rules.
    """
    llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8", ChatModelParameters(temperature=0))

    SYSTEM_INSTRUCTIONS = """You are an expert cybersecurity analyst specializing in threat assessment and risk analysis.

    Your methodology:
    1. Analyze the threat landscape systematically
    2. Research authoritative sources when available
    3. Provide comprehensive risk assessment with actionable recommendations
     Focus on practical, implementable security measures"""

    controlled_agent = RequirementAgent(
        llm=llm,
        tools=[ThinkTool(), WikipediaTool()],
        memory=UnconstrainedMemory(),
        instructions=SYSTEM_INSTRUCTIONS,
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],

        requirements=[
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,
                min_invocations=1,
                max_invocations=3,
                consecutive_allowed=False
            ),
            ConditionalRequirement(
                WikipediaTool,
                only_after=[ThinkTool],
                min_invocations=1,
                max_invocations=2
            )
        ]
    )


    ANALYSIS_QUERY = """Analyze the cybersecurity risks of quantum computing for financial institutions.
    What are the main threats, timeline for concern, and recommended preparation strategies?"""

    result = await controlled_agent.run(ANALYSIS_QUERY)
    print(f"\n Controlled Execution Analysis:\n{result.answer.text}")

async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    await controlled_execution_example()

if __name__ == "__main__":
    asyncio.run(main())

# t9

import asyncio
import logging
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool

async def reasoning_enhanced_agent_example():
    llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8", ChatModelParameters(temperature=0))

    SYSTEM_INSTRUCTION = """You are an expert cybersecurity analyst specializing in threat assessment and risk analy.

    Your methodology:
    1. Analyze the threat landscape systematically
    2. Research authoritative sources when available
    3. Provide comprehensive risk assessment with actionable recommendations
    4. Focus on practical, implementable security measures"""

    reasoning_agent = RequirementAgent(
        llm=llm,
        tools=[ThinkTool(), WikipediaTool()],
        memory=UnconstrainedMemory(),
        instructions=SYSTEM_INSTRUCTION,
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        requirements=[
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,
                force_after=Tool,
                min_invocations=5,
                consecutive_allowed=False
            )
        ]
    )

    ANALYSIS_QUERY = """Analyze the cybersecurity risks of quantum computing for financial institutions.
    What are the main threats, timeline for concern, and recommended preparation strategies?"""

    result = await reasoning_agent.run(ANALYSIS_QUERY)
    print(f"\n Reasoning + Research Analysis:\n{result.answer.text}")

async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    await reasoning_enhanced_agent_example()

if __name__ == "__main__":
    asyncio.run(main())

# t10

import asyncio
import logging
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.experimental.requirements.ask_permission import AskPermissionRequirement
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool

async def production_security_example():
    """
    Production-Ready RequirementAgent with Security Approval

    AskPermissionRequirement adds human-in-the-loop security controls.
    Same query, same tracking - but now with approval workflow.
    """
    llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-7b-128e-instruct-fp8", ChatModelParameters(temperature=0))

    SYSTEM_INSTRUCTIONS = """You are an expert cybersecurity analyst specializing in threat assessment and risk analysis.

    Your methodology:
    1. Analyze the threat landscape systematically
    2. Research authoritative sources when available
    3. Provide comprehensive risk assessment with actionable recommendations
    4. Focus on practical, implementable security measures"""

    secure_agent = RequirementAgent(
        llm=llm,
        tools=[ThinkTool(), WikipediaTool()],
        memory=UnconstrainedMemory(),
        instructions=SYSTEM_INSTRUCTIONS,
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],

        requirements=[
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,
                min_invocations=1,
                max_invocations=2,
                consecutive_allowed=False
            ),
            AskPermissionRequirement(
                WikipediaTool,
            ),
            ConditionalRequirement(
                WikipediaTool,
                only_after=[ThinkTool],
                min_invocations=0,
                max_invocations=1
            )
        ]
    )

    ANALYSIS_QUERY = """Analyze the cybersecurity risks of quantum computing for financial institutions.
    What are the main threats, timeline for concern, and recommended preparation strategies?"""

    result = await secure_agent.run(ANALYSIS_QUERY)
    print(f"\n Security-Approved Analysis:\n{result.answer.text}")

async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    await production_security_example()

if __main__ == "__main__":
    asyncio.run(main())

# t11

import asyncio
import logging
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from pydantic import BaseModel, Field
from typing import Any

class CalculatorInput(BaseModel):
    """Input model for basic mathematical calculations."""
    expression: str = Field(description="Mathematical expression using +, -, *, / (e.g., '10 + 5', '20-8', '4*6', '15/3')")

class SimpleCalculatorTool(Tool[CalculatorInput, ToolRunOptions, StringToolOutput]):
    """A simple calculator tool for basic arithmetic operations: add, subtract, multiply, divide."""
    name = "SimpleCalculator"
    description = "Performs basirc arithmetic calculations: addition (+), subtraction (-), multiplication (*), and division (/)."
    input_schema = CalculatorInput

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "calculator", "basic"],
            creator=self,
        )

    def _safe_calculate(self, expression: str) -> float:
        """Safely evaluate basic arithmetic expressions."""
        # Remove spaces for processing
        expr = expression.replace(' ', '')

        allowed_chars = set('0123456789+-*/().')
        if not all(c in allowed_chars for c in expr):
            raise ValueError("Only numbers and basic operators (+, -, *, /, parentheses) are allowed")

        try:
            result = eval(expr, {"__builtins__": {}}, {} )
            return = float(result)
        except ZeroDivisionError:
            raise ValueError("Division by zero is not allowed")
        except Exception as e:
            raise ValueError(f"Invalid arithmetic expression: {str(e)}")

    async def _run(
        self, input: CalculatorInput, options: ToolRunOptions | None, context: RunContext
    ) -> StringToolOutput:
        """Perform basic arithmetic calculations."""
        try:
            expression = input.expression.strip()

            # Perform calculation
            result = self._safe_calculate(expression)

            # Format result
            output = f"Simple Calculator\n"
            output += f"Expression: {expression}\n"
            output += f"Result: {result}\n"

            # Add operation type hint
            if '+' in expression:
                output += "Operation: Addition"
            elif '-' in expression:
                output += "Operation: Subtraction"
            elif '*' in expression:
                output += "Operation: Multiplication"
            elif '/' in expression:
                output += "Operation: Division"
            else:
                output += "Operation: Basic Arithmetic"

            return StringToolOutput(output)

        except ValueError as e:
            return StringToolOutput(f"Calculation Error: {str(e)}")
        except Exception as e:
            return StringToolOutput(f"Unexpected Error: {str(e)}")

    async def calculator_agent_example():
        """RequirementAgent with SimpleCalculatorTool - Interactive Math Assistant"""

        llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8" ChatModelParameters(temperature=0))

        calculator_agent = RequirementAgent(
            llm=llm,
            tools=[SimpleCalculatorTool()],
            memory=UnconstrainedMemory(),
            instruction="""You are a helpful math assistant. When  users ask for calculations,
            use the SimpleCalculator tool to provide accurate results.
            Always show both the expression and the calculated result.""",
            middleware=[GlobalTrajectoryMiddleware(included=[Tool])],
        )

        math_queries = [
            "What is 15 + 27?",
            "Calculate 144 divided by 12",
            "I need to know what 8 time 9 equals",
            "What's (10 + 5) * 3 - 7?"
        ]

        for query in math_queries:
            print(f"\nHuman: {query}")
            result = await calculator_agent.run(query)
            print(f"Agent: {result.answer.text}")

async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    await calculator_agent_example()

if __name__ == "__main__":
    asyncio.run(main())

# t12

import asyncio
import logging
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.experimental.ask_permission import AskPermissionRequirement
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool

async def multi_agent_travel_planner_with_language():
    """
    Advanced Multi-Agent Travel Planning System with Language Expert

    This system demonstrates:
    1. Specialized agent roles and coordination
    2. Tool-based inter-agent communication
    3. Requirements-based execution control
    4. Language and cultural expertise integration
    5. Comprehensive travel planning workflow
    """
    llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8", ChatModelParameters(temperature=0))

    destination_expert = RequirementAgent(
        llm=llm,
        tools=[WikipediaTool(), ThinkTool()],
        memory=UnconstrainedMemory(),
        instruction="""You are a Destination Research Expert specializing in travel destination analysis.

        Your expertise:
        - Landmarks and tourist activities
        - Best times to visit and seasonal considerations
        - Transportation options and accessibility
        - Safety considerations and travel advisories

        Always provide detailed, factual information with clear source attribution.""",
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        requirements=[
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,
                min_invocations=1,
                max_invocations=5,
                consecutive_allowed=False
            ),
            ConditionalRequirement(
                WikipediaTool,
                only_after=[ThinkTool],
                min_invocations=1,
                max_invocations=4,
                consecutive_allowed=False
            ),
        ]
    )

    travel_meteorologist = RequirementAgent(
        llm=llm,
        tools=[OpenMeteoTool(), ThinkTool()],
        memory=UnconstrainedMemory(),
        instructions="""You are a travel Meteorologist specializing in weather analysis for travel planning.

        Your expertise:
        - Climate patterns and seasonal weather analysis
        - Travel-specific weather recommendations
        - Packing suggestions based on weather forecasts
        - Activity planning based on weather conditions
        - Regional climate variations and microclimates
        - Weather-related travel risks and precautions

        Focus on actionable weather guidance for travelers."""
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        requirements=[
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,
                min_invocations=1,
                max_invocations=2
            ),
            ConditionalRequirement(
                OpenMeteoTool,
                only_after=[ThinkTool],
                min_invocations=1,
                max_invocations=1
            )
        ]
    )

    language_and_culture_expert = RequirementAgent(
        llm=llm,
        tools=[WikipediaTool(), ThinkTool()],
        memory=UnconstrainedMemory(),
        instructions="""You are a Langage & Cultural Expert specializing in linguistic and cultural guidance for travelers.

        Your expertise:
        - Local languages and dialects spoken in destinations
        - Essential phrases and communication tips for travelers
        - Cultural etiquette, customs, and social norms
        - Religious and cultural sensitivities to be aware of
        - Local communication styles and business etiquette
        - Cultural festivals, events, and local celebrations
        - Dining customs, tipping practices, and social interactions

        Always emphasize cultural sensitivity and respectful travel practices.""",
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        requirements=[
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,
                min_invocations=1,
                max_invocations=3,
                consecutive_allowed=False
            )
        ]
    )

    # create handoff tools for coordination with unique names
    handoff_to_destination = HandoffTool(
        destination_expert,
        name="DestinationResearch",
        description="Consult our Destination Research Expert for comprehensive information about travel destinations, attractions, and practical travel guidance."
    )
    handoff_to_weather = HandoffTool(
        travel_meteorologist,
        name="WeatherPlanning",
        description="Consult our Travel Meteorologist for weather forecasts, climate analysis, and weather-appropriate travel recommendations."
    )
    handoff_to_language = HandoffTool(
        language_and_culture_expert,
        name="LanguageCulturalGuidance",
        description="Consult our Language & Cultural Expert for essential phrases, cultural etiquette, and communication guidance for respectful travel."
    )

    travel_coordinator = RequirementAgent(
        llm=llm,
        tools=[handoff_to_destination, handoff_to_weather, handoff_to_language, ThinkTool()],
        memory=UnconstrainedMemory(),
        instructions="""You are the Travel Coordinator, the main interface for comprehensive travel planning.

        Your role:
        - Understand traveler requirements and preferences
        - Coordinate with specialized expert agents as needed
        - Synthesize information from multiple sources
        - Create comprehensive, actionable travel recommendations
        - Ensure all aspects of travel planning are covered

        Available Expert Agents:
        - Destination Expert: Practical destination information
        - Travel Meteorologist: Weather analysis and climate recommendations  
        - Language Expert: Language tips, cultural etiquette, and communication guidance

        Coordination Process:
        1. Think about what information is needed for comprehensive travel planning
        2. Delegate specific queries to appropriate expert agents using handoff tools
        3. Gather insights from multiple specialists
        4. Synthesize information into cohesive travel recommendations
        5. Provide a complete travel planning summary

        Always ensure travelers receive well-rounded guidance covering destinations and landmarks, weather, and cultural considerations.""",
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        requirements=[
            ConditionalRequirement(ThinkTool, consecutive_allowed=False),
            AskPermissionRequirement(["DestinationResearch", "WeatherPlanning", "LanguageCulturalGuidance"])
        ]
    )

    query = """I'm planning a 2-week cultural immersion trip to Japan (Tokyo and Osaka) as a first-time visitor. 
    I want to experience traditional culture, visit historical sites, and interact with locals. 
    I speak only English and want to be respectful of Japanese customs. 
    What should I know about the destination, weather expectations, and language/cultural tips?"""

    result = await travel_coordinator.run(query)
    print(f"\n Comprehensive Travel Plan:\n{result.answer.text}")

async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    await multi_agent_travel_planner_with_language()

if __name__ == "__main__":
    asyncio.run(main())