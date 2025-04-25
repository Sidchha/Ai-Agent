import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor
from tools import search_tool, wiki_tool, save_tool  # Assuming these tools are defined in tools.py
import google.generativeai as genai
from datetime import datetime

# Load environment variables
load_dotenv()

# Define the response schema
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Setup Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-1.5-pro-latest")  # Use the correct model version for Gemini

# Define the parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Build the prompt with necessary variables
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a research assistant. You have access to the following tools:
        - search_tool: search the web for information and instructions.
        - wiki_tool: retrieve articles from Wikipedia.
        - save_tool: save research output to a file.

        Use these tools when needed. Include any tools used under "tools_used" in your final output.

        Format your response like this:
        {format_instructions}
    """),
    ("human", "{query}"),
    ("placeholder", "{chat_history}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())

# Create an AgentExecutor without relying on `create_tool_calling_agent`
tools = [search_tool, wiki_tool, save_tool]

# Manually handle the agent logic
def invoke_agent(query):
    # Generate the response from the model
    formatted_prompt = prompt.format(query=query)
    response = llm.generate_content(formatted_prompt)

    # Parse the response using Pydantic schema
    try:
        structured_response = parser.parse(response.text)
        print("Structured Response:", structured_response)

        # Call tools based on the agent's logic (this can be customized as needed)
        # Example: If the agent suggests using the `search_tool`
        if "search_tool" in structured_response.tools_used:
            search_result = search_tool.func(query)
            print("Search result:", search_result)
        
        if "wiki_tool" in structured_response.tools_used:
            wiki_result = wiki_tool.func(query)
            print("Wikipedia result:", wiki_result)
        
        # Save the research output
        if "save_tool" in structured_response.tools_used:
            save_result = save_tool.func(str(structured_response), "research_output.txt")
            print(save_result)

    except Exception as e:
        print("Error parsing response:", e)
        print("Raw Response:", response)

# Get user input and invoke the agent
query = input("What can I help you research? ")
invoke_agent(query)