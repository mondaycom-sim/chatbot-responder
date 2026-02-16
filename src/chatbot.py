"""Multi-framework chatbot responder."""
from crewai import Agent, Task, Crew
from autogen import AssistantAgent
from llama_index.core import VectorStoreIndex


def create_crew():
    researcher = Agent(role="Researcher", goal="Find relevant board data")
    writer = Agent(role="Writer", goal="Generate user response")
    crew = Crew(agents=[researcher, writer], tasks=[])
    return crew


def handler(event, context):
    crew = create_crew()
    result = crew.kickoff()
    return {"statusCode": 200, "body": str(result)}
