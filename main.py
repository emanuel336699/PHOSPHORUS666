import os
import logging
from crewai import Crew, Task
from Agents import WorkflowAgents
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq

os.environ["OPENAI_API_BASE"] = 'http://localhost:11434/v1'
os.environ['OPENAI_MODEL_NAME'] = 'llama3'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PhosphorusAI')

def main():
    logger.info("Starting the Phosphorus AI System")
    print(f"GROQ API Key: {os.environ.get('GROQ_API_KEY')}")

    agents_instance = WorkflowAgents()

    AI_Agent_Creator_and_Executor_agent = agents_instance.AI_Agent_Creator_and_Executor_agent()

    workflow_goal = "Create the interface for the android app called Phosphorus."
    task_description = "Finding ways to implement the AI system into the app, running the code."

    create_workflow_task = Task(
        description=task_description,
        agent=AI_Agent_Creator_and_Executor_agent,
        expected_output="A detailed plan for implementing the AI system into the Phosphorus Android app"
    )

    agents_list = [
        AI_Agent_Creator_and_Executor_agent,
        agents_instance.Workflow_Orchestrator_agent(),
        agents_instance.Visualization_Agent(),
        agents_instance.Communication_Facilitator_Agent(),
        agents_instance.Data_Collector_Agent(),
        agents_instance.Learning_Agent(),
    ]
    tasks_list = [create_workflow_task]

    crew = Crew(
        agents=agents_list,
        tasks=tasks_list,
        verbose=True
    )

    result = crew.kickoff()
    logger.info(result)

if __name__ == "__main__":
    main()
