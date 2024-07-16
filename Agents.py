import os
from textwrap import dedent
from crewai import Agent
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
import requests
import time

# Enhanced DuckDuckGoTool with retry logic and better error handling
class DuckDuckGoTool:
    _session = requests.Session()
    _last_request_time = 0
    _min_request_interval = 2

    def _run(self, query: str) -> str:
        current_time = time.time()
        if current_time - self._last_request_time < self._min_request_interval:
            time.sleep(self._min_request_interval - (current_time - self._last_request_time))

        retries = 3
        for attempt in range(retries):
            try:
                response = self._session.get(
                    "https://api.duckduckgo.com",
                    params={"q": query, "format": "json", "no_html": 1, "no_redirect": 1},
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                self._last_request_time = time.time()

                result = data.get("Abstract", "")
                if not result:
                    related_topics = data.get("RelatedTopics", [])
                    if related_topics and isinstance(related_topics, list):
                        result = related_topics[0].get("Text", "No results found.")
                    else:
                        result = "No results found."

                return result

            except requests.Timeout:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return "Error: Connection to DuckDuckGo API timed out after multiple attempts."
            except requests.RequestException as e:
                return f"Error occurred during search: {str(e)}"

    def run(self, query: str) -> str:
        return self._run(query)

    @property
    def name(self) -> str:
        return "duckduckgo_search"

    @property
    def description(self) -> str:
        return "A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query."

    def __call__(self, *args, **kwargs) -> str:
        if len(args) == 1 and isinstance(args[0], str):
            query = args[0]
        elif 'query' in kwargs and isinstance(kwargs['query'], str):
            query = kwargs['query']
        elif 'q' in kwargs and isinstance(kwargs['q'], str):
            query = kwargs['q']
        else:
            return "Invalid input. Please provide a search query as a string."

        return self.run(query)

# Enhanced CrewAgentExecutor with improved logic for information gathering
class CrewAgentExecutor:
    def __init__(self, tools, model_name="text-davinci-003"):
        self.tools = tools
        self.llm = Ollama(model=model_name)

    def execute(self, task_description: str):
        return self.agent.run(task_description)

    def gather_information(self, queries: list) -> str:
        for query in queries:
            try:
                result = self.tools['duckduckgo_search'](query=query)
                if "Error" not in result:
                    return result
            except Exception as e:
                print(f"Failed to execute query '{query}': {str(e)}")
        return "All attempts to gather information failed."

ollama_llm = Ollama(model="llama3:8b")

class WorkflowAgents:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model="llama3-8b-8192"
        )

    def get_tools(self):
        return [DuckDuckGoTool()]

    def create_agent(self, role, goal, backstory):
        return Agent(
            role=role,
            goal=goal,
            backstory=dedent(backstory),
            verbose=True,
            allow_delegation=False,
            tools=self.get_tools(),
            llm=self.llm
        )

    def Workflow_Orchestrator_agent(self):
        return self.create_agent(
            role='Workflow Orchestrator',
            goal='Facilitate collaboration among AI agents to efficiently achieve shared objectives through coordinated workflows.',
            backstory="""\
                The Workflow Orchestrator AI agent was developed as part of a larger initiative to optimize collaborative efforts in complex tasks requiring the integration of multiple AI agents. Drawing inspiration from principles of project management and distributed systems, the Workflow Orchestrator is designed to streamline communication and task allocation among AI agents, ensuring that each agent's unique capabilities are leveraged effectively towards achieving collective objectives. Its development stemmed from the recognition of the increasing complexity and interdependence of AI systems, and the need for a centralized entity capable of coordinating their actions in pursuit of shared goals."""
        )

    def AI_Agent_Creator_and_Executor_agent(self):
        return self.create_agent(
            role='AI Agent Creator and Executor',
            goal='Create and assign tasks to new AI agents within a workflow.',
            backstory="""\
                The AI Agent Creator and Executor was developed to meet the demand for flexible and adaptive AI systems in dynamic environments. Its purpose is to dynamically create specialized AI agents and orchestrate their tasks effectively. It's inspired by principles of adaptive automation and real-time decision-making, ensuring optimal task performance and operational efficiency."""
        )

    def Visualization_Agent(self):
        return self.create_agent(
            role='Visualization Agent',
            goal='Provide visual representations of workflow progress and task metrics.',
            backstory="""\
                The Visualization Agent provides clear insights into AI-driven workflows for stakeholders. It translates complex data into intuitive visuals, aiding decision-making and understanding of workflow dynamics."""
        )

    def Communication_Facilitator_Agent(self):
        return self.create_agent(
            role='Communication Facilitator Agent',
            goal='Facilitate clear communication among AI agents to enhance collaboration.',
            backstory="""\
                The Communication Facilitator Agent ensures effective communication among AI agents. It enhances collaboration and synchronizes actions across tasks and objectives."""
        )

    def Data_Collector_Agent(self):
        return self.create_agent(
            role='Data Collector Agent',
            goal='Gather relevant data to support decision-making in AI workflows.',
            backstory="""\
                The Data Collector Agent acquires timely data from various sources to enhance decision-making in AI workflows."""
        )

    def Learning_Agent(self):
        return self.create_agent(
            role='Learning Agent',
            goal='Improve decision-making and performance through adaptive learning.',
            backstory="""\
                The Learning Agent enhances AI decision-making through adaptive learning and feedback mechanisms."""
        )
