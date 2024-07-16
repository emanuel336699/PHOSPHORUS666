from crewai import Task
from textwrap import dedent
from crewai import Task
from textwrap import dedent

class AIAgentCreatorExecutorTasks():
    def create_workflow_task(self, agent, workflow_goal, task_description):
        task = Task(
            f"Goal: {workflow_goal}. Steps: {task_description}",
            agent=agent  # Assign the agent to the task
        )        
        expected_output = "Successful integration and functionality of the AI within the app."

        task = Task(
            agent=agent, 
            description=task_description,            expected_output=expected_output
        )

        return task
    
class Task:
    def __init__(self, agent, description, expected_output=None):
        self.assigned_agent = agent
        self.description = description
        self.expected_output = expected_output

def execute(self):
    for task in self.tasks:
        if hasattr(task, 'assigned_agent') and task.assigned_agent:
            task.assigned_agent.perform_task(task.description)
        else:
            print(f"Task '{task.description}' has no assigned agent or attribute missing.")


        