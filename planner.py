import os
from dotenv import load_dotenv
from typing import List

from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from state import *
from utils import *


load_dotenv()

planning_model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    # azure_deployment="o3-mini",
    azure_deployment="gpt-4.1",
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)


### Planner
# - Starts the task processing
# - Creates plan

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


def planner(state: TaskState):
    """Creates step by step plan to finish defined task"""

    question = state["question"]

    planner_prompt = (
        "For the given objective, come up with a simple step by step plan. "
        "This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. "
        "The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps. "
        "You have tools for web search, audio transcription, youtube video transcription and python code execution at your disposal. "
        "For simple tasks you MUST not generate many steps. Single step plan is also good. "
    )

    # Add file to prompt
    system_content = add_file_to_prompt(planner_prompt, state)

    # Build messages
    planner_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=question),
    ]

    planning_model_structured = planning_model.with_structured_output(Plan)

    plan = planning_model_structured.invoke(planner_messages)

    return {
        "plan": plan.steps
    }