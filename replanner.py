import os
from dotenv import load_dotenv
from typing import Union, Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from state import *
from utils import *
from planner import *

load_dotenv()

# replanning_model = AzureChatOpenAI(
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     # azure_deployment="o3-mini",
#     azure_deployment="gpt-4.1",
#     openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
# )

replanning_model = ChatOpenAI(
    model="gpt-4.1", # Due to Azure content filters
)

# replanning_model = ChatOllama(model="gemma3")


class Answer(BaseModel):
    """Response to user."""
    response: str

class Act(BaseModel):
    """Action to perform."""
    action: Union[Answer, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

### Replanner
# - Adjusts plan based on already done steps
# - Decides when the plan is finished

def replanner(state: TaskState):
    """Updates plan based on work already done"""

    question = state["question"]
    plan = state["plan"]
    past_steps = state["past_steps"]

    # Add last message from assistant to past steps
    if len(state["assistant_messages"]) > 0:
        last_message = state["assistant_messages"][-1]
        task = plan[0]
        past_steps.append((task, last_message.content))

    replanner_prompt = (
        "For the given objective, come up with a simple step by step plan. "
        "This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. "
        "The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.\n\n"
        "Your objective was this:\n"
        f"{question}\n\n"
        "Your original plan was this:\n"
        f"{plan}\n\n"
        "You have currently done the follow steps:\n"
        f"{past_steps}\n\n"
        "Update your plan accordingly. If no more steps are needed and you MUST return to the user, then respond with that. "
        "Do NOT come up with answer by yourself. "
        "Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan. "
        "Answer has to be formatted as specified in the question. "
        "Be very concise and output only the answer."
    )

    # Highlight feedback from validator (if present)
    validator_feedback = ""
    if len(past_steps) > 0:
        last_step_type, last_step_message = past_steps[-1]
        if last_step_type == "Answer validation":
            validator_feedback = last_step_message

    if validator_feedback:
        replanner_prompt += (
            "\n\nPay most attention to following feedback. If feedback mentions content issues, create new plan and let it rework.\n"
            f"{validator_feedback}"
        )

    # Add file to prompt
    system_content = add_file_to_prompt(replanner_prompt, state)

    replanner_messages = [
        SystemMessage(content=system_content),
    ]

    replanning_model_structured = replanning_model.with_structured_output(Act)

    replan = replanning_model_structured.invoke(replanner_messages)

    if isinstance(replan.action, Answer):
        return {
            "answer": replan.action.response,
            "assistant_messages": [],
            "tool_messages": []
        }
    else:
        return {
            "plan": replan.action.steps,
            "answer": "",
            "assistant_messages": [],
            "tool_messages": []
        }


def answer_provided_condition(state: TaskState)  -> Literal["validator", "assistant"]:
    """If answer was provided, proceed to validator. Otherwise give it back to rework."""

    if "answer" in state and state["answer"]:
        return "validator"
    else:
        return "assistant"