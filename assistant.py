import os
from dotenv import load_dotenv
from typing import Literal

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from state import TaskState
from utils import *


load_dotenv()

assistant_model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    # azure_deployment="gpt-4.1-mini",
    azure_deployment="gpt-4.1",
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)


### Assistant with tools
# - Can use tools
# - Executes next step in the plan

def assistant(state: TaskState):

    question = state["question"]
    tool_messages = state['tool_messages']
    assistant_messages = state['assistant_messages']
    plan = state["plan"]
    knowledge = state["collected_knowledge"]
    past_steps = state["past_steps"]
    tools = state["tools_list"]


    ### Process data returned from tool

    if len(tool_messages) > 0:
        # Add tool answer to assistant history (local for 1 task processing)
        assistant_messages.extend(tool_messages)
        # Collect gained knowledge (global for full processing)
        knowledge += "\n\n".join([str(tool_message.content) for tool_message in tool_messages]) + "\n\n"


    ### Starting new partial task, build the system prompt

    if len(assistant_messages) == 0:

        assistant_system_prompt = (
            f"You are an AI assistant anwering questions. "
            "Your goal is get closer to answering following question:\n"
            f"{question}\n\n"
            "When doing web search, be very specific and precise with your queries and specify all the details - language, year, etc. "
            "When generating Python code, do not continue, until you generate syntatically correct code."
        )

        if knowledge:
            assistant_system_prompt += f"\n\nYou can use following knowledge for your answer:\n{knowledge}"

        if past_steps:
            assistant_system_prompt += f"\n\nYou can use following history:\n{str(past_steps)}"

        # Add file to prompt
        system_content = add_file_to_prompt(assistant_system_prompt, state)

        # Note: System prompt, but human message, because it can contain images, which are not correctly interpreted in system prompt
        system_message = HumanMessage(
            content=system_content
        )

        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = (
            "\n\nFor the following plan:\n"
            f"{plan_str}\n\nYou are tasked with executing step {1}, {task}."
        )

        human_message = HumanMessage(content=task_formatted)

        assistant_messages.append(system_message)
        assistant_messages.append(human_message)


    ### Continue processing partial task

    response = assistant_model.bind_tools(tools, parallel_tool_calls=False).invoke(assistant_messages)
    assistant_messages.append(response)

    return {
        "tool_messages": [response],
        "assistant_messages": assistant_messages,
        "collected_knowledge": knowledge
    }


def tools_or_replanner_condition(state: TaskState) -> Literal["tools", "replanner"]:
    """
    If last message in the state is a tool call, navigate to the tools node, otherwise to the validator node.
    """

    tool_messages = state["tool_messages"]

    if len(tool_messages) > 0:
        tool_message = tool_messages[-1]
    else:
        return "replanner"

    if hasattr(tool_message, "tool_calls") and len(tool_message.tool_calls) > 0:
        return "tools"

    return "replanner"