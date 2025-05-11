import os
from dotenv import load_dotenv
from typing import Union, Literal, Any

from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import END

from state import TaskState


load_dotenv()

validator_model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="gpt-4.1",
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)


class AnswerFeedback(BaseModel):
    """Feedback to the final answer to the question"""

    answer_accepted: bool = Field(
        description="Flag if answer to the initial question was validated. "
            "If all format and other prerequisites were fulfilled, return True. "
            "Otherwise return false to get back to processing."
    )
    answer_feedback: str = Field(
        description="In case of incorrect answer, provide explanation what need to be changed."
    )


### Validator
# - Validates if the produced answer has requested format

def validator(state: TaskState):

    question = state["question"]
    answer = state["answer"]

    prompt = (
        "You are a format reviewer. "
        "Your role is to check whether the assistant's answer matches exactly the required structure. "
        "Except requirement directly in the question, the answer has to be short, concise and to the point. "
        "It should contain only the requested information without additional words. "
        "No additional puctuation and words are permitted. " # Added
        "Decline any leading up phrases. " # Added
        "Require perfecetion. "
        "Focus only on formatting, not on verifying the facts or contents. "
        "If the format does not match, briefly explain what should be adjusted.\n\n"
        "The question:\n"
        f"{question}\n\n"
        "Assistant's answer:\n"
        f"{answer}"
    )

    structured_validator_model = validator_model.with_structured_output(AnswerFeedback)

    response = structured_validator_model.invoke([HumanMessage(prompt)])

    if not response.answer_accepted:
        answer = ""

    return {
        # "past_steps": [("Output format validation", response.content)], # This will be done by replanner via assistant_messages
        "tool_messages": [],
        "past_steps": [("Answer validation", response.answer_feedback)],
        "answer": answer
    }

def validator_approval_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
) -> Literal["replanner", END]:
    """If the last answer is not correct, let the assistant rework it."""

    # Check if answer stayed filled or if it was revoked and deleted
    answer = state["answer"]

    if answer:
        return END
    return "replanner"