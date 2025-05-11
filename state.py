from typing import TypedDict, Annotated, Optional, Tuple, List

import operator
from langchain_community.tools import BaseTool
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class TaskState(TypedDict):

    # Input data
    question: str
    task_id: str
    file_name: Optional[str]
    file_type: Optional[str]
    file_content: Optional[bytes]
    tools_list: Optional[List[BaseTool]]

    # Output data
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    answer: str
    assistant_messages: list[AnyMessage] # Note: no automatic addition for better control
    tool_messages: list[AnyMessage] # Separated from assistant messages, because without automatic addition, tool will replace the list
    collected_knowledge: str