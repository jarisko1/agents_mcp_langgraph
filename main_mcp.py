import os
from dotenv import load_dotenv
import asyncio

from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_mcp_adapters.client import MultiServerMCPClient


load_dotenv()

from utils import *
from tools import *
from state import *
from planner import *
from assistant import *
from  validator import *
from replanner import *

os.environ["LANGCHAIN_TRACING_V2"] = "true"


async def call_model():

    async with MultiServerMCPClient(
        {
            "hf_agents_tools": {
                "command": "python",
                "args": ["./mcp_server.py"],
                "transport": "stdio",
            },
        } # type: ignore
    ) as client:

        tools = client.get_tools()


        ### Graph Definition ###

        # Create the graph
        task_graph = StateGraph(TaskState)

        # Add nodes
        task_graph.add_node("planner", planner)
        task_graph.add_node("replanner", replanner)
        task_graph.add_node("validator", validator)
        task_graph.add_node("assistant", assistant)
        task_graph.add_node("tools", ToolNode(
            tools=tools,
            name="tools",
            messages_key="tool_messages"
        ))

        # Add edges
        task_graph.add_edge(START, "planner")
        task_graph.add_edge("planner", "assistant")
        task_graph.add_conditional_edges("assistant", tools_or_replanner_condition, ["tools", "replanner"]) # Continue with tools or proceed to replanner
        task_graph.add_edge("tools", "assistant")
        task_graph.add_conditional_edges("replanner", answer_provided_condition, ["validator", "assistant"]) # Validate answer continue working
        task_graph.add_conditional_edges("validator", validator_approval_condition, ["replanner", END]) # Go to END or back to assistant for rework

        compiled_graph = task_graph.compile()

        # from IPython.display import display, Image
        # display(Image(compiled_graph.get_graph(xray=True).draw_mermaid_png()))


        ### Execution ###

        question_generator = get_question(random=False)
        all_answers_payload = []

        for question, task_id, file_name in question_generator:

            if task_id != "cca530fc-4052-43b2-b130-b30968d8aa44":
                continue

            answers_payload = []

            print("Task ID:", task_id)
            print("Question:", question)
            print("File Name:", file_name or "No file")

            file_content = None
            file_type = None
            if file_name:
                file_type, file_content = read_file(file_name)

            task_answer = None

            # Try multiple times
            for i in range(1):
                try:
                    task_answer = await compiled_graph.ainvoke(
                        {
                            "question": question,
                            "task_id": "1",
                            "file_name": file_name,
                            "file_type": file_type,
                            "file_content": file_content,
                            "tools_list": tools,
                            "answer": "",
                            "tool_messages": [],
                            "assistant_messages": [],
                            "collected_knowledge": ""
                        },
                        {"recursion_limit": 30},
                    )
                except GraphRecursionError as e:
                    print("Recursion error, trying again...")
                    continue
                except Exception as e:
                    print("Other error, skipping question\nError details:", e)
                    break
                break

            if task_answer:
                print("Answer:", task_answer["answer"])
                print("-" * 50)

                # answers_payload.append({"task_id": task_id, "submitted_answer": task_answer["answer"]})
                answers_payload = [{"task_id": task_id, "submitted_answer": task_answer["answer"]}]
                all_answers_payload.extend(answers_payload)

                # TODO: Copy the final implementation to a Hugging Face space
                submission_data = {"username": "jarisko", "agent_code": "https://github.com/jarisko1/agents_mcp_langgraph", "answers": answers_payload}

                final_status = submit_answer(submission_data)
                print(final_status)
                print("=" * 100)

            else:
                print("No answer produced")
                print("=" * 100)

            # break



        ### Submit all produced answers ###

        # print("=" * 100)
        # all_submission_data = {"username": "jarisko", "agent_code": "https://github.com/jarisko1/agents_mcp_langgraph", "answers": all_answers_payload}
        # final_status = submit_answer(all_submission_data)
        # print(final_status)

        return all_answers_payload


if __name__ == "__main__":
    result = asyncio.run(call_model())
    print(result)