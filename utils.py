import requests
import os
from typing import Tuple
import mimetypes
import base64

import pandas as pd

from state import TaskState


api_url = "https://agents-course-unit4-scoring.hf.space"
questions_url = f"{api_url}/questions"
random_question_url = f"{api_url}/random-question"
files_url = f"{api_url}/files"
submit_url = f"{api_url}/submit"


def get_question(random: bool = True):
    """
    Get a random question from the API.
    Returns
        - question text
        - task id
        - binary file (if any)
        - file name (if any)
    """

    if random:
        used_url = random_question_url
    else:
        used_url = questions_url

    question_data = requests.get(used_url, timeout=15)
    question_data.raise_for_status()
    questions_json = question_data.json()

    # Create list if only 1 question was requested
    if not isinstance(questions_json, list):
        questions_json = [questions_json]

    for question_json in questions_json:

        question = question_json["question"]
        task_id = question_json["task_id"]

        file_name = question_json.get("file_name", None)

        if file_name:
            file_url = f"{files_url}/{task_id}"
            response = requests.get(file_url, timeout=15)
            response.raise_for_status()
            with open(os.path.join("tmp", file_name), "wb") as f:
                f.write(response.content)

        yield (question, task_id, file_name)


def read_file(file_name: str) -> Tuple[str, str | bytes]:
    """
    Returns file content as a string
    """

    file_path = os.path.join("tmp", file_name)

    mime_type, _ = mimetypes.guess_type(file_path)

    # Images
    if mime_type and mime_type.startswith("image/"):
        with open(file_path, "rb") as f:
            return "image", f.read()

    extension = file_name.split(".")[-1]
    if not extension:
        extension = "unknown"

    # Excel files
    if extension in ["xlsx", "xls"]:
        return extension, pd.read_excel(file_path).to_string()

    # mp3 files
    if extension in ["mp3"]:
        return "audio", ""

    # Anything else
    with open(file_path, "r") as f:
        return extension, f.read()


def submit_answer(submission_data: dict):
    """
    Submit the answers for checking.
    """

    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            # f"Submission Successful!\n"
            # f"User: {result_data.get('username')}\n"
            # f"Overall Score: {result_data.get('score', 'N/A')}% "
            # f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            # f"Message: {result_data.get('message', 'No message received.')}"
            result_data.get('score', 'N/A')
        )
        return final_status

    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        return status_message

    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        return status_message

    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        return status_message

    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        return status_message



def add_file_to_prompt(prompt:str, state:TaskState) -> str:
    """Adds file to provided prompt and returns enhanced prompt"""

    file_name = state["file_name"]
    file_type = state["file_type"]
    file_content = state["file_content"]

    enhanced_prompt = ""

    if file_name:

        if file_type == "image":
            image_base64 = base64.b64encode(file_content).decode("utf-8")

            enhanced_prompt = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                    },
                }
            ]

        elif file_type == "audio":

            prompt += (
                "You can use following audio file to answer the question:\n"
                f"{os.path.join('tmp', file_name)}"
            )

            enhanced_prompt = [{"type": "text", "text": prompt}]

        else:

            prompt += (
                "You can use following information to answer the question:\n"
                f"{file_content}"
            )

            enhanced_prompt = [{"type": "text", "text": prompt}]


    else:
        enhanced_prompt = [{"type": "text", "text": prompt}]

    return enhanced_prompt