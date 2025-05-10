from dotenv import load_dotenv
import os

# YouTube transcription tool
from langchain.document_loaders import YoutubeLoader

# Python tool
from langchain_experimental.utilities import PythonREPL

# Search tools
from langchain_community.tools import Tool, DuckDuckGoSearchRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

# TTS tool
import assemblyai as aai


### Audio tool

def transcribe_audio(audio_file: str) -> str | None:
    """Transcribes provided audio file into text"""

    load_dotenv()

    aai.settings.api_key = os.environ["ASSEMBLY_AI_API_KEY"]

    config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)

    transcript = aai.Transcriber(config=config).transcribe(audio_file)

    if transcript.status == "error":
        print(f"Transcription failed: {transcript.error}")
        return ""

    return transcript.text


transcribe_audio_tool = Tool(
    name="audio_transcription",
    func=transcribe_audio,
    description="Returns text from an audio file"
)

### YouTube transcription tool
def transcribe_video(video_url: str) -> str:
    """Transcribes provided YouTube video into text"""

    text = ""

    loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=1htKBjuUWec", add_video_info=False)
    docs = loader.load()
    if len(docs) > 0:
        text = docs[0].page_content

    return text


transcribe_video_tool = Tool(
    name="video_transcription",
    func=transcribe_video,
    description="Returns text from YouTube video"
)


### Search tool

search_tool = TavilySearchResults(max_results=3)

# search_tool = DuckDuckGoSearchRun()

# search_wrapper = GoogleSerperAPIWrapper()
# search_tool = Tool(
#         name="search_tool",
#         func=search_wrapper.run,
#         description="useful for when you need to ask with search",
#     )


### Python tool

python_repl = PythonREPL()

python_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

tools = [
    search_tool,
    python_tool,
    transcribe_audio_tool,
    transcribe_video_tool,
]

