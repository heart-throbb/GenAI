import os
from dotenv import load_dotenv
from crewai import Agent
from langchain_openai import ChatOpenAI
from tools import yt_tool

load_dotenv()

# Fail clearly if the key is missing from .env
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Define the LLM explicitly (reads OPENAI_API_KEY from the environment)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7) 


## Create a senior blog content researcher

blog_researcher = Agent(
    role="Blog Researcher from Youtube Videos",
    goal="Get the relevant video transcription for the topic {topic} from the provided YT channel",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI, Data Science, Machine Learning "
        "and Gen AI, and providing suggestions."
    ),
    tools=[yt_tool],
    llm=llm,
    allow_delegation=True,
)

## creating a senior blog writer agent with YT tool

blog_writer = Agent(
    role="Blog Writer",
    goal="Narrate compelling tech stories about the video {topic} from YT video",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft "
        "engaging narratives that captivate and educate, bringing new "
        "discoveries to light in an accessible manner."
    ),
    tools=[yt_tool],
    llm=llm,
    allow_delegation=False,
)
