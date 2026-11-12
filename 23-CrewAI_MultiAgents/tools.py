from dotenv import load_dotenv

# Load the API key before the tool is created (the tool uses OpenAI embeddings)
load_dotenv()

from crewai_tools import YoutubeChannelSearchTool

# Initialize the tool with a specific Youtube channel handle to target your search
yt_tool = YoutubeChannelSearchTool(youtube_channel_handle="@krishnaik06")
