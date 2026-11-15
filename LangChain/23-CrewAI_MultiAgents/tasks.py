from crewai import Task
from tools import yt_tool
from agents import blog_researcher, blog_writer

## Research Task
research_task = Task(
    description=(
        "Search the YouTube channel for videos about {topic}. "
        "Get detailed information about the video from the channel video."
    ),
    expected_output="A comprehensive 3 paragraphs long report based on the {topic} of video content.",
    tools=[yt_tool],
    agent=blog_researcher,
)

# Writing task with language model configuration
write_task = Task(
    description=(
        "Using the research findings, get the info from the youtube channel "
        "on the topic {topic} and write a blog post."
    ),
    expected_output="Summarize the info from the youtube channel video on the topic {topic} and create the content for the blog",
    tools=[yt_tool],
    agent=blog_writer,
    context=[research_task],  # writer receives the researcher's output
    async_execution=False,
    output_file="new-blog-post.md",  # Example of output customization
)
