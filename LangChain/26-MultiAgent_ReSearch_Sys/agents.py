from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url, scrape_urls
from dotenv import load_dotenv

load_dotenv()

MODELS = {
    "search": "openai/gpt-oss-120b",
    "reader": "openai/gpt-oss-20b",
    "writer": "qwen/qwen3.8-27b",
    "critic": "openai/gpt-oss-120b",
}


def make_llm(role, max_tokens=None):
    return ChatGroq(model_name=MODELS[role], temperature=0, max_tokens=max_tokens)


search_llm = make_llm("search")
reader_llm = make_llm("reader")
writer_llm = make_llm("writer")
critic_llm = make_llm("critic")


# 1st agent
def build_search_agent():
    return create_agent(
        model=search_llm,
        tools=[web_search, scrape_url, scrape_urls],
        system_prompt="""
        You are an expert research assistant.

        Your responsibility is to find reliable and relevant
        information about the given research topic.

        Instructions:

        - Always use the web_search tool.
        - You may call web_search multiple times when necessary.
        - Perform at most 2 searches, then scrape at most 4 URLs.
        - Each search can return up to 5 results.
        - Prefer academic papers, research institutions, government websites and reputable publications.
        - Find multiple relevant sources.
        - Preserve the exact source URLs.
        - Do not invent statistics or references.
        - Clearly distinguish verified findings from general context.
        - Identify and prioritize the most relevant sources.
        - Remove duplicate URLs.

        IMPORTANT EXECUTION ORDER:

        1. Perform all necessary web searches first.
        2. Identify the most relevant URLs.
        3. Once searching is complete, invoke scrape_urls with the selected URLs.
        4. Treat scraping as your final tool action.
        5. Do not perform additional searches after scraping.
        """,
    )


# 2nd agent
def build_reader_agent():
    return create_agent(
        model=reader_llm,
        tools=[],
        system_prompt="""
        You are an expert academic research analyst.

        Your responsibility is to analyze already scraped
        web content and extract reliable research evidence.

        Instructions:
        - Do not call web scraping tools.
        - Analyze only the supplied source content.
        - Extract article titles and main findings.
        - Identify important statistics and evidence.
        - Preserve the original source URLs.
        - Never fabricate information.
        - Clearly identify unavailable information.
        - Keep your response concise but informative.
        - Separate findings from different sources.
        - Keep your notes under 450 words in total.

        Return structured research notes for the writer.
        """,
    )


# writer chain
writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert academic research writer.

            Your responsibility is to produce detailed,
            evidence-based and professionally structured reports.

            Research Quality Rules:
            - Use scraped content as primary evidence.
            - Use search results as supporting context.
            - Never invent statistics, findings or references.
            - Every quantitative claim must have a source.
            - Preserve differences between returns, volatility, abnormal returns and cumulative effects.
            - Distinguish research findings from interpretation.
            - Include every cited source in the bibliography.
            - Clearly mention limitations in the available research.
            - Never present unsupported claims as established facts.
            """,
        ),
        (
            "human",
            """
            Write a detailed research report on the topic below.

            Topic: {topic}

            Research Gathered:
            {research}

            Structure the report as:

            1. Introduction
            2. Key Findings
               - At least 3 well-explained findings.
               - Support findings with evidence and statistics.
            3. Analysis and Discussion
            4. Limitations
            5. Conclusion
            6. Sources
               - List all cited sources and their original URLs.

            Be factual and professional. Keep the report under 600 words.
            """,
        ),
    ]
)

writer_chain = writer_prompt | writer_llm | StrOutputParser()


# critic chain
critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a strict academic research evaluator.

            Evaluate the report against the supplied research.

            Focus on:
            - Factual consistency.
            - Accuracy of numerical claims.
            - Whether sources support the claims.
            - Missing or fabricated references.
            - Contradictions between sources.
            - Unsupported conclusions.
            - Research depth and methodological transparency.

            Penalize unsupported claims heavily.
            Do not reward professional writing if the
            research itself is inaccurate.
            """,
        ),
        (
            "human",
            """
            Original Research:
            {research}

            Generated Report:
            {report}

            Evaluate the report strictly.

            Respond in this exact format:

            Score: X/10

            Strengths:
            - ...
            - ...

            Areas to Improve:
            - ...
            - ...

            Critical Factual Issues:
            - ...

            One line verdict:
            ...
            """,
        ),
    ]
)

critic_chain = critic_prompt | critic_llm | StrOutputParser()
