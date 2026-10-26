import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.callbacks import StreamlitCallbackHandler
import os
import wikipedia.wikipedia as wp

wp.USER_AGENT = "MySearchAgent/1.0 (youremai@examp.com)"

## Arxiv and wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name="Search")


st.title("LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi,I'm a chatbot who can search the web. How can I help you?",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=api_key, model_name="openai/gpt-oss-120b", streaming=True
    )
    tools = [search, arxiv, wiki]

    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the available tools to look up "
                "information when needed, then give a clear answer.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    search_agent = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.invoke(
            {"input": st.session_state.messages}, {"callbacks": [st_cb]}
        )["output"]
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
