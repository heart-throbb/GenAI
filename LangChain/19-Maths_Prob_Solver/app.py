import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMMathChain, LLMChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import Tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.callbacks import StreamlitCallbackHandler

## Set up the Stramlit app
st.set_page_config(page_title="Text To Math Problem Solver And Data Serach Assistant")
st.title("Text To Math Problem Solver")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq APPI key to continue")
    st.stop()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key.strip(),
)

## Initializing the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find the various information on the topics mentioned",
)

## Initializa the MAth tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions. Only input mathematical expression need to be provided",
)

prompt = """
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template = PromptTemplate(input_variables=["question"], template=prompt)

## Combine all the tools into chain
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning_tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions.",
)

## initialize the agents

tools = [wikipedia_tool, calculator, reasoning_tool]

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful math problem solver and information assistant. "
            "Use the Calculator tool for calculations and Wikipedia for factual searches. "
            "Explain your answers clearly and step by step.",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=agent_prompt,
)

assistant_agent = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# maintaining Sesssions
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a Math chatbot who can answer all your maths questions",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

## LEts start the interaction
question = st.text_area(
    "Enter youe question:",
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?",
)

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.invoke(
                {"input": question},
                config={"callbacks": [st_cb]},
            )["output"]
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("### Response:")
            st.success(response)

    else:
        st.warning("Please enter the question")
