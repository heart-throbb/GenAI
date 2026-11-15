from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Prompt Template
prompt = ChatPromptTemplate.from_template("Explain {topic} in simple words")

# 2. Model
model = ChatGroq(model="openai/gpt-oss-20b")

# 3. Output Parser
parser = StrOutputParser()


chain = prompt | model | parser

result = chain.invoke("Machine Learning")
print(result)
