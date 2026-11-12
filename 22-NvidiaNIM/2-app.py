import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

## Check the NVIDIA API key
if not os.getenv("NVIDIA_API_KEY"):
    st.error("NVIDIA_API_KEY not found. Add it to your .env file.")
    st.stop()

DATA_DIR = "./us_census"
MAX_PAGES = 30  # limit pages to keep embedding time and API usage low


def vector_embedding():
    if "vectors" not in st.session_state:
        if not os.path.isdir(DATA_DIR):
            st.error(f"Folder '{DATA_DIR}' not found. Add your PDF files there.")
            return
        embeddings = NVIDIAEmbeddings()
        docs = PyPDFDirectoryLoader(DATA_DIR).load()  ## Data ingestion
        if not docs:
            st.error(f"No PDF files found in '{DATA_DIR}'.")
            return
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        final_documents = splitter.split_documents(docs[:MAX_PAGES])  ## Chunking
        st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)


st.title("NVIDIA NIM Demo")
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question: {input}
""")

question = st.text_input("Enter your question from the documents")

if st.button("Documents Embedding"):
    with st.spinner("Embedding documents..."):
        vector_embedding()
    if "vectors" in st.session_state:
        st.write("Vector store DB is ready")

if question:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Documents Embedding' first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.perf_counter()
        response = retrieval_chain.invoke({"input": question})
        st.caption(f"Response time: {time.perf_counter() - start:.2f}s")

        st.write(response["answer"])

        with st.expander("Document Similarity Search"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------")
