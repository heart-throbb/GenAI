import validators, streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

## Streamlit APP
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜"
)
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")


## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

## GPT-oss Model USsing Groq API
llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error(
            "Please enter a valid Url. It can may be a YT video url or website url"
        )
    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url, add_video_info=False
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        },
                    )
                docs = loader.load()

                ## Split Documents into Smaller Chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3000, chunk_overlap=200
                )
                split_docs = text_splitter.split_documents(docs)
                # st.write(f"Total chunks: {len(split_docs)}")

                ## Map Reduce Summarization Chain
                chain = load_summarize_chain(llm, chain_type="map_reduce")
                output = chain.invoke({"input_documents": split_docs})
                output_summary = output["output_text"]

                st.success(output_summary)
        except Exception as e:
            st.error("Something went wrong!")
            st.write("Error type:", type(e).__name__)
            st.write("Error details:", str(e))
            print(repr(e))
