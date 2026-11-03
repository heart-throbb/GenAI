import validators
import streamlit as st
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

## Streamlit app
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜"
)
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

## Get the Hugging Face API token and the URL (YT or website) to be summarized
with st.sidebar:
    hf_api_key = st.text_input("Huggingface API Token", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error(
            "Please enter a valid URL. It may be a YouTube video URL or a website URL"
        )
    else:
        try:
            with st.spinner("Waiting..."):
                ## Initialize the LLM only after the token has been validated
                endpoint = HuggingFaceEndpoint(
                    repo_id="meta-llama/Llama-3.1-8B-Instruct",
                    provider="deepinfra",
                    max_new_tokens=512,
                    temperature=0.7,
                    huggingfacehub_api_token=hf_api_key.strip(),
                )
                llm = ChatHuggingFace(llm=endpoint)

                ## Load the website or YouTube video data
                if is_youtube_url(generic_url):
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

                ## Chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                result = chain.invoke({"input_documents": docs})

                st.success(result["output_text"])
        except Exception as e:
            st.exception(e)
