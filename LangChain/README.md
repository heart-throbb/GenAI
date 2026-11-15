# LangChain Learning Lab

> A hands-on collection of notebooks and small applications exploring LangChain, retrieval-augmented generation, chatbots, tools, and knowledge graphs.

This folder is organized as a numbered learning path, from loading and preparing data to building more complete AI applications. Most examples are independent, so you can also jump directly to a topic that interests you.

## What you'll find

| #   | Topic                                                                               | What it covers                                                             |
| --- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 01  | [Data ingestion](./1-DataIngestion/)                                                | Loading source material and preparing it for downstream processing.        |
| 02  | [Data transformation](./2-DataTransformer/)                                         | Splitting text, HTML, and JSON into useful chunks.                         |
| 03  | [Embeddings](./3-Embeddings/)                                                       | Creating embeddings with Ollama, Hugging Face, and OpenAI.                 |
| 04  | [Vector databases](./4-VectorDB%27s/)                                               | Working with FAISS and Chroma for vector search.                           |
| 05  | [Runnables](./5-Runnables/)                                                         | Composing sequence, parallel, and passthrough runnable patterns.           |
| 06  | [OpenAI with LangChain](./6-OpenAI--LangChain/)                                     | Getting started and building a simple OpenAI-powered app.                  |
| 07  | [Ollama with LangChain](./7-Ollama--LangChain/)                                     | Connecting LangChain applications to local Ollama models.                  |
| 08  | [Groq, LLMs, and LCEL](./8-Groq--SimpleLLM--LCEL/)                                  | Using Groq models and LangChain Expression Language (LCEL).                |
| 09  | [Conversation history](./9-Chatbot_with_Conv._History/)                             | Adding message history, prompt templates, and context windows to chatbots. |
| 10  | [Chat with retrievers](./10-Chat_with_Retrievers/)                                  | Retrieving relevant content to ground conversational answers.              |
| 11  | [Conversational Q&A with history](./11-Conv_Q%26A_Chatbot%2BHistory/)               | Combining question answering with conversation history.                    |
| 12  | [Chatbots](./12-Chatbots/)                                                          | Comparing chatbot examples using OpenAI and Ollama.                        |
| 13  | [Groq applications](./13-Groq/)                                                     | Building Groq-powered applications, including an embedding-based example.  |
| 14  | [PDF conversational RAG](./14-RAG_Pdf_Conv_Q%26A_History/)                          | Asking questions over PDFs while retaining chat context.                   |
| 15  | [Search, tools, and agents](./15-SearchEngine%20--%20Tools%26Agents/)               | Using tools and agents to build a search-oriented application.             |
| 16  | [Chat with SQL](./16-Chat_SQL/)                                                     | Exploring natural-language interaction with a SQL database.                |
| 17  | [Text summarization](./17-Text_Summarization/)                                      | Comparing Stuff, Map-Reduce, and Refine summarization strategies.          |
| 18  | [YouTube and website summarization](./18-YT_Vid_%26_WebSite_Content_Summarization/) | Summarizing video and web content.                                         |
| 19  | [Math problem solver](./19-Maths_Prob_Solver/)                                      | Building an application to help solve math problems.                       |
| 20  | [Hugging Face with LangChain](./20-HuggingFace-With%2BLangchain/)                   | Connecting Hugging Face models and endpoints to LangChain.                 |
| 21  | [Code assistant with Code Llama](./21-CodeAssistant%2BCodeLama/)                    | Exploring a code-assistant application with Code Llama.                    |
| 22  | [NVIDIA NIM](./22-NvidiaNIM/)                                                       | Streaming and application examples using NVIDIA NIM.                       |
| 23  | [CrewAI multi-agent workflows](./23-CrewAI_MultiAgents/)                            | Organizing multi-agent work with agents, tasks, and tools.                 |
| 24  | [Hybrid search](./24-Hybrid_Search/)                                                | Experimenting with combined keyword and vector retrieval.                  |
| 25  | [Graph databases](./25-GraphDB/)                                                    | Working with Neo4j, graph retrieval, and GraphRAG concepts.                |

## Suggested learning path

1. **Prepare data:** ingestion → transformation.
2. **Build retrieval foundations:** embeddings → vector databases → hybrid search.
3. **Compose applications:** runnables → model integrations → conversation history.
4. **Add knowledge and actions:** retrievers → RAG → tools and agents.
5. **Explore advanced patterns:** summarization → SQL → multi-agent workflows → graph databases.

## Getting started

The examples use different model providers, integrations, and services, so there is no single dependency list that fits every folder. Start with the topic you want to run, inspect its imports and any folder-specific instructions, then install the packages it needs in an isolated Python environment.

[`uv`](https://docs.astral.sh/uv/getting-started/installation/) is a fast Python package and project manager; it can install packages faster than `pip` in many cases. The commands below show both options. Run them from the repository root.

### Option 1: Install with uv (recommended)

```bash
uv venv .venv
```

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
uv pip install -r LangChain/requirements.txt
```

### Option 2: Install with pip

Create and activate a virtual environment with Python:

```bash
python -m venv .venv
```

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then install the dependencies:

```bash
pip install -r LangChain/requirements.txt
```

Once dependencies are installed, open a notebook with Jupyter or run its Python script. Some examples also need a provider API key, a local model runtime such as Ollama, or a service such as Neo4j.

> **Note:** this file covers Python packages across the LangChain examples. It does not install external services or model runtimes, and a particular example may need additional setup.

## API keys and local services

- Keep provider credentials in environment variables or a local `.env` file.
- Never commit API keys, database credentials, or other secrets.
- Check each example for provider-specific setup and service requirements.
- Local-model examples may require Ollama and a model to be installed separately.

## Notes

- These are learning examples, not a single packaged application; run each notebook or script from its own topic folder when appropriate.
- LangChain and provider integrations evolve quickly. If an example uses an older API, consult the current documentation for the relevant package.

## Repository-Owner

- `heart-throbb`
