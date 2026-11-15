# Generative AI Learning Repository

> A hands-on collection of notebooks and applications exploring LLMs, LangChain, retrieval, agents and vector databases.

This repository is a personal learning workspace for building with generative AI. It brings together experiments, notebooks, and small applications across several frameworks and providers. The examples are organized by topic so you can follow a learning path or jump straight into an area of interest.

## Explore the repository

| Area                      | What's inside                                                                                                                                                                                                                                      |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [LangChain](./LangChain/) | A 25-part learning path covering data ingestion and splitting, embeddings, vector stores, LCEL, chat history, RAG, tools, agents, summarization, SQL, hybrid search, and graph databases. Start with the [LangChain guide](./LangChain/README.md). |

## Getting started

Examples may have different Python dependencies, API requirements, and external services. Start with the README and dependency file for the specific area you want to explore. For LangChain, use the commands below from the repository root.

### Install with uv (recommended)

[`uv`](https://docs.astral.sh/uv/getting-started/installation/) is a fast Python package manager and can install packages faster than `pip` in many cases.

```bash
uv venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the LangChain examples' dependencies:

```bash
uv pip install -r LangChain/requirements.txt
```

### Or install with pip

```bash
python -m venv .venv
```

Activate the environment using the platform-specific command above, then install:

```bash
pip install -r LangChain/requirements.txt
```

The LangChain requirements file covers packages used across those examples; it does not install external services or model runtimes. Other areas, such as the AWS examples, may have their own dependencies and setup instructions.

## Before running an example

- Check the example's README, imports, and provider documentation for any extra setup.
- Some examples require API keys, a local model runtime such as Ollama, or services such as Neo4j.
- Keep credentials in environment variables or a local `.env` file. Never publish API keys, tokens, or database credentials.
- Notebooks and application scripts are independent experiments; run them from their relevant folder when needed.

## Learning path

For a guided introduction, begin with [LangChain](./LangChain/README.md): data ingestion → text splitting → embeddings → vector search → models and prompts → chat history → RAG → tools and agents → advanced retrieval and graph workflows.

## Disclaimer

This is a work in progress for learning and experimentation, not a production-ready library or application. APIs and provider integrations change over time, so examples may need updates to work with current package versions.
