# 🔬 AI Research Assistant

A multi-agent research assistant built with LangChain, Groq, and Tavily. Enter a topic to search the web, scrape relevant sources, generate a structured report, and have a critic review the result.

The project is designed to work with Groq and Tavily free-tier accounts where available. Provider access, model availability, and usage limits can change.

## How it works

```
Topic ──► Search Agent ──► Reader Agent ──► Writer Chain ──► Critic Chain
          (Tavily +        (extracts        (structured      (scores report,
          web scraping)    findings)        report)          flags issues)
```

| Step | Component    | What it does                                                                             | Model                 |
| ---- | ------------ | ---------------------------------------------------------------------------------------- | --------------------- |
| 1    | Search Agent | Runs up to 2 web searches, then scrapes up to 4 selected URLs                            | `openai/gpt-oss-120b` |
| 2    | Reader Agent | Extracts titles, findings, statistics, and limitations while preserving source URLs      | `openai/gpt-oss-20b`  |
| 3    | Writer Chain | Writes a structured report with findings, analysis, limitations, conclusion, and sources | `qwen/qwen3.8-27b`    |
| 4    | Critic Chain | Scores the report out of 10 and lists strengths, improvements, and factual issues        | `openai/gpt-oss-120b` |

The model IDs are configured by role in `agents.py`. The Search Agent and Critic Chain currently share a model, so those stages also share that model's provider limits.

## Project structure

```
.
├── app.py        # Streamlit interface
├── pipeline.py   # Orchestrates the four steps, retry logic, context trimming
├── agents.py     # Model config, agents, writer and critic chains
├── tools.py      # Tavily search and web scraping tools
└── .env          # Local API keys (not committed)
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/heart-throbb/GenAI.git
cd GenAI/LangChain/26-MultiAgent_ReSearch_Sys
```

If you already cloned the repository, change into `LangChain/26-MultiAgent_ReSearch_Sys` from its root directory.

### 2. Create an environment and install the libraries

The project uses the following third-party Python libraries:

- `beautifulsoup4` — extracts readable text from webpages.
- `langchain` and `langchain-core` — agent, tool, prompt, and chain components.
- `langchain-groq` — Groq chat-model integration.
- `langchain-tavily` — Tavily web-search integration.
- `python-dotenv` — loads API keys from a local `.env` file.
- `requests` — fetches webpage content.
- `rich` — formats terminal output.
- `streamlit` — runs the web interface.

[`uv`](https://docs.astral.sh/uv/) is recommended: it is a fast Python package manager and typically installs and resolves packages faster than `pip`.

```bash
uv venv
```

Activate the virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the libraries with `uv`:

```bash
uv pip install \
  beautifulsoup4 \
  langchain \
  langchain-core \
  langchain-groq \
  langchain-tavily \
  python-dotenv \
  requests \
  rich \
  streamlit
```

You can also use Python's built-in `venv` with `pip`:

```bash
python -m venv .venv
```

Activate the environment using the platform-specific command above, then install the same libraries:

```bash
python -m pip install \
  beautifulsoup4 \
  langchain \
  langchain-core \
  langchain-groq \
  langchain-tavily \
  python-dotenv \
  requests \
  rich \
  streamlit
```

### 3. Add your API keys

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

Get credentials from the [Groq Console](https://console.groq.com) and [Tavily](https://tavily.com). Keep `.env` private; do not commit real credentials.

## Usage

**Web app**

```bash
streamlit run app.py
```

Enter a topic, click **Start Research**, and the results appear in four tabs: Final Report (downloadable as Markdown), Search Results, Scraped Research, and Critic Feedback.

**Command line**

```bash
python pipeline.py
```

## Changing models

All models are set in one place, the `MODELS` dictionary in `agents.py`:

```python
MODELS = {
    "search": "openai/gpt-oss-120b",
    "reader": "openai/gpt-oss-20b",
    "writer": "qwen/qwen3.8-27b",
    "critic": "openai/gpt-oss-120b",
}
```

Check current model limits in the [Groq Console](https://console.groq.com/settings/limits).

## Limiting output size (optional)

By default no output cap is set. If you hit token-limit errors, you can pass `max_tokens` to `make_llm` for any step in `agents.py`:

```python
search_llm = make_llm("search")                      # no cap (default)
reader_llm = make_llm("reader", max_tokens=1500)     # cap the Reader's output
writer_llm = make_llm("writer", max_tokens=2000)     # cap the Writer's output
critic_llm = make_llm("critic", max_tokens=1000)     # cap the Critic's output
```

`max_tokens` limits how long the model's **response** can be, not how much input it receives. A lower cap keeps generated responses shorter. To reduce the **input** size, lower the character slices in `pipeline.py` (`[:6000]` and `[:5000]`) or the page limit in `tools.py`.

Suggested starting values:

| Step   | `max_tokens` | Note                                                           |
| ------ | ------------ | -------------------------------------------------------------- |
| Search | leave unset  | Output is mostly short tool calls, and a cap could cut one off |
| Reader | 1500         | Notes are limited to about 450 words                           |
| Writer | 2000         | A 600-word report needs about 900 tokens, the rest is headroom |
| Critic | 1000         | Feedback is short                                              |

`gpt-oss` models spend part of this budget on hidden reasoning. If a step returns an empty or cut-off answer, raise its cap by 500.

## Managing provider limits

- **Model selection:** each pipeline role can be configured independently, although Search and Critic currently share a model and its provider limits.
- **Trimmed context:** scraped pages are capped at 1,500 characters each, search snippets at 300, and the Writer and Critic only receive capped portions of the Reader's notes.
- **Shorter outputs:** the Reader is limited to about 450 words and the Writer to about 600 words.
- **Retry on 429:** `invoke_with_retry` in `pipeline.py` waits and retries when Groq reports a per-minute rate limit.
- Avoid starting a second research run within a minute of the first.

## Troubleshooting

| Problem                                         | Cause and fix                                                                                                                                                          |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Error code: 413`                               | The request payload is too large; this is not a rate-limit response. Reduce the character caps in `pipeline.py` (the `[:6000]` and `[:5000]` slices) or in `tools.py`. |
| `Error code: 429`                               | Per-minute limit reached. The pipeline retries automatically. If it keeps failing, wait a minute, or move that step to another model in `MODELS`.                      |
| Report or feedback is empty or cut off          | `gpt-oss` models spend part of their output budget on hidden reasoning. Try a different model for that step.                                                           |
| `<think>...</think>` text appears in the report | Qwen is returning its reasoning. Strip that block from the output, or switch the Writer to `openai/gpt-oss-20b`.                                                       |
| "Search Agent did not invoke a scraping tool"   | The agent finished without scraping. Run again, or check the Search Agent prompt in `agents.py`.                                                                       |

## Limitations

- Scraping uses plain HTTP requests, so JavaScript-heavy or paywalled pages may return little or no text.
- Only the first 1,500 characters of each page are used, so reports reflect page openings (abstracts, summaries), not full papers.
- The critic checks the report against the Reader's notes, not the original pages, so errors made by the Reader are not caught.
- Free-tier rate limits can slow runs or cause occasional failures.

## Tech stack

Python, Streamlit, LangChain, Groq, Tavily, BeautifulSoup

## Repository Owner

`heart-throbb`
