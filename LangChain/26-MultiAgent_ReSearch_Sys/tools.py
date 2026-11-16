from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv
from rich import print
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

tavily = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"), max_results=5)


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns titles, URLs and snippets."""
    results = tavily.invoke({"query": query})
    out = []
    for r in results["results"]:
        out.append(
            f"Title: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r['content'][:300]}\n"
        )
    return "\n----\n".join(out)


def scrape_single_url(url: str) -> str:
    """Scrape an individual webpage."""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(
            ["script", "style", "nav", "footer", "header", "aside", "noscript"]
        ):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else "Unknown title"
        content = soup.get_text(separator=" ", strip=True)
        return f"Title: {title}\n" f"URL: {url}\n" f"Content: {content[:1500]}"
    except Exception as e:
        return f"Could not scrape URL {url}: {str(e)}"


@tool(return_direct=True)
def scrape_url(url: str) -> str:
    """Scrape a single webpage and return clean text content."""
    return scrape_single_url(url)


@tool(return_direct=True)
def scrape_urls(urls: list[str]) -> str:
    """Scrape up to five webpages concurrently."""
    # Remove duplicates and select the five most relevant URLs
    urls = list(dict.fromkeys(urls))[:5]
    if not urls:
        return "No URLs provided for scraping."
    print(f"\nScraping {len(urls)} URLs concurrently...")
    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        results = list(executor.map(scrape_single_url, urls))
    return "\n\n--------------------\n\n".join(results)
