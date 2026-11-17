from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain
import re
import time


def invoke_with_retry(runnable, inputs, retries=3, wait=25):
    for attempt in range(retries):
        try:
            return runnable.invoke(inputs)
        except Exception as e:
            is_429 = getattr(e, "status_code", None) == 429 or "Error code: 429" in str(
                e
            )
            if is_429 and attempt < retries - 1:
                time.sleep(wait)
                continue
            raise


def run_research_pipeline(topic: str, progress_callback=None) -> dict:
    state = {}

    def update_progress(step, message):
        print(f"[STEP {step}] {message}")
        if progress_callback:
            progress_callback(step, message)

    def get_text(content):
        """Convert LangChain message content into text."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return "\n".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )

        return str(content)

    def extract_urls(content):
        """Extract and deduplicate URLs while preserving their order."""
        urls = re.findall(r"https?://[^\s<>\"']+", content)
        urls = [url.rstrip(".,;:!?)]}") for url in urls]
        return list(dict.fromkeys(urls))

    try:
        # SEARCH AGENT
        update_progress(1, "Search Agent is searching and scraping sources...")

        search_agent = build_search_agent()

        search_result = search_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"""
                        Research the following topic in detail: {topic}

                        Find reliable and recent sources using web_search.
                        You may perform multiple searches if necessary.

                        After selecting the most relevant sources,
                        scrape them using the available scraping tools.

                        Preserve all original source URLs.
                        """,
                    )
                ]
            }
        )

        messages = search_result["messages"]

        # COLLECT ACTUAL TOOL OUTPUTS
        search_outputs = []
        scraped_outputs = []

        for message in messages:
            tool_name = getattr(message, "name", None)
            content = get_text(message.content)

            if tool_name == "web_search":
                search_outputs.append(content)

            elif tool_name in ("scrape_url", "scrape_urls"):
                scraped_outputs.append(content)

        # # Preserve the Search Agent's final response too
        # final_search_response = get_text(messages[-1].content)

        # state["search_results"] = "\n\n".join(search_outputs + [final_search_response])

        state["search_results"] = "\n\n".join(search_outputs)

        update_progress(1, "Search completed successfully!")

        # EXTRACT SCRAPED SOURCES
        update_progress(2, "Collecting scraped research sources...")

        if not scraped_outputs:
            raise ValueError(
                "Search Agent did not invoke a scraping tool. "
                "Check its system prompt and tool execution."
            )

        scraped_sources = []

        for output in scraped_outputs:
            # scrape_urls separates individual sources using this delimiter
            source_blocks = output.split("\n\n--------------------\n\n")

            for source_content in source_blocks:
                urls = extract_urls(source_content)

                if not urls:
                    continue

                url = urls[0]

                # Avoid duplicate scraped sources
                if any(source["url"] == url for source in scraped_sources):
                    continue

                scraped_sources.append({"url": url, "content": source_content})

        if not scraped_sources:
            raise ValueError("No successfully scraped sources were returned.")

        state["sources"] = scraped_sources
        state["urls"] = [source["url"] for source in scraped_sources]

        print("\nScraped Sources:")
        for index, source in enumerate(scraped_sources, start=1):
            print(f"{index}. {source['url']}")

        # READER AGENT
        update_progress(2, "Reader Agent is analyzing extracted content...")

        reader_input = "\n\n".join(f"""
            SOURCE {index}
            Original URL: {source["url"]}

            Extracted Content:
            {source["content"]}
            """ for index, source in enumerate(scraped_sources, start=1))

        reader_agent = build_reader_agent()

        reader_result = invoke_with_retry(
            reader_agent,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""
                        Research topic: {topic}

                        Analyze the following scraped sources.

                        Extract:
                        - Article titles
                        - Main findings
                        - Important statistics
                        - Relevant evidence
                        - Research limitations

                        Preserve every original source URL.
                        Do not invent information.

                        SCRAPED SOURCES:

                        {reader_input}
                        """,
                    }
                ]
            },
        )

        state["scraped_content"] = get_text(reader_result["messages"][-1].content)

        update_progress(2, "Research analysis completed!")

        # WRITER CHAIN
        update_progress(3, "Writer is generating the report...")

        search_context = state["search_results"][:2500]

        research_combined = f"""
        SEARCH RESULTS:
        {state["search_results"][:1500]}

        READER ANALYSIS:
        {state["scraped_content"][:6000]}
        """

        state["report"] = invoke_with_retry(
            writer_chain, {"topic": topic, "research": research_combined}
        )

        update_progress(3, "Research report generated!")

        # CRITIC CHAIN
        update_progress(4, "Critic is evaluating the report...")

        critic_research = state["scraped_content"][:5000]

        state["feedback"] = invoke_with_retry(
            critic_chain, {"report": state["report"], "research": critic_research}
        )

        update_progress(4, "Report review completed!")

        return state

    except Exception as e:
        print(f"\nPipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")

    result = run_research_pipeline(topic)

    print("\nFINAL REPORT:\n")
    print(result["report"])

    print("\nCRITIC FEEDBACK:\n")
    print(result["feedback"])
