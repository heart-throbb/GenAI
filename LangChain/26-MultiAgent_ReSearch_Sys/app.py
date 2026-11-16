import streamlit as st
from pipeline import run_research_pipeline
from datetime import datetime

# PAGE CONFIGURATION

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CUSTOM CSS

st.markdown(
    """
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(120deg, #172554, #312e81);
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 36px;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
}

.stage {
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}

.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 45px;
    font-weight: 600;
}

</style>
""",
    unsafe_allow_html=True,
)


# SESSION STATE

if "research_result" not in st.session_state:
    st.session_state.research_result = None

if "research_topic" not in st.session_state:
    st.session_state.research_topic = ""


# SIDEBAR

with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("Multi-Agent AI Research System")

    st.divider()

    st.markdown("### Pipeline Architecture")

    st.markdown("""
    1. 🔎 Search Agent
    2. 📖 Reader Agent
    3. ✍️ Writer Chain
    4. 🧐 Critic Chain
    """)

    st.divider()

    st.markdown("### About")

    st.write(
        "An AI-powered research assistant that searches "
        "the web, scrapes relevant sources, generates "
        "research reports, and evaluates their quality."
    )


# MAIN HEADER

st.markdown(
    """
<div class="hero">
    <h1>🔬 AI Research Assistant</h1>
    <p>
        Research anything with autonomous AI agents.
        Discover sources, extract insights, and generate
        comprehensive research reports.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# USER INPUT

st.subheader("What would you like to research?")

topic = st.text_area(
    "Research Topic",
    placeholder="Example: Latest advancements in Retrieval Augmented Generation (RAG) in 2026",
    height=110,
    key="topic_input",
)

run_button = st.button(
    "🚀 Start Research",
    type="primary",
    disabled=not topic.strip(),
)


# PIPELINE EXECUTION

if run_button:

    st.session_state.research_result = None
    st.session_state.research_topic = topic.strip()

    progress_bar = st.progress(0)

    status_container = st.status(
        "Initializing research pipeline...",
        expanded=True,
    )

    stage_names = {
        1: "🔎 Search Agent",
        2: "📖 Reader Agent",
        3: "✍️ Writer Chain",
        4: "🧐 Critic Chain",
    }

    stage_progress = {}

    for step, name in stage_names.items():
        stage_progress[step] = status_container.empty()
        stage_progress[step].markdown(f"⏳ **{name}** — Waiting...")

    completed_steps = set()

    def progress_callback(step, message):

        completed_steps.add(step)

        progress_bar.progress(len(completed_steps) / len(stage_names))

        stage_progress[step].markdown(f"🔄 **{stage_names[step]}** — {message}")

    try:

        with status_container:

            result = run_research_pipeline(
                topic.strip(),
                progress_callback=progress_callback,
            )

        progress_bar.progress(100)

        for step, name in stage_names.items():
            stage_progress[step].markdown(f"✅ **{name}** — Completed")

        status_container.update(
            label="Research completed successfully!",
            state="complete",
            expanded=False,
        )

        st.session_state.research_result = result

        st.success("Your research report is ready!")

    except Exception as e:

        status_container.update(
            label="Research pipeline failed",
            state="error",
            expanded=True,
        )

        st.error(f"An error occurred: {str(e)}")


# DISPLAY RESULTS

if st.session_state.research_result:

    result = st.session_state.research_result

    st.divider()

    st.header("📊 Research Results")

    st.caption(f"Topic: {st.session_state.research_topic}")

    # SOURCE SUMMARY

    sources = result.get("sources", [])

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Sources Scraped", len(sources))

    with col2:
        st.metric(
            "Search URLs Found",
            len(result.get("urls", [])),
        )

    # RESULTS TABS

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📝 Final Report",
            "🔎 Search Results",
            "📚 Scraped Research",
            "🧐 Critic Feedback",
        ]
    )

    # FINAL REPORT

    with tab1:

        st.subheader("Generated Research Report")

        report = result.get("report", "")

        report_text = report.content if hasattr(report, "content") else str(report)

        st.markdown(report_text)

        st.divider()

        st.download_button(
            label="⬇️ Download Research Report",
            data=report_text,
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # SEARCH RESULTS

    with tab2:

        st.subheader("Search Agent Output")

        search_results = result.get(
            "search_results",
            "No results available.",
        )

        st.markdown(str(search_results))

    # SCRAPED CONTENT

    with tab3:

        st.subheader("Research Sources")

        if sources:

            for index, source in enumerate(sources, start=1):

                with st.expander(f"Source {index}: {source['url']}"):

                    st.markdown(f"**Original URL:** {source['url']}")

                    st.markdown(source["content"])

        else:
            st.info("No scraped sources available.")

        st.divider()

        st.subheader("Reader Agent Analysis")

        scraped_content = result.get(
            "scraped_content",
            "No scraped content available.",
        )

        st.markdown(str(scraped_content))

    # CRITIC FEEDBACK

    with tab4:

        st.subheader("Critic's Review")

        feedback = result.get("feedback", "")

        feedback_text = (
            feedback.content if hasattr(feedback, "content") else str(feedback)
        )

        st.markdown(feedback_text)

    st.divider()

    st.caption(f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}")


# EMPTY STATE

elif not run_button:

    st.info(
        "👋 Enter a research topic above and start " "your AI-powered research journey."
    )
