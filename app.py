"""
Streamlit UI for the Regulatory Document Intelligence Agent.

Run: streamlit run app.py
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.agent import query_agent
from src.ingest import load_vectorstore, build_vectorstore, CHROMA_DIR
import os


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pharma Doc Agent",
    page_icon="💊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("💊 Pharma Doc Agent")
    st.caption("Regulatory Document Intelligence powered by LangGraph + RAG")

    # st.divider()

    # API key input
    # api_key = st.text_input(
    #     "Anthropic API Key",
    #     type="password",
    #     help="Get your key at console.anthropic.com",
    # )

    api_key = "Please add your API KEY HERE"
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()

    # Vector store status
    st.subheader("📄 Document Store")
    if os.path.exists(CHROMA_DIR):
        try:
            vs = load_vectorstore()
            count = vs._collection.count()
            st.success(f"Loaded: {count} chunks indexed")
        except Exception as e:
            st.error(f"Error loading store: {e}")
    else:
        st.warning("No documents indexed yet")
        st.caption("Add PDFs to `data/` and run:")
        st.code("python src/ingest.py", language="bash")

    if st.button("🔄 Re-index Documents"):
        with st.spinner("Indexing..."):
            try:
                build_vectorstore(reset=True)
                st.success("Done!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # Example questions
    st.subheader("💡 Example Questions")
    examples = [
        "What were the primary endpoints in the Phase 3 clinical trial?",
        "What safety concerns did the FDA reviewer identify?",
        "Summarize the Chemistry, Manufacturing, and Controls (CMC) review.",
        "What was the FDA's benefit-risk assessment?",
        "Were there any post-marketing requirements or commitments?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending_question"] = ex

    st.divider()
    st.caption("Built with LangGraph • ChromaDB • Anthropic Claude • Streamlit")


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
st.title("Regulatory Document Intelligence Agent")
st.caption("Ask questions about FDA drug review documents using RAG + Chain-of-Thought reasoning")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Sources"):
                for s in message["sources"]:
                    st.caption(f"• {s['source']}, Page {s['page']}")

# Handle pending question from sidebar
if "pending_question" in st.session_state:
    prompt = st.session_state.pop("pending_question")
else:
    prompt = st.chat_input("Ask about FDA drug review documents...")

if prompt:
    # Check API key
    # if not os.environ.get("ANTHROPIC_API_KEY"):
    #     st.error("Please enter your Anthropic API key in the sidebar.")
    #     st.stop()

    # Check vector store
    if not os.path.exists(CHROMA_DIR):
        st.error("No documents indexed. Add PDFs to `data/` and run `python src/ingest.py`.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and analyzing documents..."):
            try:
                result = query_agent(
                    question=prompt,
                    chat_history=st.session_state.chat_history,
                )

                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("📎 Sources"):
                        for s in result["sources"]:
                            st.caption(f"• {s['source']}, Page {s['page']}")

                # Update state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )
                st.session_state.chat_history.extend(
                    [
                        HumanMessage(content=prompt),
                        AIMessage(content=result["answer"]),
                    ]
                )

            except Exception as e:
                st.error(f"Error: {e}")
                st.caption("Check your API key and that documents are indexed.")
