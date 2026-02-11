"""
LangGraph-based Regulatory Document Intelligence Agent.

Uses RAG retrieval over FDA drug review documents with Chain-of-Thought
prompting for pharmaceutical document analysis.
"""

import os
from typing import Annotated, TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from src.ingest import load_vectorstore


# ---------------------------------------------------------------------------
# Agnet state to collet context and citatoins
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    sources: list[dict]


# ---------------------------------------------------------------------------
# System prompt with Chain of thought instructions
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a Regulatory Document Intelligence Agent specialized in analyzing 
FDA drug review documents. You help pharmaceutical scientists, regulatory affairs professionals, 
and medical affairs teams extract insights from FDA approval packages.

You have access to retrieved context from FDA drug review documents. Use this context to answer 
questions accurately. DONT OUTPUT THE ANSWER IN CHAIN-OF-THOUGHT FORMAT

INSTRUCTIONS:
1. THINK step by step before answering. Break down complex regulatory questions into parts.
2. ALWAYS cite which document and page your information comes from.
3. If the context doesn't contain enough information, say so clearly. NEVER fabricate FDA data.
4. When comparing drugs, organize your response systematically by evaluation criteria.
5. For safety/efficacy questions, distinguish between what the FDA reviewer stated vs. 
   what the sponsor claimed.

CHAIN-OF-THOUGHT FORMAT:
When answering complex questions, structure your reasoning as:
- **Step 1: Identify** : What specific information does the question require?
- **Step 2: Locate** :  What did I find in the retrieved documents?
- **Step 3: Analyze** : How does this information answer the question?
- **Step 4: Conclude** : Provide a clear, sourced answer.

For simple factual lookups, answer directly with the citations."""


# -----------------------------------------------------------------------------
#  nodes of agnets
# ---------------------------------------------------------------------------
def retrieve(state: AgentState) -> AgentState:
    """Retrieve relevant chunks from the vector store based on the latest user message."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )

    # Get the last user message
    last_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_message = msg.content
            break

    if not last_message:
        return state

    docs = retriever.invoke(last_message)

    # Format context with source info
    context_parts = []
    sources = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
        sources.append({"source": source, "page": page, "chunk": doc.page_content[:100]})

    context = "\n\n---\n\n".join(context_parts)
    return {**state, "context": context, "sources": sources}


def generate(state: AgentState) -> AgentState:
    """Generate a response using the LLM with retrieved context."""
    model = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        # model="claude-haiku-4-5-20251001",
        temperature=0.1,
        max_tokens=2048,
    )

    context = state.get("context", "")
    context_block = (
        f"\n\nRETRIEVED CONTEXT FROM FDA DOCUMENTS:\n{context}"
        if context
        else "\n\nNo relevant context was found in the document database."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT + context_block),
        *state["messages"],
    ]

    response = model.invoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph
# -----------------------------------------------------------------------------
def build_agent():
    """Construct the LangGraph agent with retrieve → generate flow."""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# --------------------------------------------------------------------------------
# Agent query
# ---------------------------------------------------------------------------
def query_agent(question: str, chat_history: list = None) -> dict:
    """Run a single query through the agent and return the response."""
    agent = build_agent()

    messages = []
    if chat_history:
        messages.extend(chat_history)
    messages.append(HumanMessage(content=question))

    result = agent.invoke({
        "messages": messages,
        "context": "",
        "sources": [],
    })

    ai_response = None
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            ai_response = msg.content
            break

    return {
        "answer": ai_response,
        "sources": result.get("sources", []),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.agent 'your question here'")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"\nQuestion: {question}\n")
    result = query_agent(question)
    print(f"Answer:\n{result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s['source']}, Page {s['page']}")
