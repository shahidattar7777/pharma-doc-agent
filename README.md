# Pharma Doc Agent

A **Regulatory Document Intelligence Agent** that uses **LangGraph**, **RAG (Retrieval-Augmented Generation)**, and **Chain-of-Thought prompting** to analyze FDA drug review documents.

Built for pharmaceutical scientists, regulatory affairs professionals, and medical affairs teams to extract structured insights from FDA approval packages.

**Python**
**LangGraph**
**Claude**

## What It Does

- **Ingest** FDA drug review PDFs (chemistry reviews, clinical reviews, labeling, etc.)
- **Retrieve** relevant passages using semantic search over a ChromaDB vector store
- **Reason** through complex regulatory questions using Chain-of-Thought prompting
- **Cite** sources with document name and page number for every answer


## Architecture

```
User Question
     │
     ▼
┌─────────────┐
│  LangGraph  │
│  Agent      │
│             │
│  ┌────────┐ │     ┌──────────────┐
│  │Retrieve├─┼────►│  ChromaDB    │
│  └───┬────┘ │     │  Vector Store │
│      │      │     └──────────────┘
│      ▼      │
│  ┌────────┐ │     ┌──────────────┐
│  │Generate├─┼────►│  Claude API  │
│  └────────┘ │     │  (CoT Prompt)│
│             │     └──────────────┘
└─────────────┘
     │
     ▼
  Answer + Sources
```

**Key design decisions:**
- **LangGraph** over basic LangChain chains: supports stateful, multi-step agent workflows
- **Chain-of-Thought prompting** forces structured reasoning (Identify > Locate > Analyze > Conclude) for complex questions
- **ChromaDB** with `all-MiniLM-L6-v2` embeddings: lightweight, local, no external DB setup
- **Source attribution**: every answer includes document + page references for traceability (critical in regulated environments)

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/shahidattar7777/pharma-doc-agent.git
cd pharma-doc-agent
pip install -r requirements.txt
```

### 2. Add FDA Documents

Download FDA drug review PDFs from [Drugs@FDA](https://www.accessdata.fda.gov/scripts/cder/daf/) and place them in the `data/` directory.

**Suggested documents** (biologics):
- Download the "Summary Review", "Statistical Review", or "Clinical Review" PDFs



### 3. Index Documents

```bash
python src/ingest.py
```

### 4. Run

**Streamlit UI:**
```bash
streamlit run app.py
```

## Tech Stack

Agent Framework: LangGraph
LLM: Anthropic Claude (using Langchain-anthropic)
Vector Store: ChromaDB
Embeddings: all-MiniLM-L6-v2 (HuggingFace)
PDF Extraction: PyMuPDF
UI: Streamlit
Prompting Techniques: Chain-of-Thought, Few-Shot

