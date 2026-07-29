# Multi-Agent AI College Assistant

An intelligent **Agentic AI-powered College Assistant** built using **LangGraph, LangChain, Retrieval-Augmented Generation (RAG), and Streamlit**.

The system uses multiple specialized AI agents to understand student queries, route them to the appropriate agent, retrieve relevant information from college documents, and generate context-aware responses.

This project demonstrates the implementation of **multi-agent AI workflows, intelligent routing, RAG pipelines, vector search, tool-based agents, and LLM-powered applications**.

---

## Overview

College-related information is often scattered across academic handbooks, fee structures, regulations, and other documents.

The **Multi-Agent AI College Assistant** provides a conversational interface where students can ask questions in natural language instead of manually searching through multiple documents.

Example queries:

- "What are the academic requirements for this semester?"
- "What is the fee structure?"
- "What are the rules mentioned in the academic handbook?"
- "How much do I need to pay?"
- "Explain the academic regulations."

The system analyzes the query and routes it to the appropriate specialized AI agent.

---

## Features

- Multi-Agent Architecture
- LLM-powered Query Understanding
- Intelligent Agent Routing
- Retrieval-Augmented Generation (RAG)
- Semantic Document Retrieval
- LangGraph-based Agent Workflow
- PDF Knowledge Base
- Conversational User Interface
- Context-Grounded Responses
- Interactive Streamlit Application

---

## How It Works

Instead of using a single AI agent for every task, the application uses specialized agents responsible for different types of college-related queries.

When a student submits a question:

1. The query is received through the Streamlit interface.
2. The system analyzes the intent of the query.
3. A routing mechanism determines which specialized agent should handle the request.
4. The selected agent retrieves relevant information from its knowledge source.
5. Retrieved context is provided to the LLM.
6. The LLM generates a response grounded in the retrieved college documents.
7. The final answer is returned to the student.

---

## System Architecture

```text
                     ┌───────────────────┐
                     │   Student Query   │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │   Router Agent    │
                     │ Query Classifier  │
                     └─────────┬─────────┘
                               │
                   ┌───────────┴───────────┐
                   │                       │
                   ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │  Academic Agent   │   │     Fee Agent     │
        └─────────┬─────────┘   └─────────┬─────────┘
                  │                       │
                  ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │ Academic Handbook │   │   Fee Structure   │
        │    RAG System     │   │    RAG System     │
        └─────────┬─────────┘   └─────────┬─────────┘
                  │                       │
                  └───────────┬───────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │        LLM        │
                    │ Response Generator│
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   Final Response  │
                    └───────────────────┘
```

---

## RAG Pipeline

The application uses **Retrieval-Augmented Generation (RAG)** to reduce hallucinations and generate answers based on actual college documents.

```text
PDF Documents
      │
      ▼
Document Loading
      │
      ▼
Text Chunking
      │
      ▼
Embedding Generation
      │
      ▼
Vector Store
      │
      ▼
Similarity Search
      │
      ▼
Relevant Context
      │
      ▼
LLM
      │
      ▼
Grounded Response
```

Instead of relying entirely on the LLM's internal knowledge, relevant document sections are retrieved and supplied as context before generating the final answer.

---

## Agentic Workflow

The project uses **LangGraph** to manage the flow between different components of the AI system.

The routing layer determines which agent is best suited for the user's query.

### Academic Agent

Responsible for queries related to:

- Academic regulations
- Academic policies
- Course-related information
- Student guidelines
- Academic requirements

The agent retrieves information from the academic handbook before generating its response.

### Fee Agent

Responsible for queries related to:

- Fee structures
- Payment-related information
- College fees
- Financial information available in the knowledge base

The agent retrieves relevant information from the fee structure document.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core application development |
| **LangChain** | LLM application and RAG components |
| **LangGraph** | Agent orchestration and workflow management |
| **LLM API** | Natural language understanding and generation |
| **Vector Search / Embeddings** | Semantic document retrieval |
| **Streamlit** | Interactive web interface |
| **PDF Documents** | Knowledge base |

---

## Project Structure

```text
Multi-AI-Agent-College-Assistant/
│
├── app.py
├── chat_v3.py
├── academics_handbook.pdf
├── fee_structure.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

> The project can be further modularized into separate agent, retrieval, workflow, and utility components as it grows.

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd Multi-AI-Agent-College-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**macOS/Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root.

```env
YOUR_API_KEY=your_api_key_here
```

Replace the variable name with the API key required by the LLM provider used in the project.

> Never commit your `.env` file or API keys to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local Streamlit URL displayed in your terminal.

---

## Example Queries

Try asking the assistant:

```text
What is the fee structure?

What are the important academic regulations?

Explain the academic guidelines.

What information is available regarding student fees?

Summarize the important rules from the academic handbook.
```

The router analyzes each request and sends it to the appropriate specialized agent.

---

## Key Concepts Demonstrated

This project demonstrates practical implementation of:

- Agentic AI
- Multi-Agent Systems
- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Prompt Engineering
- Semantic Search
- Embeddings
- Vector Retrieval
- Agent Routing
- LangGraph Workflows
- Conversational AI

---

## Why Multi-Agent Architecture?

A traditional chatbot typically sends every query through the same processing pipeline.

This project instead separates responsibilities between specialized agents.

For example:

```text
"What are the academic regulations?"
            │
            ▼
      Academic Agent


"What is the semester fee?"
            │
            ▼
         Fee Agent
```

This architecture makes the system easier to extend.

Additional agents could later be introduced for:

```text
Placement Agent
      │
Library Agent
      │
Hostel Agent
      │
Examination Agent
      │
Attendance Agent
```

without redesigning the entire application.

---

## Future Improvements

- [ ] Add more specialized college agents
- [ ] Implement conversation memory
- [ ] Add source citations to generated responses
- [ ] Improve retrieval using hybrid search
- [ ] Add reranking for retrieved documents
- [ ] Implement RAG evaluation
- [ ] Add guardrails for unsupported queries
- [ ] Add response confidence scoring
- [ ] Add persistent vector storage
- [ ] Containerize the application using Docker
- [ ] Add automated testing
- [ ] Deploy the application publicly

Potential future agents:

- Library Agent
- Examination Agent
- Placement Agent
- Hostel Agent
- Attendance Agent

---

## What I Learned

Through this project, I gained hands-on experience with:

- Designing multi-agent AI systems
- Building workflows using LangGraph
- Implementing Retrieval-Augmented Generation
- Working with embeddings and semantic retrieval
- Routing queries between specialized AI agents
- Integrating LLMs with external knowledge sources
- Building interactive AI applications using Streamlit
- Structuring agentic applications around specialized responsibilities

---

## Limitations

The current system operates on a limited set of college documents and therefore can only reliably answer questions supported by its knowledge base.

LLM-generated responses may occasionally be incomplete or inaccurate. For important academic or financial decisions, users should verify information against official college sources.

---

## Contributing

Contributions, suggestions, and improvements are welcome.

To contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your changes.
5. Open a pull request.

---

## License

This project is intended for educational and portfolio purposes.

