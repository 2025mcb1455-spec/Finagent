# Finagent
Internal Hackathon for Inter-IIT Tech 15.0

FinAgent

Demo Video: [Insert YouTube Link Here]

A fault-tolerant, RAG-augmented multi-agent system powered by openai/gpt-oss-120b on Groq. It ingests financial documents and live market signals, processing them through a LangGraph state machine to synthesize structured, observable trading decisions.

Tech Stack

LLM Engine: openai/gpt-oss-120b (via Groq)

Embeddings: all-MiniLM-L6-v2 (via HuggingFace)

Orchestration: LangGraph & LangChain

Infrastructure: FAISS, yfinance, Pandas, Streamlit

Quickstart

Clone the repository and navigate to the directory:

git clone https://github.com/YourUsername/FinAgent.git
cd FinAgent


Install dependencies:

pip install -r requirements.txt


Run the application:

streamlit run app.py


Usage

Provide your Groq API Key in the sidebar. Upload contextual financial documents, then input live tickers or upload a CSV batch to trigger the analysis pipeline.
