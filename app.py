import streamlit as st
import os
import datetime
import tempfile
import json
import random
import pandas as pd
import yfinance as yf
from typing import TypedDict, List, Annotated, Dict, Any
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# --- RAG IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- UI CONFIGURATION ---
st.set_page_config(page_title="FinAgent | Multi-Agent System", layout="wide")
st.title("📈 FinAgent: Real-Time Investment Orchestrator")

# --- SESSION STATE INITIALIZATION ---
if "history" not in st.session_state:
    st.session_state.history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Cache the local embedding model
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

# --- SIDEBAR: SETTINGS & RAG UPLOADS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Groq API Key:", type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        
    st.markdown("---")
    st.header("📂 1. Knowledge Base Management")
    uploaded_files = st.file_uploader("Upload Sector Reports (PDF/TXT)", accept_multiple_files=True)
    
    if st.button("🧠 Ingest Documents to FAISS"):
        if uploaded_files:
            with st.spinner("Chunking and Embedding documents..."):
                all_docs = []
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        if file.name.endswith(".pdf"):
                            loader = PyPDFLoader(tmp_path)
                        else:
                            loader = TextLoader(tmp_path)
                            
                        docs = loader.load()
                        for d in docs:
                            d.metadata["source"] = file.name 
                        all_docs.extend(docs)
                    finally:
                        os.unlink(tmp_path)
                
                if all_docs:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    splits = text_splitter.split_documents(all_docs)
                    
                    if st.session_state.vector_store is None:
                        st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                    else:
                        st.session_state.vector_store.add_documents(splits)
                        
                    st.success(f"✅ Embedded {len(splits)} chunks into Vector Store!")
        else:
            st.warning("Please upload a file first.")
        
    st.markdown("---")
    st.header("📡 2. Signal Feed Config")
    feed_source = st.selectbox("Market Data Source", ["API Ticker Symbol", "Simulated Stream", "File Path (CSV)"])
    alert_threshold = st.slider("High-Confidence Alert Threshold", 0, 100, 70)

if not api_key:
    st.warning("👈 Please enter your Groq API Key in the sidebar to start.")
    st.stop()

# --- INITIALIZE LLM ---
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

# --- LANGGRAPH STATE & SCHEMAS ---
class DecisionArtifact(BaseModel):
    asset: str = Field(description="The ticker symbol or market sector")
    action: str = Field(description="Exactly: BUY, SELL, HOLD, or WATCH")
    confidence_level: str = Field(description="High, Medium, or Low")
    confidence_score: int = Field(description="Score from 0 to 100")
    evidence: List[str] = Field(description="2-3 bullet points of evidence")
    risk_flags: List[str] = Field(description="Contradictory indicators or risks")

class AgentState(TypedDict):
    normalized_signal: Dict[str, Any]
    retrieved_contexts: List[Dict[str, str]]
    classification: str
    hypothesis: str
    confidence_score: int
    decision_artifact: Dict[str, Any]
    agent_logs: Annotated[List[str], add_messages]
    errors: List[str]

class AnalysisOutput(BaseModel):
    classification: str = Field(description="bullish, bearish, or neutral")
    hypothesis: str = Field(description="Concise statement of opportunity/risk")
    confidence_score: int = Field(description="Integer 0-100")

# --- AGENT NODES ---
def analysis_node(state: AgentState) -> dict:
    signal = state.get("normalized_signal", {})
    contexts = state.get("retrieved_contexts", [])
    
    if contexts:
        context_str = "\n".join([f"- [{c['source']}]: {c['text']}" for c in contexts])
    else:
        context_str = "No specific reference documents provided. Rely on intrinsic market knowledge."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an elite quantitative analyst. Classify the market signal and provide a hypothesis based strictly on the current price and context."),
        ("user", "SIGNAL:\n{signal}\n\nCONTEXT:\n{context_str}")
    ])
    
    try:
        chain = prompt | llm.with_structured_output(AnalysisOutput)
        result = chain.invoke({"signal": str(signal), "context_str": context_str})
        return {
            "classification": result.classification,
            "hypothesis": result.hypothesis,
            "confidence_score": result.confidence_score,
            "agent_logs": [f"[AGENT: Analyst] Classification: {result.classification.upper()} | Confidence: {result.confidence_score} | Hypothesis formulated."]
        }
    except Exception as e:
        return {"errors": [f"[AGENT: Analyst] Error encountered: {str(e)}"]}

def synthesis_node(state: AgentState) -> dict:
    signal = state.get("normalized_signal", {})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Portfolio Committee Engine. Synthesize the analyst's hypothesis into a final decision artifact."),
        ("user", "ASSET: {asset}\nCLASSIFICATION: {classification}\nHYPOTHESIS: {hypothesis}\nCONFIDENCE: {confidence}")
    ])
    
    try:
        chain = prompt | llm.with_structured_output(DecisionArtifact)
        artifact = chain.invoke({
            "asset": signal.get("asset", "UNKNOWN"),
            "classification": state.get("classification", ""),
            "hypothesis": state.get("hypothesis", ""),
            "confidence": state.get("confidence_score", 0)
        })
        return {
            "decision_artifact": artifact.model_dump(),
            "agent_logs": [f"[AGENT: Portfolio Synthesizer] Artifact constructed. Final Action Tool Called: {artifact.action}"]
        }
    except Exception as e:
        return {"errors": [f"[AGENT: Portfolio Synthesizer] Error encountered: {str(e)}"]}

# --- BUILD GRAPH ---
graph_builder = StateGraph(AgentState)
graph_builder.add_node("analyze", analysis_node)
graph_builder.add_node("synthesize", synthesis_node)
graph_builder.set_entry_point("analyze")
graph_builder.add_edge("analyze", "synthesize")
graph_builder.add_edge("synthesize", END)
fin_agent = graph_builder.compile()

# --- MAIN DASHBOARD LAYOUT (WITH TABS) ---
# ADDED THE 3RD TAB HERE
tab_dash, tab_debug, tab_chat = st.tabs(["📈 Live Dashboard", "🔍 RAG Debug & Search", "💬 Chat with FinAgent"])

with tab_dash:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📡 Live Signal Input")
        
        tickers_to_process = []
        prices_to_process = {}
        
        if feed_source == "API Ticker Symbol":
            asset_input = st.text_input("API Ticker Symbol(s) - Comma separated:", value="RELIANCE.NS, ZOMATO.NS, NVDA")
            st.info("🟢 **Live Mode Active:** Fetching real-time market valuations.")
            if asset_input:
                tickers_to_process = [t.strip().upper() for t in asset_input.split(",") if t.strip()]
                
        elif feed_source == "Simulated Stream":
            asset_input = st.text_input("Simulated Ticker:", value="NVDA")
            price_input = st.number_input("Simulated Price (₹ or $)", value=132.40)
            if asset_input:
                tickers_to_process = [asset_input.strip().upper()]
                prices_to_process[tickers_to_process[0]] = price_input
                
        elif feed_source == "File Path (CSV)":
            st.info("📂 Upload a CSV with a 'Ticker' column (and optional 'Price' column).")
            uploaded_signal_file = st.file_uploader("Upload CSV File", type=["csv"])
            if uploaded_signal_file:
                try:
                    df = pd.read_csv(uploaded_signal_file)
                    if 'Ticker' in df.columns:
                        for _, row in df.iterrows():
                            tick = str(row['Ticker']).strip().upper()
                            tickers_to_process.append(tick)
                            if 'Price' in df.columns and pd.notna(row['Price']):
                                prices_to_process[tick] = float(row['Price'])
                        st.success(f"Loaded {len(tickers_to_process)} tickers from file.")
                    else:
                        st.error("CSV format invalid. Ensure there is a 'Ticker' header.")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        if st.button("🚀 Run FinAgent Pipeline", type="primary", use_container_width=True):
            if not tickers_to_process:
                st.error("Please provide at least one valid ticker symbol or upload a valid CSV.")
            else:
                for current_asset in tickers_to_process:
                    with st.spinner(f"Processing {current_asset}..."):
                        
                        dynamic_run_logs = []
                        
                        if current_asset in prices_to_process:
                            current_price = prices_to_process[current_asset]
                            currency_symbol = "₹" if current_asset.endswith(".NS") or current_asset.endswith(".BO") else "$"
                            dynamic_run_logs.append(f"[SYSTEM] Local/CSV price utilized: {currency_symbol}{current_price}")
                        else:
                            actual_price = None
                            ticker_data = yf.Ticker(current_asset)
                            try:
                                actual_price = ticker_data.fast_info['last_price']
                            except Exception:
                                pass
                                
                            if actual_price is None:
                                try:
                                    hist = ticker_data.history(period="1mo")
                                    if not hist.empty:
                                        actual_price = hist['Close'].iloc[-1]
                                except Exception:
                                    pass
                                    
                            if actual_price is None or str(actual_price) == "nan":
                                st.toast(f"⚠️ Yahoo API blocked Colab IP. Using mock price for {current_asset}", icon="⚠️")
                                actual_price = random.uniform(100.50, 3000.00)
                                dynamic_run_logs.append(f"[SYSTEM FALLBACK EVENT] API Fetch Failed. Injected simulated proxy price: {actual_price:.2f}")
                            else:
                                dynamic_run_logs.append(f"[SYSTEM] Live Yahoo Finance API fetch successful.")
                                
                            current_price = round(float(actual_price), 2)
                            currency_symbol = "₹" if current_asset.endswith(".NS") or current_asset.endswith(".BO") else "$"

                        retrieved_contexts = []
                        if st.session_state.vector_store is not None:
                            query = f"Market outlook, valuation analysis, and operational risks for {current_asset}"
                            docs = st.session_state.vector_store.similarity_search(query, k=2)
                            retrieved_contexts = [{"source": d.metadata.get("source", "Unknown"), "text": d.page_content} for d in docs]
                        
                        if retrieved_contexts:
                            dynamic_run_logs.append(f"[TOOL: FAISS Vector DB] Retrieved {len(retrieved_contexts)} exact RAG passages:")
                            for idx, c in enumerate(retrieved_contexts):
                                snippet = c['text'][:120].replace('\n', ' ') + "..."
                                dynamic_run_logs.append(f"  -> Source {idx+1} Attribution: {c['source']} | Passage: {snippet}")
                        else:
                            dynamic_run_logs.append("[TOOL: FAISS Vector DB] 0 passages found. ZERO-SHOT FALLBACK: Relying entirely on pre-trained intrinsic data weights.")

                        initial_state = {
                            "normalized_signal": {"asset": current_asset, "price": f"{currency_symbol}{current_price}"},
                            "retrieved_contexts": retrieved_contexts,
                            "agent_logs": dynamic_run_logs,
                            "errors": []
                        }
                        
                        final_state = fin_agent.invoke(initial_state)
                        
                        if final_state.get("errors"):
                            st.error(f"Pipeline error on {current_asset}: {final_state['errors'][-1]}")
                        else:
                            artifact = final_state.get("decision_artifact", {})
                            artifact["timestamp"] = datetime.datetime.now().strftime("%H:%M:%S")
                            artifact["price_evaluated"] = f"{currency_symbol}{current_price}"
                            
                            # WE NOW SAVE THE INTERNAL HYPOTHESIS SO THE CHATBOT CAN READ IT LATER
                            artifact["internal_hypothesis"] = final_state.get("hypothesis", "No hypothesis generated.")
                            
                            raw_logs = final_state.get("agent_logs", [])
                            artifact["logs"] = [msg.content if hasattr(msg, 'content') else str(msg) for msg in raw_logs]
                            
                            st.session_state.history.insert(0, artifact)
                            
                            if artifact.get("confidence_score", 0) >= alert_threshold:
                                st.toast(f"🚨 HIGH CONFIDENCE ALERT: {artifact.get('action')} {current_asset}", icon="🚨")
                
                st.success("🎉 Processing Complete!")

    with col2:
        st.subheader("📋 Decision Artefacts Feed")
        if not st.session_state.history:
            st.info("No decisions generated yet. Run the pipeline!")
            
        for i, art in enumerate(st.session_state.history):
            action = art.get('action', 'WATCH')
            action_color = "🟢" if action == 'BUY' else "🔴" if action == 'SELL' else "🟠"
            is_high_confidence = art.get("confidence_score", 0) >= alert_threshold
            
            with st.container(border=True):
                if is_high_confidence:
                    if action == "BUY":
                        st.success(f"🚨 **HIGH CONFIDENCE DETECTED:** Strong Buy Signal for {art.get('asset')}")
                    elif action == "SELL":
                        st.error(f"🚨 **HIGH CONFIDENCE DETECTED:** Strong Sell Signal for {art.get('asset')}")
                    else:
                        st.warning(f"🚨 **HIGH CONFIDENCE DETECTED:** Watch/Hold Signal for {art.get('asset')}")

                st.markdown(f"### {art.get('asset')} {action_color} {action}")
                st.caption(f"Time: {art.get('timestamp')} | **Evaluated Value:** {art.get('price_evaluated')} | **Confidence:** {art.get('confidence_level')} ({art.get('confidence_score')}/100)")
                
                st.markdown("**Key Evidence:**")
                for bullet in art.get('evidence', []):
                    st.markdown(f"- {bullet}")
                    
                st.markdown(f"**Risk Flags:** {', '.join(art.get('risk_flags', ['None']))}")
                
                with st.expander("🔍 Observability: Agent Trace, Sources & Fallbacks"):
                    for log in art.get("logs", []):
                        st.text(log)

        # --- Exportable Report ---
        if st.session_state.history:
            st.markdown("---")
            clean_history = []
            for entry in st.session_state.history:
                clean_entry = entry.copy()
                if "logs" in clean_entry:
                    clean_entry["logs"] = [msg.content if hasattr(msg, 'content') else str(msg) for msg in clean_entry["logs"]]
                clean_history.append(clean_entry)
                
            json_report = json.dumps(clean_history, indent=4)
            st.download_button(
                label="💾 Download Audit Report (JSON)",
                data=json_report,
                file_name=f"FinAgent_Audit_{datetime.datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )

# --- TAB 2: RAG DEBUGGER INTERFACE ---
with tab_debug:
    st.subheader("🔍 Knowledge Base Search & Evaluation")
    st.markdown("Query the FAISS vector database directly to inspect retrieved chunks and evaluate embedding quality.")
    
    search_query = st.text_input("Enter search query (e.g., 'What are the margin risks for NVDA?'):")
    num_results = st.slider("Number of chunks to retrieve (k):", min_value=1, max_value=10, value=3)
    
    if st.button("🔎 Run Vector Search", type="primary"):
        if st.session_state.vector_store is None:
            st.error("⚠️ The Vector Store is empty. Please upload and ingest a document first using the sidebar.")
        elif not search_query:
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching FAISS database..."):
                docs = st.session_state.vector_store.similarity_search(search_query, k=num_results)
                
                if not docs:
                    st.warning("No relevant chunks found.")
                else:
                    st.success(f"Retrieved {len(docs)} document chunks.")
                    for i, doc in enumerate(docs):
                        with st.expander(f"📄 Result {i+1} | Source: {doc.metadata.get('source', 'Unknown')}", expanded=True):
                            st.write(doc.page_content)

# --- TAB 3: CONVERSATIONAL INTERFACE (BONUS FEATURE) ---
with tab_chat:
    st.subheader("💬 Chat with FinAgent")
    st.markdown("Ask follow-up questions to challenge the AI's thesis on a specific generated artefact.")
    
    if not st.session_state.history:
        st.info("No decisions generated yet. Run the pipeline on the Live Dashboard first!")
    else:
        # Create a dictionary to easily reference historical decisions by a friendly dropdown name
        options = {f"{art['asset']} - {art['action']} ({art['timestamp']})": art for art in st.session_state.history}
        selected_option = st.selectbox("Select a decision to discuss:", list(options.keys()))
        selected_art = options[selected_option]
        
        # Unique chat memory key for the specific dropdown selection so contexts don't mix
        chat_key = f"chat_{selected_option}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []
            
        # Draw existing chat history
        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Chat Input Box
        if prompt_text := st.chat_input(f"e.g., Why did you rate {selected_art['asset']} as {selected_art['confidence_level']} confidence?"):
            
            # Display user message
            st.session_state[chat_key].append({"role": "user", "content": prompt_text})
            with st.chat_message("user"):
                st.markdown(prompt_text)
                
            # Process AI Response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing artefact history..."):
                    # We inject the exact context of the specific decision into the LLM prompt!
                    system_prompt = f"""You are the quantitative analyst who just recommended to {selected_art['action']} {selected_art['asset']}.
                    You evaluated it at a price of {selected_art['price_evaluated']} and gave it a confidence score of {selected_art['confidence_score']}/100.
                    
                    Here was your internal hypothesis: "{selected_art.get('internal_hypothesis', 'None recorded.')}"
                    Here was your main evidence: {', '.join(selected_art['evidence'])}
                    Here were the risks you flagged: {', '.join(selected_art['risk_flags'])}
                    
                    A portfolio manager is asking you a follow-up question. Answer concisely, professionally, and defend your thesis based on the facts provided above."""
                    
                    chat_prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("user", "{user_input}")
                    ])
                    
                    chain = chat_prompt_template | llm
                    response = chain.invoke({"user_input": prompt_text})
                    
                    st.markdown(response.content)
                    
            # Save AI response to chat history
            st.session_state[chat_key].append({"role": "assistant", "content": response.content})
