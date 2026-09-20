import json
import os
import chromadb
from google import genai
from google.genai import types
import streamlit as st
from pypdf import PdfReader

# 1. UI Setup
st.set_page_config(page_title="Maha Schemes Assistant", page_icon="🏛️", layout="wide")
st.title("🏛️ Maharashtra Education Schemes Assistant")

# 2. Setup SDK & Config
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

PRIMARY_MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.5-flash-lite"
HISTORY_FILE = "chat_history.json"
DB_PATH = "./chroma_db"

# 3. Persistent Memory Helpers
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(messages):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

if "messages" not in st.session_state:
    st.session_state.messages = load_history()

# 4. Database Persistence (Writing to Disk)
@st.cache_resource
def get_chroma_collection():
    # Upgrade to PersistentClient so your database survives restarts
    chroma_client = chromadb.PersistentClient(path=DB_PATH) 
    return chroma_client.get_or_create_collection(name="maharashtra_schemes")

collection = get_chroma_collection()

def process_uploaded_file(uploaded_file, chunk_size: int, overlap: int) -> int | bool:
    """
    Processes an uploaded text or PDF file, extracts its text,
    splits it into overlapping chunks, and upserts them into the vector DB.
    Returns number of chunks ingested, or False if no text was extracted.
    """
    text = ""
    # Extract text based on file type
    if uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "

    # If no extractable text, fail
    if not text.strip():
        return False

    # Collapse any excessive whitespace
    text = " ".join(text.split())

    # Keep overlap strictly smaller than chunk_size so the window always advances
    overlap = min(overlap, chunk_size - 1)

    # List to hold each chunk of text
    chunks = []
    start = 0

    # Chunking logic:
    # Each chunk has size `chunk_size`. The next chunk starts at `start = end - overlap`
    # so there is an `overlap` region shared with the previous chunk.
    # For example:
    #   1st chunk characters: [0:500]
    #   2nd chunk characters: [420:920]  (420 = 500 - 80, so 80 char overlap)
    #   3rd chunk: [840:1340], etc.
    # This helps preserve context across chunk boundaries.
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move start to right after previous chunk minus overlap, so chunks overlap
        start = end - overlap

    # Create unique chunk IDs using file name and chunk index
    file_prefix = uploaded_file.name.replace(" ", "_")
    ids = [f"{file_prefix}_chunk_{i}" for i in range(len(chunks))]

    # Use UPSERT to allow replacing/updating existing chunks if same file is uploaded again
    collection.upsert(
        documents=chunks,
        ids=ids
    )
    return len(chunks)

# 5. Sidebar Controls 
with st.sidebar:
    st.header("📂 Document Ingestion")
    st.caption(f"Currently tracking {collection.count()} document chunks on disk.")
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=1000, value=500, step=10)
    overlap = st.slider("Overlap", min_value=0, max_value=200, value=80, step=5)
    if overlap >= chunk_size:
        st.caption("Overlap is capped just below chunk size so chunks keep advancing.")

    uploaded_file = st.file_uploader("Upload Scheme Rules", type=["txt", "pdf"])
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            chunk_count = process_uploaded_file(uploaded_file, chunk_size, overlap)
            if chunk_count:
                st.success(f"✅ Ingested/Updated {chunk_count} chunks from {uploaded_file.name}")
            else:
                st.error("Could not extract text from this file.")
                
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.rerun()

# --- DUAL TABS INTERFACE ---
tab1, tab2 = st.tabs(["💬 Scheme Assistant", "📋 Agentic Eligibility Evaluator"])

# ==========================================
# TAB 1: CONVERSATIONAL RAG
# ==========================================
with tab1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the uploaded schemes..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_history(st.session_state.messages)

        with st.chat_message("assistant"):
            if collection.count() == 0:
                st.warning("⚠️ Please upload a document in the sidebar first.")
                st.stop()
                
            search_query = prompt

            if len(st.session_state.messages) > 1:
                with st.spinner("Resolving context..."):
                    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages[-5:]])
                    rewrite_prompt = f"Rewrite LAST_QUESTION into a standalone search query using the HISTORY. Return only the query.\n\nHISTORY:\n{history_text}\n\nLAST_QUESTION: {prompt}"

                    for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
                        try:
                            res = client.models.generate_content(
                                model=model_name,
                                contents=rewrite_prompt,
                                config=types.GenerateContentConfig(http_options=types.HttpOptions(timeout=5000)),
                            )
                            if res.text and res.text.strip():
                                search_query = res.text.strip()
                                break
                        except Exception:
                            continue

            st.caption(f"*(Search query used: `{search_query}`)*")

            with st.spinner("Searching uploaded documents..."):
                results = collection.query(query_texts=[search_query], n_results=4)
                retrieved_context = " ".join([doc for doc in results["documents"][0]])

                rag_prompt = f"Answer the user's question using ONLY the provided context below.\n\nContext:\n{retrieved_context}\n\nQuestion: {prompt}"

                output_text = ""
                for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=rag_prompt,
                            config=types.GenerateContentConfig(http_options=types.HttpOptions(timeout=10000)),
                        )
                        output_text = response.text
                        break
                    except Exception as err:
                        if model_name == PRIMARY_MODEL:
                            pass
                        else:
                            output_text = f"Both endpoints are currently busy. Please retry. ({err})"

                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
                save_history(st.session_state.messages)

# ==========================================
# TAB 2: AGENTIC ELIGIBILITY EVALUATOR
# ==========================================
with tab2:
    st.header("📋 Automated Eligibility Scorecard")
    st.write("Enter your profile details below. The AI Agent will cross-reference your data against all loaded schemes and generate an eligibility verdict.")
    
    col1, col2 = st.columns(2)
    with col1:
        user_income = st.number_input("Annual Family Income (₹)", min_value=0, value=500000, step=50000)
        user_marks = st.number_input("12th/HSC Percentage (%)", min_value=0.0, max_value=100.0, value=65.0)
        user_category = st.selectbox("Caste/Category", ["General", "OBC", "SC", "ST", "VJNT", "SBC", "Minority"])
    with col2:
        user_location = st.selectbox("Where are you studying?", ["Mumbai (MMRDA)", "Pune (PMRDA)", "Nagpur", "Aurangabad", "Other / Rural Region"])
        user_hostel = st.radio("Are you living in a hostel?", ["Yes", "No"])
        
    if st.button("Evaluate My Profile", type="primary"):
        if collection.count() == 0:
            st.error("No schemes loaded! Please upload documents in the sidebar first.")
        else:
            with st.spinner("Agent is analyzing rules and computing eligibility..."):
                # Use min() so n_results never exceeds available chunks
                num_chunks = min(collection.count(), 8)
                broad_results = collection.query(
                    query_texts=["eligibility rules income limit hostel allowance benefits scholarship marks caste category reservation"], 
                    n_results=num_chunks
                )
                broad_context = " ".join([doc for doc in broad_results["documents"][0]])
                
                agent_prompt = f"""
                You are a strict, analytical Government Scheme Evaluator. 
                Evaluate the student's profile against the provided rules and determine which schemes they qualify for.
                
                STUDENT PROFILE:
                - Income: ₹{user_income}
                - 12th Marks: {user_marks}%
                - Caste/Category: {user_category}
                - Location: {user_location}
                - Hostel Resident: {user_hostel}
                
                SCHEME RULES (Context):
                {broad_context}
                
                OUTPUT FORMAT:
                Create a clean Markdown table evaluating each scheme mentioned in the context.
                Columns: Scheme Name | Eligible? (Yes/No) | Reason | Potential Benefit
                Do not invent schemes or rules not found in the context.
                """
                
                # Resilient execution with fallback and extended timeout
                eval_output = ""
                for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
                    try:
                        eval_response = client.models.generate_content(
                            model=model_name,
                            contents=agent_prompt,
                            config=types.GenerateContentConfig(
                                http_options=types.HttpOptions(timeout=45000)  # 45 second safety ceiling
                            ),
                        )
                        eval_output = eval_response.text
                        break
                    except Exception as err:
                        if model_name == PRIMARY_MODEL:
                            st.warning(f"⚠️ Primary model timed out or busy. Rerouting to backup engine...")
                        else:
                            eval_output = f"Evaluation failed across both engines due to network load: {err}"
                
                if eval_output:
                    st.markdown(eval_output)