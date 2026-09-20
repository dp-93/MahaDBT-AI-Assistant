import chromadb
import os
import google.generativeai as genai

# 1. Setup the Gemini "Brain"
st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-3.6-flash')

# 2. The Chunker
def chunk_document(filepath, chunk_size=150, overlap=30):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = " ".join(file.read().split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap 
    return chunks

document_chunks = chunk_document("scheme_data.txt")

# 3. The Retrieval (ChromaDB)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="maharashtra_schemes")
collection.add(
    documents=document_chunks,
    ids=[f"chunk_{i}" for i in range(len(document_chunks))]
)

user_question = "How much cash do I get if I live in a hostel in Pune or Mumbai?"

results = collection.query(
    query_texts=[user_question],
    n_results=2 
)

# 4. The Generation (The Missing "G"!)
# We combine the retrieved chunks into one block of text.
retrieved_context = " ".join([doc for doc in results['documents'][0]])

# We build the "Augmented" prompt.
rag_prompt = f"""
You are a helpful assistant for Maharashtra Education Schemes. 
Answer the user's question using ONLY the provided context below. Be conversational and clear.

Context from Database:
{retrieved_context}

User Question: 
{user_question}
"""

print("🧠 Generating final answer using Gemini...\n")
response = gemini_model.generate_content(rag_prompt)

print("######################")
print("🎯 FINAL AI RESPONSE:")
print(response.text)