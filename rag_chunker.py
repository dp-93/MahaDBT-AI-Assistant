def chunk_document(filepath, chunk_size=150, overlap=20):
    """Simulates a basic RAG chunking strategy with overlap to prevent cutting off context."""
    
    # 1. Read the raw document
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # 2. Clean the text (remove extra newlines)
    text = " ".join(text.split())
    
    chunks = []
    start = 0
    text_length = len(text)
    
    # 3. Slice the document into chunks
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move forward, but step back slightly for the 'overlap'
        start = end - overlap 
        
    return chunks

# 4. Execute the pipeline
if __name__ == "__main__":
    print("🚀 Ingesting Government Scheme Data...\n")
    
    document_chunks = chunk_document("scheme_data.txt", chunk_size=120, overlap=30)
    
    for index, chunk in enumerate(document_chunks):
        print(f"--- Chunk {index + 1} ---")
        print(f"{chunk}...\n")