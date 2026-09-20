import os
import chromadb
from apify_client import ApifyClient

# 1. Initialize Apify & Target URLs
APIFY_TOKEN = os.environ.get("APIFY_API_TOKEN", "")
client = ApifyClient(token=APIFY_TOKEN)

# Targeting DHE Pune and structured scheme aggregators
run_input = {
    "startUrls": [
        {"url": "https://dhepune.gov.in/mahadbt-scholarship-2/"},
        {"url": "https://school.careers360.com/articles/mahadbt-scholarship"}
    ],
    "maxCrawlPages": 2,
    "crawlerType": "cheerio", # Fast text extraction
}

print("🕷️ Launching Apify to scrape live scheme portals...")
run = client.actor("apify/website-content-crawler").call(run_input=run_input)

# 2. Connect to Local Vector Database
print("💾 Connecting to local ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db") 
collection = chroma_client.get_or_create_collection(name="maharashtra_schemes")

# 3. Extract, Chunk, and Ingest
if run is not None:
    dataset_items = client.dataset(run.default_dataset_id).list_items().items
    total_chunks = 0
    
    for page in dataset_items:
        url = page.get("url", "unknown_url")
        text = page.get("text", "")
        
        if not text.strip():
            print(f"⚠️ No text found on {url}")
            continue
            
        # Clean and chunk the scraped text
        text = " ".join(text.split())
        chunk_size = 500
        overlap = 80
        
        start = 0
        chunks = []
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
            
        # Create unique IDs for the database
        prefix = url.split("//")[-1].replace("/", "_").replace(".", "_")
        ids = [f"{prefix}_chunk_{i}" for i in range(len(chunks))]
        
        # Upsert directly into ChromaDB
        collection.upsert(documents=chunks, ids=ids)
        total_chunks += len(chunks)
        print(f"✅ Extracted and embedded {len(chunks)} chunks from {url}")
        
    print(f"\n🎉 Pipeline Complete! {total_chunks} total live data chunks are now permanently stored in ChromaDB.")
else:
    print("❌ Apify run failed.")