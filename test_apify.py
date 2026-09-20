import os
from apify_client import ApifyClient

# 1. Initialize client
APIFY_TOKEN = os.environ.get("APIFY_API_TOKEN", "")
client = ApifyClient(token=APIFY_TOKEN)

# 2. Configure the actor input
run_input = {
    "startUrls": [{"url": "https://en.wikipedia.org/wiki/Maharashtra"}],
    "maxCrawlPages": 1,
    "crawlerType": "cheerio",  # Fast, headless text extraction
}

print("🚀 Launching Apify Actor in the cloud...")
# Run the Actor and wait for it to finish
run = client.actor("apify/website-content-crawler").call(run_input=run_input)

# 3. Fetch scraped results from the dataset (UPDATED SYNTAX)
if run is not None:
    print(f"✅ Actor finished with status: {run.status}")
    dataset_items = client.dataset(run.default_dataset_id).list_items().items

    if dataset_items:
        first_page = dataset_items[0]
        print("\n--- Scraped Metadata ---")
        print(f"URL: {first_page.get('url')}")
        print(f"Title: {first_page.get('metadata', {}).get('title')}")
        print(f"Extracted Text Snippet:\n{first_page.get('text', '')[:300]}...")
else:
    print("❌ Failed to retrieve the run object.")