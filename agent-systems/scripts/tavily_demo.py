# Explore Tavily's capabilities before using it in an agent

import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv

tavily_api_key = os.getenv("TAVILY_API_KEY")

client = TavilyClient(tavily_api_key)

def search(query: str):
    response = client.search(
        query=query,
        max_results=3,
        search_depth="basic"
    )
    return response


def extract(urls: list):
    response = client.extract(
        urls=urls
    )
    return response

def crawl(url: str, instructions: str):
    response = client.crawl(
        url=url,
        instructions=instructions,
        max_depth=4,
        max_breadth=4,
        extract_depth="advanced"
    )
    return response



search_query = "What are the key differences between GPT-4 and Claude 2?"
search_results = search(search_query)
# print(search_results)

print('=== TAVILY SEARCH RESULTS ===')
for r in search_results['results']:
    print(f'Title:   {r["title"]}')
    print(f'URL:     {r["url"]}')
    print(f'Content: {r["content"][:200]}')
    print(f'Score:   {r["score"]:.3f}')   # Relevance score 0-1
    print()


# ── Q&A mode: ask a question, get a direct answer ─────────────
# Tavily searches AND synthesises the answer for you
def qna(query):
    return client.qna_search(
        query=query
    )

query='How many Starlink satellites are in orbit in 2025?'
answer = qna(query)

print('=== DIRECT ANSWER ===')
print(answer)   # A concise, synthesised answer

# ── Extract specific content from a URL ───────────────────────

urls = ["https://docs.tavily.com/",
        "https://docs.tavily.com/sdk/python/quick-start",
        "https://docs.tavily.com/sdk/javascript/quick-start",
        "https://docs.tavily.com/documentation/best-practices/best-practices-search",
        "https://docs.tavily.com/documentation/mcp"
    ]

extracted = extract(urls)
print('=== EXTRACTED CONTENT ===')
for item in extracted['results'][:1]:
    print(item['raw_content'][:500])


crawl_url = "https://langchain-ai.github.io/langgraph/"
instructions = "Find all pages on agents"
crawl_result = crawl(crawl_url, instructions)
# print(crawl_result)
