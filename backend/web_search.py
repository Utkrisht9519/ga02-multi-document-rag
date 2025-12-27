from tavily import TavilyClient
import streamlit as st

client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])


def tavily_search(query: str, k: int = 5):
    results = client.search(query=query, max_results=k)

    web_context = []
    for r in results["results"]:
        web_context.append({
            "content": r["content"],
            "source": r["url"]
        })

    return web_context
