from tavily import TavilyClient
import streamlit as st

client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

def tavily_search(query):
    result = client.search(query=query, max_results=3)
    return "\n".join(r["content"] for r in result["results"])
