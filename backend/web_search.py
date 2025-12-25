from tavily import TavilyClient
import streamlit as st

def web_search(query):
    client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
    return client.search(query, max_results=5)
