from tavily import TavilyClient
import streamlit as st


def tavily_search(query: str, max_results: int = 3) -> str:
    """
    Perform a web search using Tavily and return formatted text
    """
    api_key = st.secrets.get("TAVILY_API_KEY")

    if not api_key:
        return ""

    client = TavilyClient(api_key=api_key)

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=False
        )

        results = []
        for item in response.get("results", []):
            title = item.get("title", "")
            content = item.get("content", "")
            url = item.get("url", "")

            results.append(f"{title}\n{content}\nSource: {url}")

        return "\n\n".join(results)

    except Exception as e:
        return ""
