def route_query(query, web_enabled):
    realtime_keywords = ["latest", "recent", "current", "today", "news"]

    if any(word in query.lower() for word in realtime_keywords):
        return "web"
    if web_enabled:
        return "hybrid"
    return "doc"
