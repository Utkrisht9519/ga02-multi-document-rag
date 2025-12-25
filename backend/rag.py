from groq import Groq
import streamlit as st

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def generate_answer(context, question):
    prompt = f"""
You are a helpful AI assistant.

Summarize and answer using ONLY the context below.

Context:
{context}

Question:
{question}
"""
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content
