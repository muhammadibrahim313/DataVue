# ai_assistant.py

import streamlit as st
from groq import Groq
import os

def ai_assistant():
    st.title("AI Assistant")

    # Initialize Groq client
    client = Groq(api_key="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")

    question = st.text_input("Ask a question:")
    if st.button("Get Answer"):
        if question:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": question}
                ],
                model="llama-3.1-70b-versatile",
            )
            st.write(f"**AI Response:** {chat_completion.choices[0].message.content}")
        else:
            st.write("Please enter a question.")
