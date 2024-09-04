import streamlit as st
from groq import Groq
import os
from datetime import datetime

def load_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff;
    }
    .main-header {
        color: #1e90ff;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    .subheader {
        color: #4169e1;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .ai-response {
        background-color: #e6f2ff;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def ai_assistant():
    load_css()
    
    st.markdown("<h1 class='main-header'>DataVue</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Your AI Assistant for Data Science and EDA</p>", unsafe_allow_html=True)
    
    
    # Initialize Groq client
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        client = Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        return
    
    # Example questions
    example_questions = [
        "How do I perform data cleaning in Python?",
        "What are the best visualization libraries for EDA?",
        "Can you explain the concept of feature engineering?",
        "How do I handle missing data in a dataset?",
        "What statistical tests should I use for hypothesis testing?",
    ]
    
    # Display example questions
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=DataVue", width=150)
        st.markdown("### Example Questions")
        for q in example_questions:
            if st.button(q, key=f"btn_{q}"):
                st.session_state.question = q

    # User input
    question = st.text_input("Ask a data science or EDA question:", key="question", help="Type your question here or select an example from the sidebar")
    
    if st.button("Get Answer", key="get_answer"):
        if question:
            with st.spinner("DataVue is thinking..."):
                try:
                    # Construct a more specific prompt
                    prompt = f"""As an AI assistant specializing in data science and exploratory data analysis, please answer the following question:

                    {question}

                    Please provide a concise explanation, along with:
                    1. A brief code example if applicable
                    2. Suggestions for relevant Python libraries or tools
                    3. A recommended online resource for further learning"""

                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are DataVue, an AI assistant specializing in data science and exploratory data analysis."},
                            {"role": "user", "content": prompt}
                        ],
                        model="llama-3.1-70b-versatile",
                    )
                    
                    response = chat_completion.choices[0].message.content
                    
                    # Display the response in a more structured way
                    st.markdown("<h2 class='subheader'>DataVue's Response:</h2>", unsafe_allow_html=True)
                    st.markdown(f"<div class='ai-response'>{response}</div>", unsafe_allow_html=True)
                    
                    # Add a feedback section
                    st.markdown("<h3 class='subheader'>Was this response helpful?</h3>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍 Yes", key="feedback_yes"):
                            st.success("Thank you for your feedback!")
                    with col2:
                        if st.button("👎 No", key="feedback_no"):
                            st.info("We're sorry the response wasn't helpful. Please try rephrasing your question or check our resources section for more information.")
                    
                    # Log the interaction
                    with open("interaction_log.txt", "a") as log_file:
                        log_file.write(f"{datetime.now()} - Question: {question}\n")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question or select an example question from the sidebar.")

    # Additional resources
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Useful Resources")
        resources = {
            "Pandas Documentation": "https://pandas.pydata.org/docs/",
            "Scikit-learn User Guide": "https://scikit-learn.org/stable/user_guide.html",
            "Seaborn Tutorial": "https://seaborn.pydata.org/tutorial.html",
            "Kaggle Learn": "https://www.kaggle.com/learn"
        }
        for name, url in resources.items():
            st.markdown(f"- [{name}]({url})")


if __name__ == "__main__":
    ai_assistant()
