import streamlit as st
from pages.landing_page import landing_page
from pages.auto_eda import auto_eda
from pages.complete_analysis import complete_analysis
from pages.ai_assistant import ai_assistant
from pages.study_resources import study_resources
from pages.about_page import about_page

PAGES = {
    "Landing Page": landing_page,
    "Auto EDA": auto_eda,
    "Complete Analysis": complete_analysis,
    "AI Assistant": ai_assistant,
    "Study Resources": study_resources,
    "About Page": about_page
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    with st.spinner(f"Loading {selection} ..."):
        page()

if __name__ == "__main__":
    main()
