import streamlit as st

# Set page configuration - this must be the first command
st.set_page_config(page_title="DataVue", page_icon=":bar_chart:", layout="wide")

def landing_page():
    st.title("Welcome to DataVue")
    st.write("""
        DataVue is an advanced data analysis tool that integrates AI for enhanced data insights and visualization. 
        Our app aims to simplify data processing, analysis, and visualization, offering features like:
        - Automated Data Integration
        - AI-Driven Insights
        - Interactive Visualization
        - Automated Reporting
    """)
    st.write("### Get Started")
    
    # Create navigation buttons
    if st.button("Go to Auto EDA"):
        st.session_state.page = "Auto EDA"
    if st.button("Go to AI Assistant"):
        st.session_state.page = "AI Assistant"
    if st.button("Go to Study Resources"):
        st.session_state.page = "Study Resources"
    if st.button("Go to About Page"):
        st.session_state.page = "About Page"
