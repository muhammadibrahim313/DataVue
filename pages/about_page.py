# about_page.py

import streamlit as st
from PIL import Image

def about_page():
    st.title("About DataVue")
    st.write("### About the Team")
    st.write("DataVue was developed by a dedicated team of data scientists and engineers committed to advancing data analysis and visualization.")

    st.write("### Contributors")
    st.write("**Basel Mathar**")
    st.write("Data Scientist and Machine Learning Engineer.")
    
    st.write("### Contact Us")
    st.write("Email: baselmathar@gmail.com")
    st.write("Connect with us on social media:")
    
    st.write("""
        [LinkedIn](https://www.linkedin.com/company/clickml/?viewAsMember=true)
        [Instagram](https://www.instagram.com/baselhusam/)
        [Facebook](https://www.facebook.com/profile.php?id=100088667931989)
    """)
