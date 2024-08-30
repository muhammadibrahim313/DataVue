# auto_eda.py

import streamlit as st
import pandas as pd

def auto_eda():
    st.title("Auto EDA")
    st.write("Upload your CSV or Excel file to perform automated EDA.")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.type == "csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "xlsx":
            df = pd.read_excel(uploaded_file)
        
        st.write(f"**Data Preview:**")
        st.write(df.head())

        st.write("### Summary Statistics")
        st.write(df.describe())

        st.write("### Missing Values")
        st.write(df.isnull().sum())

        st.write("### Correlation Matrix")
        st.write(df.corr())
