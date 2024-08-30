# complete_analysis.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def complete_analysis():
    st.title("Complete Analysis")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.type == "csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "xlsx":
            df = pd.read_excel(uploaded_file)

        st.write(f"**Data Preview:**")
        st.write(df.head())

        st.write("### Distribution of Numerical Features")
        for column in df.select_dtypes(include=['float64', 'int64']).columns:
            st.write(f"**{column} Distribution:**")
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)

        st.write("### Pairplot")
        fig = sns.pairplot(df)
        st.pyplot(fig)

        st.write("### Boxplot for Outliers")
        for column in df.select_dtypes(include=['float64', 'int64']).columns:
            st.write(f"**{column} Boxplot:**")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[column], ax=ax)
            st.pyplot(fig)
