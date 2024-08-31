import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio  # Import plotly.io for image export
from sklearn.datasets import load_wine
from groq import Groq
import io
import base64

# Initialize Groq client with API key
try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except KeyError:
    st.error("API key for Groq is missing in Streamlit secrets.")
    st.stop()
except Exception as e:
    st.error(f"Error initializing Groq client: {e}")
    st.stop()

# Set page config
st.set_page_config(page_title="DataVue - Advanced EDA", layout="wide")

# Load custom CSS
def load_css():
    st.markdown("""
    <style>
        .stApp {
            background-color: #f0f8ff; /* Light background color for the entire app */
        }
        .main-header {
            color: #1e90ff; 
            font-size: 40px; 
            font-weight: bold; 
            text-align: center; 
            margin-bottom: 30px;
        }
        .sub-header {
            color: #4169e1; 
            font-size: 30px; 
            font-weight: bold; 
            margin-top: 20px; 
            margin-bottom: 10px;
        }
        .section {
            background-color: #e6f2ff; 
            border-radius: 10px; 
            padding: 20px; 
            margin-bottom: 20px;
        }
        .button {
            background-color: #1e90ff; 
            color: white; 
            font-weight: bold; 
            border-radius: 5px; 
            padding: 10px 20px;
        }
        .button:hover {
            background-color: #4169e1;
        }
        .error {
            color: #ff4500; 
            font-weight: bold;
        }
        .success {
            color: #32cd32; 
            font-weight: bold;
        }
        .stSidebar {
            background-color: #e6f2ff; /* Sidebar background color */
        }
    </style>
    """, unsafe_allow_html=True)

# Load CSS
load_css()

# Main title
st.markdown("<h1 class='main-header'>🔍 Advanced Data Explorer with AI-Driven Insights</h1>", unsafe_allow_html=True)

# Function to load example datasets
@st.cache_data
def load_example_dataset(dataset_name):
    if dataset_name == "Iris":
        return sns.load_dataset("iris")
    elif dataset_name == "Titanic":
        return sns.load_dataset("titanic")
    elif dataset_name == "Wine":
        wine = load_wine()
        return pd.DataFrame(wine.data, columns=wine.feature_names)
    return None

# Sidebar for dataset selection and file upload
st.sidebar.markdown("<h2 class='sub-header'>📊 Data Selection</h2>", unsafe_allow_html=True)
dataset_option = st.sidebar.radio("Choose a dataset:", ("Upload your own", "Iris", "Titanic", "Wine"))

if dataset_option == "Upload your own":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
else:
    df = load_example_dataset(dataset_option)
    st.sidebar.success(f"{dataset_option} dataset loaded successfully!")

# Main content
if 'df' in locals():
    st.session_state.df = df
    st.session_state.data_loaded = True
else:
    st.session_state.data_loaded = False

if st.session_state.data_loaded:
    # EDA Section
    st.markdown("<h2 class='sub-header'>🔍 Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)
    
    with st.expander("EDA Options", expanded=True):
        st.markdown("### Select EDA Options")
        col1, col2, col3 = st.columns(3)
        with col1:
            show_head = st.checkbox("First 5 Rows", value=True)
            show_shape = st.checkbox("Show Shape", value=True)
            show_desc_num = st.checkbox("Numerical Description", value=True)
        with col2:
            show_tail = st.checkbox("Last 5 Rows")
            show_columns = st.checkbox("Show Columns")
            show_desc_cat = st.checkbox("Categorical Description")
        with col3:
            show_info = st.checkbox("Show Info")
            show_missing = st.checkbox("Show Missing Values")
            delete_cols = st.checkbox("Delete Columns")

        show_histograms = st.checkbox("Show Histograms")
        show_boxplot = st.checkbox("Show Boxplots")

    if delete_cols:
        columns_to_delete = st.multiselect("Select Columns to Delete", st.session_state.df.columns.tolist())
    else:
        columns_to_delete = []

    if st.button("Run EDA", key="run_eda"):
        df = st.session_state.df.copy()

        # Delete selected columns
        if delete_cols and columns_to_delete:
            df.drop(columns=columns_to_delete, inplace=True)
            st.session_state.df = df
            st.success(f"Deleted columns: {', '.join(columns_to_delete)}")

        # Display selected EDA options
        if show_head:
            st.markdown("<h3 class='sub-header'>First 5 Rows</h3>", unsafe_allow_html=True)
            st.dataframe(df.head())

        if show_tail:
            st.markdown("<h3 class='sub-header'>Last 5 Rows</h3>", unsafe_allow_html=True)
            st.dataframe(df.tail())

        if show_columns:
            st.markdown("<h3 class='sub-header'>Columns</h3>", unsafe_allow_html=True)
            st.write(df.columns.tolist())

        if show_shape:
            st.markdown("<h3 class='sub-header'>Shape of the DataFrame</h3>", unsafe_allow_html=True)
            st.write(df.shape)

        if show_info:
            st.markdown("<h3 class='sub-header'>DataFrame Info</h3>", unsafe_allow_html=True)
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

        if show_desc_num:
            st.markdown("<h3 class='sub-header'>Description (Numerical Features)</h3>", unsafe_allow_html=True)
            st.write(df.describe())

        if show_desc_cat:
            st.markdown("<h3 class='sub-header'>Description (Categorical Features)</h3>", unsafe_allow_html=True)
            st.write(df.describe(include=['object', 'category']))

        if show_missing:
            st.markdown("<h3 class='sub-header'>Missing Values</h3>", unsafe_allow_html=True)
            missing = df.isnull().sum()
            st.write(missing[missing > 0])

        if show_histograms:
            st.markdown("<h3 class='sub-header'>Histograms for Numerical Features</h3>", unsafe_allow_html=True)
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                fig = px.histogram(df, x=col, title=f"Histogram of {col}", color_discrete_sequence=['blue'])
                st.plotly_chart(fig)

        if show_boxplot:
            st.markdown("<h3 class='sub-header'>Boxplots for Numerical Features</h3>", unsafe_allow_html=True)
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                fig = px.box(df, y=col, title=f"Boxplot of {col}", color_discrete_sequence=['blue'])
                st.plotly_chart(fig)

    # AI-Driven Insights Section
    st.markdown("<h2 class='sub-header'>💡 AI-Driven Insights</h2>", unsafe_allow_html=True)

    sample_data = st.session_state.df.head().to_string()
    user_prompt = st.text_input("Ask AI to provide insights or recommendations about your data",
                                "What are the key trends in this dataset?")
    ai_prompt = f"Here's a sample of the dataset:\n\n{sample_data}\n\n{user_prompt}"

    if st.button("Generate AI Insights", key="generate_insights"):
        try:
            with st.spinner("Generating AI insights..."):
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": ai_prompt},
                    ],
                    model="llama-3.1-70b-versatile",
                )
            st.markdown("<h3 class='sub-header'>🔍 AI-Driven Insights</h3>", unsafe_allow_html=True)
            st.write(chat_completion.choices[0].message['content'])
        except Exception as e:
            st.error(f"Error generating insights: {e}")
