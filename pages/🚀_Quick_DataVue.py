import streamlit as st
import pandas as pd
import seaborn as sns
import ydata_profiling as pp
import sweetviz as sv
import plotly.express as px
import base64
import os
from io import BytesIO
from tempfile import NamedTemporaryFile
from streamlit.components.v1 import html



# Set page configuration
st.set_page_config(page_title="DataVue - Auto EDA", page_icon="📊", layout="wide")

# Load custom CSS
def load_css():
    st.markdown(r"""
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
            background-color: #e6f2ff;
        }
    </style>
    """, unsafe_allow_html=True)

# Load CSS
load_css()

def load_example_data(dataset_name):
    if dataset_name == "Titanic":
        return sns.load_dataset("titanic")
    elif dataset_name == "Iris":
        return sns.load_dataset("iris")

def get_download_link(file_name, link_text):
    with open(file_name, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:file/html;base64,{b64}" download="{file_name}">{link_text}</a>'
        return href

def auto_eda():
    st.sidebar.image("https://via.placeholder.com/150x150.png?text=DataVue", width=150)
    st.sidebar.title("DataVue")
    st.sidebar.subheader("Auto EDA Tool")

    # Main content
    st.markdown("<h1 class='main-header'>🔍 DataVue: Automated EDA 📊</h1>", unsafe_allow_html=True)
    st.write("Welcome to **DataVue's Auto EDA** tool! Explore your data with ease and gain valuable insights.")

    # Data selection
    data_source = st.sidebar.radio("Choose data source:", ("Upload your own", "Use example dataset"))

    if data_source == "Upload your own":
        uploaded_file = st.sidebar.file_uploader("Choose a file 📁", type=["csv", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            st.sidebar.success(f"Successfully loaded: {uploaded_file.name}")
    else:
        example_dataset = st.sidebar.selectbox("Select an example dataset:", 
                                               ("Titanic", "Iris"))
        df = load_example_data(example_dataset)
        st.sidebar.success(f"Loaded example dataset: {example_dataset}")

    if 'df' in locals():
        st.write(f"<div class='section'><strong>Data Preview:</strong></div>", unsafe_allow_html=True)
        st.write(df.head())

        # Basic info
        st.markdown("<div class='section'><h2 class='sub-header'>📋 Basic Information</h2></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        # Data types
        st.markdown("<div class='section'><h2 class='sub-header'>🏷️ Data Types</h2></div>", unsafe_allow_html=True)
        st.write(df.dtypes)

        # Summary statistics
        st.markdown("<div class='section'><h2 class='sub-header'>📊 Summary Statistics</h2></div>", unsafe_allow_html=True)
        st.write(df.describe())

        # Missing values
        st.markdown("<div class='section'><h2 class='sub-header'>🕳️ Missing Values</h2></div>", unsafe_allow_html=True)
        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Values']
        missing_data['Percentage'] = round(missing_data['Missing Values'] / len(df) * 100, 2)
        st.write(missing_data)

        # Correlation matrix
        st.markdown("<div class='section'><h2 class='sub-header'>🔗 Correlation Matrix</h2></div>", unsafe_allow_html=True)
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig)
        else:
            st.write("No numeric columns available for correlation matrix.")

        # Distribution plots
        st.markdown("<div class='section'><h2 class='sub-header'>📉 Distribution Plots</h2></div>", unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.empty:
            selected_col = st.selectbox("Select a column for distribution plot:", numeric_cols)
            fig = px.histogram(df, x=selected_col, marginal="box")
            st.plotly_chart(fig)
        else:
            st.write("No numeric columns available for distribution plots.")

        # Advanced EDA tools
        st.markdown("<div class='section'><h2 class='sub-header'>🚀 Advanced EDA Tools</h2></div>", unsafe_allow_html=True)
        tool = st.radio("Select an advanced EDA tool:", ("Pandas Profiling", "Sweetviz"))

        if tool == "Pandas Profiling":
            if st.button("Generate Pandas Profiling Report", key="pandas_button"):
                with st.spinner("Generating Pandas Profiling Report..."):
                    try:
                        # Create a NamedTemporaryFile for the Pandas Profiling report
                        with NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
                            profile = pp.ProfileReport(df, title="Pandas Profiling Report", explorative=True)
                            profile.to_file(temp_file.name)
                            temp_file.seek(0)
                            report_html = temp_file.read().decode()

                        st.components.v1.html(report_html, height=600, scrolling=True)
                        st.success("Pandas Profiling report generated.")
                        st.markdown(get_download_link(temp_file.name, "📥 Download Pandas Profiling Report"), unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error generating Pandas Profiling report: {e}")

        elif tool == "Sweetviz":
            if st.button("Generate Sweetviz Report", key="sweetviz_button"):
                with st.spinner("Generating Sweetviz Report..."):
                    try:
                        # Create a NamedTemporaryFile for the Sweetviz report
                        with NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
                            sweet_report = sv.analyze(df)
                            sweet_report.show_html(temp_file.name)
                            temp_file.seek(0)
                            report_html = temp_file.read().decode()

                        st.components.v1.html(report_html, height=600, scrolling=True)
                        st.success("Sweetviz report generated.")
                        st.markdown(get_download_link(temp_file.name, "📥 Download Sweetviz Report"), unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error generating Sweetviz report: {e}")

        # Download button for data
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a class="button" href="data:file/csv;base64,{b64}" download="datavue_data.csv">📥 Download Data as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    # Explanations
    with st.expander("ℹ️ About DataVue Auto EDA"):
        st.write("""
        DataVue's Auto EDA tool helps you quickly explore and understand your data:
        
        1. **Data Loading**: Upload your own CSV/Excel file or choose from example datasets (Titanic or Iris).
        2. **Basic Info**: Get an overview of your data's shape and basic statistics.
        3. **Data Types**: Understand the types of data in each column.
        4. **Missing Values**: Identify and quantify missing data.
        5. **Correlation Matrix**: Visualize relationships between numeric variables.
        6. **Distribution Plots**: Explore the distribution of individual variables.
        7. **Advanced EDA**: Use Pandas Profiling or Sweetviz for in-depth analysis.
        8. **Download**: Export your data and generated reports for further analysis.

        Explore these features to gain valuable insights into your dataset!
        """)

if __name__ == "__main__":
    auto_eda()
