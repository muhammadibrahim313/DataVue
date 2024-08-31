import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import PIL.Image as Image




import streamlit as st
from streamlit.components.v1 import html

from groq import Groq
import io
import base64

# Initialize Groq client with API key
api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

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

# Helper Functions
def half_divider():
    col1, col2, col3 = st.columns([0.2, 1, 0.2])
    with col2:
        st.markdown("<hr style='border-top: 2px solid #000000;'>", unsafe_allow_html=True)

def new_line():
    st.markdown("<br>", unsafe_allow_html=True)

# Congratulation Button with Balloons
def congratulation(key):
    col1, col2, col3 = st.columns([0.7, 0.7, 0.7])
    if col2.button("🎉 Congratulation", key=key):
        st.balloons()
        st.markdown("<h3 style='color:#000000; text-align:center;'>🥳 You Have Successfully Finished This Phase!</h3>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #000000;'>🔍 DataVue</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #000000;'>Your comprehensive guide to mastering Data Science and Machine Learning!</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 1px solid #000000;'>", unsafe_allow_html=True)

# Title Page with Custom Font
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@700&display=swap');
    h1 {
        font-family: 'Roboto', sans-serif;
        color: #000000;
        text-align: center;
    }
    </style>
    <h1>📊 Welcome to DataVue</h1>
""", unsafe_allow_html=True)

new_line()
st.markdown("<h4 style='text-align: center; color: #000000;'>Empowering you to unlock the potential of Data Science and Machine Learning.</h4>", unsafe_allow_html=True)
new_line()
st.markdown("<h5 style='text-align: center; color: #000000;'>Select a topic below to dive in.</h5>", unsafe_allow_html=True)
half_divider()

# Tabs with Trendy Layout and Emojis
tab_titles = [
    '🗺️ Overview', 
    '🧭 Exploratory Data Analysis (EDA)', 
    '‍📀‍‍‍‍ Missing Values', 
    '🔠 Categorical Features', 
    '🧬 Scaling & Transformation', 
    '💡 Feature Engineering', 
    '✂️ Splitting the Data', 
    '🧠 Machine Learning Models'
]

tabs = st.tabs(tab_titles)

# Overview Tab Content
with tabs[0]:
    new_line()
    st.markdown("<h2 style='text-align: center; color: #000000;'>🗺️ Overview</h2>", unsafe_allow_html=True)
    new_line()
    
    st.markdown("""
    <div style='font-size: 18px; line-height: 1.8; color: #000000;'>
        Welcome to DataVue, your modern companion on the journey to mastering Data Science and Machine Learning. Here's an overview of the essential steps in the process:
        <ul style='list-style-type: none; padding-left: 0;'>
            <li>📦 <strong style='color: #000000;'>Data Collection</strong>: Gather data from various sources like CSV files, databases, or APIs. A popular resource is <a href="https://www.kaggle.com/" target="_blank" style="color:#000000;">Kaggle</a>.</li>
            <br>
            <li>🧹 <strong style='color: #000000;'>Data Cleaning</strong>: Clean the data by removing duplicates, handling missing values, and addressing outliers. Clean data is crucial for building effective models.</li>
            <br>
            <li>⚙️ <strong style='color: #000000;'>Data Preprocessing</strong>: Transform data into a suitable format for analysis. This includes processing categorical features, numerical features, scaling, and transformation.</li>
            <br>
            <li>💡 <strong style='color: #000000;'>Feature Engineering</strong>: Manipulate features to improve model performance. This step involves feature extraction, transformation, and selection.</li>
            <br>
            <li>✂️ <strong style='color: #000000;'>Splitting the Data</strong>: Split the data into training, validation, and testing sets. The training set is for model training, the validation set for hyperparameter tuning, and the testing set for evaluation.</li>
            <br>
            <li>🧠 <strong style='color: #000000;'>Building Machine Learning Models</strong>: Build models using algorithms like Linear Regression, Logistic Regression, Decision Trees, Random Forests, SVM, KNN, and Neural Networks.</li>
            <br>
            <li>⚖️ <strong style='color: #000000;'>Evaluating Machine Learning Models</strong>: Evaluate models using metrics like accuracy, precision, recall, F1 score, MSE, RMSE, MAE, and R-squared.</li>
            <br>
            <li>📐 <strong style='color: #000000;'>Tuning Hyperparameters</strong>: Optimize hyperparameters to enhance model performance. Examples include tuning the number of estimators for Random Forest or the number of neighbors for KNN.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    new_line()
# Exploratory Data Analysis (EDA)
with tabs[1]:
    new_line()
    st.markdown("<h2 style='text-align: center;' id='eda'>🧭 Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)
    new_line()
    st.markdown(
        "Exploratory Data Analysis (EDA) is the process of analyzing data sets to summarize their main characteristics, often with visual methods. "
        "EDA helps us understand the data before formal modeling or hypothesis testing. It’s crucial for making informed decisions about the right models and avoiding errors. "
        "It also aids in finding patterns, spotting anomalies, testing hypotheses, and checking assumptions through summary statistics and graphical representations.",
        unsafe_allow_html=True
    )
    new_line()

    st.markdown("<h6>The following are some of the key steps in EDA:</h6>", unsafe_allow_html=True)
    st.markdown("- **Data Collection:** Gather data from various sources like CSV files, databases, APIs, etc. 🌐", unsafe_allow_html=True)
    st.markdown("- **Data Cleaning:** Clean the data by removing duplicates, handling missing values, and dealing with outliers. 🧹", unsafe_allow_html=True)
    st.markdown("- **Data Preprocessing:** Transform the data into a suitable format for analysis, including handling categorical features, numerical features, scaling, and transformation. 🔄", unsafe_allow_html=True)
    st.markdown("- **Data Visualization:** Visualize the data using plots such as bar plots, histograms, scatter plots, etc. 📊", unsafe_allow_html=True)
    st.markdown("- **Data Analysis:** Analyze the data using statistical methods like mean, median, mode, and standard deviation. 📈", unsafe_allow_html=True)
    st.markdown("- **Data Interpretation:** Interpret the data to draw conclusions and make decisions. 🧠", unsafe_allow_html=True)
    new_line()

    st.markdown("<h6>Key Questions Answered in EDA:</h6>", unsafe_allow_html=True)
    st.markdown("- **What is the size of the data?**", unsafe_allow_html=True)
    st.code("""df = pd.read_csv('data.csv') 
df.shape""", language="python")

    st.markdown("- **What are the features in the data?**", unsafe_allow_html=True)
    st.code("""df.columns""", language="python")

    st.markdown("- **What are the data types of the features?**", unsafe_allow_html=True)
    st.code("""df.dtypes""", language="python")

    st.markdown("- **What are the missing values in the data?**", unsafe_allow_html=True)
    st.code("""df.isnull().sum()""", language="python")

    st.markdown("- **What are the outliers in the data?**", unsafe_allow_html=True)
    st.code("""df.describe()""", language="python")

    st.markdown("- **What are the correlations between the features?**", unsafe_allow_html=True)
    st.code("""df.corr()""", language="python")

    st.markdown("- **What are the distributions of the features?**", unsafe_allow_html=True)
    st.code("""df.hist()""", language="python")

    st.divider()

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris

def new_line():
    st.write("\n")

# EDA with selected dataset
# new_line()
# st.subheader("Select a Dataset to Perform EDA on")
# # dataset = st.selectbox("Select a dataset", ["Select", "Iris", "Titanic", "Wine Quality"])
# dataset=st.selectbox("Select a dataset", ["Select", "Iris", "Titanic", "Wine Quality"], key="EDA")

# Perform Missing Values on the Dataset
st.divider()
st.markdown("#### Select Dataset to Perform  EDA ")
dataset = st.selectbox("Select dataset", ["Select", "Iris", "Titanic", "Wine Quality"])
    
new_line()

if dataset == "Iris":
    # Iris Dataset
    st.markdown(
        "The Iris dataset is a classic dataset in machine learning, introduced by Ronald Fisher in 1936. It consists of 150 samples of three Iris species with four features each: sepal length, sepal width, petal length, and petal width. 🌸"
        " It's often used in classification and clustering tasks and is available in the scikit-learn library.",
        unsafe_allow_html=True
    )
    new_line()

    # Perform EDA Process
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target'] = df['target'].apply(lambda x: iris.target_names[x])
    df['target'] = df['target'].astype('category')

    # Read the data
    st.subheader("Read the Data")
    st.write("You can read the data using the following code:")
    st.code("""from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target'] = df['target'].apply(lambda x: iris.target_names[x])
df['target'] = df['target'].astype('category')""", language="python")
    st.write(df)

    # Data Size
    st.subheader("Data Size")
    st.write("The size of the data is:")
    st.code("""df.shape""", language="python")
    st.write(df.shape)
    st.markdown("The data has 150 rows and 5 columns.")
    new_line()

    # Data Types
    st.subheader("Data Types")
    st.write("The data types of the features are:")
    st.code("""df.dtypes""", language="python")
    st.write(df.dtypes)
    st.markdown("The data has 4 numerical features and 1 categorical feature.")
    new_line()

    # Missing Values
    st.subheader("Missing Values")
    st.write("The missing values in the data are:")
    st.code("""df.isnull().sum()""", language="python")
    st.write(df.isnull().sum())
    st.markdown("The data has no missing values.")
    new_line()

    # Description
    st.subheader("Description")
    st.write("The descriptive statistics of the data are:")
    st.code("""df.describe()""", language="python")
    st.write(df.describe())
    st.markdown("The `.describe()` method summarizes the central tendency, dispersion, and shape of a dataset’s distribution, excluding NaN values.")
    new_line()

    # Distribution of Features
    st.subheader("Distribution of Features")
    st.write("The distribution of each feature is shown below:")

    for feature in iris.feature_names:
        st.markdown(f"<h6>{feature.capitalize()} (cm)</h6>", unsafe_allow_html=True)
        st.code(f"""fig = px.histogram(df, x='{feature}', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x=feature, marginal='box')
        st.write(fig)
        new_line()

    # Relationship between Features
    st.subheader("Relationship between Features")
    st.write("The relationship between pairs of features is visualized below:")

    scatter_plots = [
        ('sepal length (cm)', 'sepal width (cm)'),
        ('sepal length (cm)', 'petal length (cm)'),
        ('sepal length (cm)', 'petal width (cm)'),
        ('sepal width (cm)', 'petal length (cm)')
    ]

    for x, y in scatter_plots:
        st.markdown(f"<h6>{x.capitalize()} vs {y.capitalize()}</h6>", unsafe_allow_html=True)
        st.code(f"""fig = px.scatter(df, x='{x}', y='{y}', color='target')
fig.show()""", language="python")
        fig = px.scatter(df, x=x, y=y, color='target')
        st.write(fig)
        new_line()

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    st.write("The correlation matrix is visualized below:")
    st.code("""fig = px.imshow(df[iris.feature_names].corr(), color_continuous_scale='Blues')
fig.show()""", language="python")
    fig = px.imshow(df[iris.feature_names].corr(), color_continuous_scale='Blues')
    st.write(fig)
    new_line()

    # Distribution of the Target
    st.subheader("Distribution of the Target")
    st.write("The distribution of the target is shown below:")
    st.code("""fig = px.histogram(df, x='target')
fig.show()""", language="python")
    fig = px.histogram(df, x='target')
    st.write(fig)
    st.markdown("The target is balanced, with each class having an equal number of samples.")
    new_line()

    # Problem Type
    st.subheader("Problem Type")
    st.write("The problem type is:")
    st.code("""df['target'].value_counts()""", language="python")
    st.write(df['target'].value_counts())
    st.markdown("This is a classification problem because the target variable is categorical.")
    new_line()

    # Conclusion
    st.subheader("Conclusion")
    st.write(
        "From the EDA process, we can conclude the following:"
        "- The data is clean and ready for further analysis."
        "- It has 150 rows and 5 columns."
        "- There are 4 numerical features and 1 categorical feature."
        "- There are no missing values or outliers."
        "- The target variable is balanced."
    )
    new_line()

    st.write("For more information, check out these resources:")
    st.markdown("- [Introduction to EDA](https://towardsdatascience.com/what-is-exploratory-data-analysis-eda-87f48d2a83fe)")
    st.markdown("- [Iris Dataset Overview](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)")
    new_line()

elif dataset == "Titanic":
    # Titanic Dataset
    st.markdown(
        "The Titanic dataset is a multivariate dataset that contains data about the passengers of the Titanic. It consists of 891 samples of passengers with various features such as age, sex, and class. This dataset is frequently used for classification tasks to predict survival rates and is widely utilized in data mining and machine learning examples.",
        unsafe_allow_html=True
    )
    new_line()

    # Perform EDA Process
    try:
        import pandas as pd
        import plotly.express as px
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        # Read the dataset
        titanic = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

        # Read the data
        st.subheader("Read the Data")
        st.write("You can read the data using the following code:")
        st.code("""import pandas as pd
titanic = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')""", language="python")
        st.write(titanic.head())
        new_line()

        # Data Size
        st.subheader("Data Size")
        st.write("The size of the data is:")
        st.code("""titanic.shape""", language="python")
        st.write(f"The data has {titanic.shape[0]} rows and {titanic.shape[1]} columns.")
        new_line()

        # Data Types
        st.subheader("Data Types")
        st.write("The data types of the features are:")
        st.code("""titanic.dtypes""", language="python")
        st.write(titanic.dtypes)
        new_line()

        # Missing Values
        st.subheader("Missing Values")
        st.write("The missing values in the data are:")
        st.code("""titanic.isnull().sum()""", language="python")
        st.write(titanic.isnull().sum())
        st.markdown("The dataset contains missing values in several columns. It is important to handle these before further analysis.")
        new_line()

        # Descriptive Statistics
        st.subheader("Descriptive Statistics")
        st.write("The descriptive statistics of the data are:")
        st.code("""titanic.describe(include='all')""", language="python")
        st.write(titanic.describe(include='all'))
        new_line()

        # Distribution of Features
        st.subheader("Distribution of Features")
        st.write("The distribution of key features is shown below:")

        # Numerical Features Distribution
        numerical_features = ['Age', 'Fare']
        for feature in numerical_features:
            st.markdown(f"<h6>{feature.capitalize()}</h6>", unsafe_allow_html=True)
            st.code(f"""fig = px.histogram(titanic, x='{feature}', marginal='box', title='Distribution of {feature.capitalize()}')
fig.update_layout(bargap=0.1)
fig.show()""", language="python")
            fig = px.histogram(titanic, x=feature, marginal='box', title=f'Distribution of {feature.capitalize()}')
            fig.update_layout(bargap=0.1)
            st.write(fig)
            new_line()

        # Categorical Features Distribution
        categorical_features = ['Sex', 'Embarked', 'Pclass']
        for feature in categorical_features:
            st.markdown(f"<h6>{feature.capitalize()}</h6>", unsafe_allow_html=True)
            st.code(f"""fig = px.histogram(titanic, x='{feature}', title='Distribution of {feature.capitalize()}')
fig.update_layout(bargap=0.1)
fig.show()""", language="python")
            fig = px.histogram(titanic, x=feature, title=f'Distribution of {feature.capitalize()}')
            fig.update_layout(bargap=0.1)
            st.write(fig)
            new_line()

        # Relationship between Features
        st.subheader("Relationship between Features")
        st.write("The relationships between pairs of features are visualized below:")

        scatter_plots = [
            ('Age', 'Fare'),
            ('Age', 'Pclass'),
            ('Fare', 'Pclass')
        ]

        for x, y in scatter_plots:
            st.markdown(f"<h6>{x.capitalize()} vs {y.capitalize()}</h6>", unsafe_allow_html=True)
            st.code(f"""fig = px.scatter(titanic, x='{x}', y='{y}', color='Survived', title='{x.capitalize()} vs {y.capitalize()}')
fig.update_layout(title_font_size=20, title_x=0.5)
fig.show()""", language="python")
            fig = px.scatter(titanic, x=x, y=y, color='Survived', title=f'{x.capitalize()} vs {y.capitalize()}')
            fig.update_layout(title_font_size=20, title_x=0.5)
            st.write(fig)
            new_line()

#         # Correlation Matrix
#         st.subheader("Correlation Matrix")
#         st.write("The correlation matrix of numerical features is visualized below:")
#         st.code("""fig = px.imshow(titanic.corr(), color_continuous_scale='Blues', title='Correlation Matrix')
# fig.update_layout(title_font_size=20, title_x=0.5)
# fig.show()""", language="python")
#         fig = px.imshow(titanic.corr(), color_continuous_scale='Blues', title='Correlation Matrix')
#         fig.update_layout(title_font_size=20, title_x=0.5)
#         st.write(fig)
#         new_line()

        # Target Variable Distribution
        st.subheader("Distribution of the Target")
        st.write("The distribution of the target variable (Survived) is shown below:")
        st.code("""fig = px.histogram(titanic, x='Survived', title='Distribution of Survival')
fig.update_layout(bargap=0.1)
fig.show()""", language="python")
        fig = px.histogram(titanic, x='Survived', title='Distribution of Survival')
        fig.update_layout(bargap=0.1)
        st.write(fig)
        new_line()

        # Additional Analysis: Survival by Pclass and Sex
        st.subheader("Survival Rate by Pclass and Sex")
        st.write("The survival rate by passenger class and sex is shown below:")

        # Survival Rate by Pclass
        st.markdown("<h6>Survival Rate by Passenger Class</h6>", unsafe_allow_html=True)
        st.code("""fig = px.bar(titanic.groupby('Pclass')['Survived'].mean().reset_index(), x='Pclass', y='Survived', title='Survival Rate by Passenger Class')
fig.update_layout(title_font_size=20, title_x=0.5)
fig.show()""", language="python")
        fig = px.bar(titanic.groupby('Pclass')['Survived'].mean().reset_index(), x='Pclass', y='Survived', title='Survival Rate by Passenger Class')
        fig.update_layout(title_font_size=20, title_x=0.5)
        st.write(fig)
        new_line()

        # Survival Rate by Sex
        st.markdown("<h6>Survival Rate by Sex</h6>", unsafe_allow_html=True)
        st.code("""fig = px.bar(titanic.groupby('Sex')['Survived'].mean().reset_index(), x='Sex', y='Survived', title='Survival Rate by Sex')
fig.update_layout(title_font_size=20, title_x=0.5)
fig.show()""", language="python")
        fig = px.bar(titanic.groupby('Sex')['Survived'].mean().reset_index(), x='Sex', y='Survived', title='Survival Rate by Sex')
        fig.update_layout(title_font_size=20, title_x=0.5)
        st.write(fig)
        new_line()

        # Problem Type
        st.subheader("Problem Type")
        st.write("The problem type is:")
        st.code("""titanic['Survived'].value_counts()""", language="python")
        st.write(titanic['Survived'].value_counts())
        st.markdown("This is a classification problem because the target variable 'Survived' is categorical.")
        new_line()

        # Conclusion
        st.subheader("Conclusion")
        st.write(
            "From the EDA process, we can conclude the following:"
            "- The dataset contains both numerical and categorical features."
            "- There are missing values in several columns that need to be addressed."
            "- Various features are analyzed, including distributions, correlations, and relationships."
            "- Survival rates vary by passenger class and sex."
            "- This is a classification problem focusing on predicting survival status."
        )
        new_line()

        st.write("For more information, check out these resources:")
        st.markdown("- [Introduction to EDA](https://towardsdatascience.com/what-is-exploratory-data-analysis-eda-87f48d2a83fe)")
        st.markdown("- [Titanic Dataset Overview](https://www.kaggle.com/c/titanic)")
        st.markdown("- [Data Cleaning and Feature Engineering](https://towardsdatascience.com/data-cleaning-and-feature-engineering-2f7c85cba067)")
        new_line()

    except Exception as e:
        st.error(f"An error occurred: {e}")











elif dataset == "Wine Quality":
    # Wine Quality Dataset
    st.markdown(
        "The Wine Quality dataset contains information about red and white wines with various physicochemical properties and a quality rating. The dataset is used for regression and classification tasks. It is available from the UCI Machine Learning Repository and is useful for analyzing and predicting wine quality.",
        unsafe_allow_html=True
    )
    new_line()

    # Perform EDA Process
    try:
        import pandas as pd
        import plotly.express as px
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the dataset
        red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

        red_wine = pd.read_csv(red_wine_url, delimiter=';')
        white_wine = pd.read_csv(white_wine_url, delimiter=';')

        # Combine the datasets for unified analysis
        red_wine['type'] = 'Red'
        white_wine['type'] = 'White'
        wine = pd.concat([red_wine, white_wine], ignore_index=True)

        # Define numerical features
        numerical_features = wine.select_dtypes(include=[np.number]).columns.tolist()

        # Read the data
        st.subheader("Read the Data")
        st.write("You can read the data using the following code:")
        st.code("""import pandas as pd

# Load the dataset from UCI repository
red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

red_wine = pd.read_csv(red_wine_url, delimiter=';')
white_wine = pd.read_csv(white_wine_url, delimiter=';')

# Combine the datasets
red_wine['type'] = 'Red'
white_wine['type'] = 'White'
wine = pd.concat([red_wine, white_wine], ignore_index=True)""", language="python")
        st.write(wine.head())
        new_line()

        # Data Size
        st.subheader("Data Size")
        st.write("The size of the data is:")
        st.code("""wine.shape""", language="python")
        st.write(f"The data has {wine.shape[0]} rows and {wine.shape[1]} columns.")
        new_line()

        # Data Types
        st.subheader("Data Types")
        st.write("The data types of the features are:")
        st.code("""wine.dtypes""", language="python")
        st.write(wine.dtypes)
        new_line()

        # Missing Values
        st.subheader("Missing Values")
        st.write("The missing values in the data are:")
        st.code("""wine.isnull().sum()""", language="python")
        st.write(wine.isnull().sum())
        st.markdown("The dataset contains no missing values, so it is ready for analysis.")
        new_line()

        # Descriptive Statistics
        st.subheader("Descriptive Statistics")
        st.write("The descriptive statistics of the data are:")
        st.code("""wine.describe()""", language="python")
        st.write(wine.describe())
        new_line()

        # Distribution of Features
        st.subheader("Distribution of Features")
        st.write("The distribution of key features is shown below:")

        # Dropdown for feature selection
        selected_feature = st.selectbox("Select a feature to display:", options=numerical_features)
        
        if selected_feature:
            st.markdown(f"<h6>{selected_feature.replace('_', ' ').capitalize()}</h6>", unsafe_allow_html=True)
            st.code(f"""fig = px.histogram(wine, x='{selected_feature}', color='type', marginal='box', title='Distribution of {selected_feature.replace("_", " ").capitalize()}')
fig.update_layout(bargap=0.1)
fig.show()""", language="python")
            fig = px.histogram(wine, x=selected_feature, color='type', marginal='box', title=f'Distribution of {selected_feature.replace("_", " ").capitalize()}')
            fig.update_layout(bargap=0.1)
            st.write(fig)
            new_line()

        # Categorical Features Distribution
        st.subheader("Distribution of Quality Ratings")
        st.write("The distribution of wine quality ratings is shown below:")
        st.code("""fig = px.histogram(wine, x='quality', color='type', title='Distribution of Wine Quality Ratings')
fig.update_layout(bargap=0.1)
fig.show()""", language="python")
        fig = px.histogram(wine, x='quality', color='type', title='Distribution of Wine Quality Ratings')
        fig.update_layout(bargap=0.1)
        st.write(fig)
        new_line()

        # Relationship between Features
        st.subheader("Relationship between Features")
        st.write("The relationships between pairs of features are visualized below:")

        # Dropdown for scatter plot selection
        x_feature = st.selectbox("Select x-axis feature:", options=numerical_features)
        y_feature = st.selectbox("Select y-axis feature:", options=numerical_features)

        if x_feature and y_feature:
            st.markdown(f"<h6>{x_feature.replace('_', ' ').capitalize()} vs {y_feature.replace('_', ' ').capitalize()}</h6>", unsafe_allow_html=True)
            st.code(f"""fig = px.scatter(wine, x='{x_feature}', y='{y_feature}', color='type', title='{x_feature.replace("_", " ").capitalize()} vs {y_feature.replace("_", " ").capitalize()}')
fig.update_layout(title_font_size=20, title_x=0.5)
fig.show()""", language="python")
            fig = px.scatter(wine, x=x_feature, y=y_feature, color='type', title=f'{x_feature.replace("_", " ").capitalize()} vs {y_feature.replace("_", " ").capitalize()}')
            fig.update_layout(title_font_size=20, title_x=0.5)
            st.write(fig)
            new_line()

        # Correlation Matrix (Numerical Features Only)
        st.subheader("Correlation Matrix")
        st.write("The correlation matrix of numerical features is visualized below:")
        numerical_data = wine[numerical_features]  # Select only numerical features
        st.code("""fig = px.imshow(numerical_data.corr(), color_continuous_scale='Blues', title='Correlation Matrix')
fig.update_layout(title_font_size=20, title_x=0.5)
fig.show()""", language="python")
        fig = px.imshow(numerical_data.corr(), color_continuous_scale='Blues', title='Correlation Matrix')
        fig.update_layout(title_font_size=20, title_x=0.5)
        st.write(fig)
        new_line()

        # Target Variable Analysis
        st.subheader("Quality Ratings by Type")
        st.write("The average quality ratings for red and white wines are shown below:")

        # Quality Ratings by Type
        st.markdown("<h6>Average Quality Ratings by Wine Type</h6>", unsafe_allow_html=True)
        st.code("""fig = px.bar(wine.groupby('type')['quality'].mean().reset_index(), x='type', y='quality', title='Average Quality Ratings by Wine Type')
fig.update_layout(title_font_size=20, title_x=0.5)
fig.show()""", language="python")
        fig = px.bar(wine.groupby('type')['quality'].mean().reset_index(), x='type', y='quality', title='Average Quality Ratings by Wine Type')
        fig.update_layout(title_font_size=20, title_x=0.5)
        st.write(fig)
        new_line()

        # Problem Type
        st.subheader("Problem Type")
        st.write("The problem type is:")
        st.code("""wine['quality'].value_counts()""", language="python")
        st.write(wine['quality'].value_counts())
        st.markdown("This is a regression problem if predicting quality ratings, or a classification problem if categorizing quality ratings into discrete classes.")
        new_line()

        # Conclusion
        st.subheader("Conclusion")
        st.write(
            "From the EDA process, we can conclude the following:"
            "- The dataset contains various physicochemical features and quality ratings for red and white wines."
            "- The data includes both numerical and categorical features."
            "- Various features have been analyzed through their distributions, relationships, and correlations."
            "- Quality ratings vary between red and white wines, with different average ratings."
            "- The problem can be approached as either a regression or classification problem depending on the target variable."
        )
        new_line()

        st.write("For more information, check out these resources:")
        st.markdown("- [Introduction to Exploratory Data Analysis (EDA)](https://towardsdatascience.com/introduction-to-exploratory-data-analysis-eda-8727be46c918)")
        st.markdown("- [Feature Engineering and Model Building](https://towardsdatascience.com/feature-engineering-and-data-preprocessing-for-machine-learning-dfd637d6bce6)")
        new_line()

    except Exception as e:
        st.error(f"An error occurred: {e}")











# Missing Values
with tabs[2]:

    new_line()
    st.markdown("<h2 align='center'>📀 Missing Values</h2>", unsafe_allow_html=True)
    
    # What are Missing Values?
    new_line()
    st.markdown("Missing values are entries that are absent for a variable in the dataset, often represented by `NaN` or `None`. They are common in real-world datasets due to human errors, data collection issues, or processing mistakes. Missing values can disrupt Machine Learning algorithms as they usually require complete datasets. Handling missing values is crucial to ensure accurate and effective analysis and model performance.", unsafe_allow_html=True)
    new_line()

    # Why Handle Missing Values?
    st.markdown("#### ❓ Why Should We Handle Missing Values?")
    st.markdown("Handling missing values is essential because most Machine Learning algorithms cannot process incomplete data. Ignoring or improperly managing these missing values can lead to biased, inaccurate, or invalid results. Proper handling ensures that the data is clean and usable for analysis and modeling.", unsafe_allow_html=True)
    new_line()

    # Methods to Handle Missing Values
    st.markdown("#### 🧐 How to Handle Missing Values?")
    st.markdown("Here are some common methods for dealing with missing values:")
    new_line()

    st.markdown("#### 🌎 General Approaches")
    st.markdown("- **Drop Missing Values:** Removing rows or columns with missing values is simple but may lead to loss of data. This method is generally used when missing values are minimal.")
    st.markdown("  - **Drop Rows:** Suitable when only a few rows are missing values.")
    st.markdown("  - **Drop Columns:** Suitable when a column has a high percentage of missing values, typically more than 50%.")

    st.markdown("##### 🔷 For Numerical Features")
    st.markdown("- **Fill with the Mean:** Use the mean of the feature to fill missing values. This method works well when there are no significant outliers.")
    st.latex(r''' \mu = \frac{1}{n} \sum_{i=1}^{n} x_i ''')
    new_line()

    st.markdown("- **Fill with the Median:** Use the median to handle missing values. This method is robust against outliers.")
    st.latex(r''' \tilde{x} = \begin{cases} x_{\frac{n+1}{2}} & \text{if } n \text{ is odd} \\ \frac{x_{\frac{n}{2}} + x_{\frac{n}{2}+1}}{2} & \text{if } n \text{ is even} \end{cases} ''')
    new_line()

    st.markdown("- **Fill with the Mode:** Use the most frequent value for categorical features.")
    st.latex(r''' mode = \text{the most frequent value} ''')
    new_line()

    st.markdown("##### 🔶 For Categorical Features")
    st.markdown("- **Fill with the Most Frequent Value:** This method replaces missing values with the most common value in the feature.")
    st.latex(r''' mode = \text{the most frequent value} ''')
    new_line()

    # How to Handle Missing Values in Python?
    st.markdown("#### 🐍 How to Handle Missing Values in Python?")
    st.markdown("Let's explore how to implement these methods in Python.")
    new_line()
    
    # Drop the rows that contain missing values
    st.markdown("- **Drop Rows with Missing Values**")
    st.code("""df.dropna(axis=0, inplace=True)""", language="python")
    new_line()

    # Drop the columns that contain missing values
    st.markdown("- **Drop Columns with Missing Values**")
    st.code("""df.dropna(axis=1, inplace=True)""", language="python")
    new_line()
    
    # Fill with mean
    st.markdown("- **Fill with the Mean**")
    st.code("""df[feature] = df[feature].fillna(df[feature].mean())""", language="python")
    new_line()

    # Fill with median
    st.markdown("- **Fill with the Median**")
    st.code("""df[feature] = df[feature].fillna(df[feature].median())""", language="python")
    new_line()

    # Fill with mode
    st.markdown("- **Fill with the Mode**")
    st.code("""df[feature] = df[feature].fillna(df[feature].mode()[0])""", language="python")
    new_line()

    # Fill with the most frequent value
    st.markdown("- **Fill with the Most Frequent Value**")
    st.code("""df[feature] = df[feature].fillna(df[feature].mode()[0])""", language="python")

    # Perform Missing Values on the Dataset
    st.divider()
    st.markdown("#### Select Dataset to Perform Filling Missing Values")
    dataset = st.selectbox("Select Dataset", ["Select", "Iris", "Titanic", "Wine Quality"])

    if dataset == "Select":
        pass
    
    elif dataset == "Iris":
        from sklearn.datasets import load_iris
        
        df  = load_iris()
        df = pd.DataFrame(df.data, columns=df.feature_names)
        df['target'] = load_iris().target
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Missing Values:")
        st.markdown("The Missing Values in the Dataset are:")
        st.code("""df.isnull().sum()""", language="python")
        st.write(df.isnull().sum())
        st.markdown("The Dataset has no missing values. Thus, no handling of missing values is required.")
        new_line()

        congratulation("missing_iris")

    elif dataset == "Titanic":
        import seaborn as sns

        # Load Titanic dataset using Seaborn
        df = sns.load_dataset('titanic')
        st.markdown("#### The Dataset")
        st.write(df.head())

        st.markdown("#### The Missing Values:")
        st.markdown("The Missing Values in the Dataset are:")
        st.code("""df.isnull().sum()""", language="python")
        st.write(df.isnull().sum())
        st.markdown("The Dataset has missing values and needs handling.")
        st.code("""null_val_df = df.isnull().sum()
null_val_df[null_val_df>0]""", language="python")
        null_val_tit = df.isnull().sum()
        st.write(null_val_tit[null_val_tit>0])
        new_line()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h5 align='left'> <b>Age</b> Feature</h5>", unsafe_allow_html=True)
            new_line()
            st.write(f"No. missing values: **{df[['age']].isnull().sum().values[0]}** ")
            st.write(df[['age']].describe().T[['mean','50%']])
            st.write("No Outliers")
            st.markdown("The used method: :green[Mean]")

        with col2:
            st.markdown("<h5 align='left'> <b> Cabin </b> Feature</h5>", unsafe_allow_html=True)
            new_line()
            st.write(f"No. missing values: **687**")
            st.code("df[['cabin']].isnull().sum().values[0] / len(df)")
            st.write(f"The Percentage of missing: **{687/len(df):.2f}%**")
            st.write("The used method: :green[Drop the Column]")

        with col3:
            st.markdown("<h5 align='left'> Embarked Feature</h5>", unsafe_allow_html=True)
            new_line()
            st.write("No. missing values: **2**")
            new_line()
            st.write("Categorical Feature")
            new_line()
            st.write("The used method: :green[Fill with the Most Frequent Value]")

        # Fill the age feature with the mean
        st.divider()
        st.markdown("#### Filling the Missing Values")
        new_line()

        st.markdown("##### Fill the `Age` Feature with the `Mean`")
        st.code("""df['age'] = df['age'].fillna(df['age'].mean())""", language="python")
        new_line()

        # Drop the Cabin feature
        st.markdown("##### Drop the `Cabin` Feature")
        st.code("""df.drop('cabin', axis=1, inplace=True)""", language="python")
        new_line()

        # Fill the Embarked feature with the most frequent value
        st.markdown("##### Fill the `Embarked` Feature with the Most Frequent Value")
        st.code("""df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])""", language="python")
        new_line()

        congratulation("missing_titanic")
        
    elif dataset == "Wine Quality":
        from sklearn.datasets import load_wine

        df = load_wine()
        df = pd.DataFrame(df.data, columns=df.feature_names)
        df['target'] = load_wine().target
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Missing Values:")
        st.markdown("The Missing Values in the Dataset are:")
        st.code("""df.isnull().sum()""", language="python")
        st.write(df.isnull().sum())
        st.markdown("The Dataset has no missing values. Thus, no handling of missing values is required.")
        new_line()

        congratulation("missing_wine")

    new_line()
    st.markdown("<h4 align='center'> 📚 Online Study Resources </h4>", unsafe_allow_html=True)
    st.markdown("""
    - [Khan Academy: Statistics and Probability](https://www.khanacademy.org/math/statistics-probability) - Provides a solid foundation in statistics, including handling missing data.
    - [Coursera: Data Science Specialization](https://www.coursera.org/specializations/jhu-data-science) - Offers courses covering data cleaning and preprocessing techniques, including handling missing values.
    - [Towards Data Science: Handling Missing Values](https://towardsdatascience.com/handling-missing-values-in-machine-learning-8e5d7b3c9d7d) - An article discussing various techniques to handle missing values with practical examples.
    - [Scikit-learn Documentation: Missing Values](https://scikit-learn.org/stable/modules/impute.html) - Official documentation on handling missing values with Scikit-learn’s impute module.
    - [DataCamp: Handling Missing Values](https://www.datacamp.com/community/tutorials/missing-values-python) - A comprehensive guide on different strategies for handling missing values in Python.
    """)
    new_line()
    st.markdown("Congratulations, you have learned about handling missing values. Keep practicing to master these techniques!")














import seaborn as sns

# Categorical Features
with tabs[3]:

    new_line()
    st.markdown("<h2 align='center'> 🔠 Categorical Features </h1>", unsafe_allow_html=True)

    # What are Categorical Features?
    new_line()
    st.markdown("Categorical features are variables that take on a limited and usually fixed number of possible values. They are also known as nominal features and can be divided into two types: **Ordinal Features** and **Nominal Features**.", unsafe_allow_html=True)
    new_line()

    # Ordinal Features
    st.markdown("#### 🔷 Ordinal Features")
    st.markdown("Ordinal features are categorical features with a defined order or ranking among their values. For example, the `Size` feature might have values like `Small`, `Medium`, and `Large`. These values have a natural ordering: `Small` < `Medium` < `Large`. Another example is `Education`, which could have values like `High School`, `Bachelor`, `Master`, and `Ph.D`, with a similar order: `High School` < `Bachelor` < `Master` < `Ph.D`.", unsafe_allow_html=True)
    new_line()

    # Nominal Features
    st.markdown("#### 🔶 Nominal Features")
    st.markdown("Nominal features are categorical features where the values have no intrinsic order. For instance, the `Gender` feature with values like `Male` and `Female` has no inherent ranking or order.", unsafe_allow_html=True)
    new_line()

    # How to Handle Categorical Features?
    st.markdown("#### 🧐 How to Handle Categorical Features?")
    st.markdown("Handling categorical features often involves converting them into numerical values for machine learning models. Here are common methods to handle categorical features:")

    st.markdown("- **One Hot Encoding**")
    st.markdown("- **Ordinal Encoding**")
    st.markdown("- **Label Encoding**")
    st.markdown("- **Count Frequency Encoding**")
    st.markdown("In the following sections, we will explore each method in detail, demonstrating how to implement them in Python.")

    st.divider()

    # One Hot Encoding
    st.subheader("🥇 One Hot Encoding")
    st.markdown("One Hot Encoding converts categorical values into a binary matrix where each category is represented by a unique column. This method is ideal for nominal features where there is no order. Each category is represented as a binary vector of length equal to the number of categories.", unsafe_allow_html=True)
    new_line()

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Before One Hot Encoding**")
        st.dataframe(pd.DataFrame(np.array(['a','b','c','b','a']), columns=['Feature']), width=250, height=250)

    with col2:
        st.write("**After One Hot Encoding**")
        st.dataframe(pd.DataFrame(np.array([[1,0,0],[0,1,0],[0,0,1],[0,1,0],[1,0,0]]), columns=['Feature_a', 'Feature_b', 'Feature_c']), width=250, height=250)

    new_line()
    st.write("In One Hot Encoding, each category value is converted into a binary vector. For example, the value `a` is encoded as `[1,0,0]`, `b` as `[0,1,0]`, and `c` as `[0,0,1]`.")
    st.code("""from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
df_encoded = encoder.fit_transform(df[['Feature']])
df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out())""", language="python")
    new_line()

    # Ordinal Encoding
    st.subheader("♾️ Ordinal Encoding")
    st.markdown("Ordinal Encoding converts categorical values into numerical values based on their order. This method is suitable for ordinal features where the order of the categories is meaningful. For instance, `Small`, `Medium`, and `Large` can be encoded as `1`, `2`, and `3`, respectively.", unsafe_allow_html=True)
    new_line()

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Before Ordinal Encoding**")
        st.dataframe(pd.DataFrame(np.array(['a','b','c','b','a']), columns=['Feature']), width=250, height=250)

    with col2:
        st.write("**After Ordinal Encoding**")
        st.dataframe(pd.DataFrame(np.array([1,2,3,2,1]), columns=['Feature']), width=250, height=250)

    new_line()
    st.write("In Ordinal Encoding, categorical values are replaced with numerical values reflecting their order. For example, `a` becomes `1`, `b` becomes `2`, and `c` becomes `3`.")
    st.code("""from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(categories=[['a', 'b', 'c']])
df['Feature'] = encoder.fit_transform(df[['Feature']])""", language="python")
    new_line()

    # Label Encoding
    st.subheader("🏷️ Label Encoding")
    st.markdown("Label Encoding is similar to Ordinal Encoding but does not necessarily imply an order among categories. Each unique category is assigned a unique integer value. However, Label Encoding may mislead models to infer an ordinal relationship where none exists.", unsafe_allow_html=True)
    new_line()

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Before Label Encoding**")
        st.dataframe(pd.DataFrame(np.array(['a','b','c','b','a']), columns=['Feature']), width=250, height=250)

    with col2:
        st.write("**After Label Encoding**")
        st.dataframe(pd.DataFrame(np.array([1,2,3,2,1]), columns=['Feature']), width=250, height=250)

    new_line()
    st.write("In Label Encoding, each category is assigned a unique integer. For example, `a` becomes `1`, `b` becomes `2`, and `c` becomes `3`. This method is typically used for ordinal features but can also be applied to nominal features in specific scenarios.")
    st.code("""from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Feature'] = encoder.fit_transform(df['Feature'])""", language="python")
    new_line()

    # Count Frequency Encoding
    st.subheader("〰️ Count Frequency Encoding")
    st.markdown("Count Frequency Encoding replaces categories with their frequency of occurrence in the dataset. This method is useful for nominal features where the frequency of categories can provide meaningful insights.", unsafe_allow_html=True)
    new_line()

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Before Count Frequency Encoding**")
        st.dataframe(pd.DataFrame(np.array(['a','b','c','b','a']), columns=['Feature']), width=250, height=250)

    with col2:
        st.write("**After Count Frequency Encoding**")
        st.dataframe(pd.DataFrame(np.array([2/5, 2/5, 1/5, 2/5, 2/5]), columns=['Feature']), width=250, height=250)

    new_line()
    st.write("In Count Frequency Encoding, each category is replaced by its frequency in the dataset. For instance, `a` occurs 2 times out of 5, so it is encoded as `2/5`.")
    st.code("""df['Feature'] = df['Feature'].map(df['Feature'].value_counts(normalize=True))""", language="python")
    new_line()

    # Perform Categorical Features on the Dataset
    st.divider()
    st.markdown("#### Select Dataset to Perform Categorical Features on")
    dataset = st.selectbox("Select Dataset", ["Select", "Iris", "Titanic" ], key="categorical_data")

    if dataset == "Iris":
        from sklearn.datasets import load_iris

        df = load_iris()
        df = pd.DataFrame(df.data, columns=df.feature_names)
        df['target'] = load_iris().target
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Categorical Features:")
        st.markdown("The Categorical Features in the Dataset are:")
        st.code("""df.select_dtypes(include='object').columns""", language="python")
        st.write(df.select_dtypes(include='object').columns)
        st.markdown("The Dataset has no categorical features. Thus, no handling of categorical features is required.")
        new_line()

        congratulation("categorical_iris")

    if dataset == 'Titanic':
        df = sns.load_dataset('titanic')
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Categorical Features:")
        st.markdown("The Categorical Features in the Dataset are:")
        st.code("""df.select_dtypes(include='object').columns""", language="python")
        col1, col2, col3 = st.columns(3)
        col1.markdown("The Categorical Features")
        col1.write(df.select_dtypes(include='object').columns)
        col2.markdown("The No. of Unique Values")
        col2.write(df.select_dtypes(include='object').nunique())
        col3.markdown("The No. of Missing Values")
        col3.write(df.select_dtypes(include='object').isnull().sum())
        st.markdown("The Dataset has several categorical features that require handling, including `sex`, `embarked`, and `cabin`.")
        new_line()

        congratulation("categorical_titanic")

    # if dataset == 'Wine Quality':
    #     from sklearn.datasets import load_wine
    #     df = pd.DataFrame(df.data, columns=df.feature_names)
    #     st.markdown("#### The Dataset")
    #     st.write(df)
        
        
    

    #     st.markdown("#### The Categorical Features:")
    #     st.markdown("The Categorical Features in the Dataset are:")
    #     st.code("""df.select_dtypes(include='object').columns""", language="python")
    #     col1, col2, col3 = st.columns(3)
    #     col1.markdown("The Categorical Features")
    #     col1.write(df.select_dtypes(include='object').columns)
    #     col2.markdown("The No. of Unique Values")
    #     col2.write(df.select_dtypes(include='object').nunique())
    #     col3.markdown("The No. of Missing Values")
    #     col3.write(df.select_dtypes(include='object').isnull().sum())
    #     st.markdown("The Dataset has no categorical features. Thus, no handling of categorical features is required.")
    #     new_line()

    #     congratulation("categorical_wine_quality")




    st.markdown("<h2 align='center'> 🤔 Want to Learn More?</h2>", unsafe_allow_html=True)

    st.markdown("- **Scikit-Learn Documentation on Categorical Features Handling**: [Scikit-Learn](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)")
    st.markdown("- **Pandas Documentation on Categorical Data**: [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html)")
    st.markdown("- **Understanding Encoding Techniques**: [Towards Data Science](https://towardsdatascience.com/encoding-categorical-features-for-machine-learning-4f6f31d84cf5)")

    st.write("Understanding how to properly handle categorical features is crucial for building effective machine learning models. By mastering these encoding techniques, you will be able to preprocess your data effectively and improve your model's performance.")
    
    
    




from sklearn.preprocessing import *
# Scaling & Transformation
with tabs[4]:

    new_line()
    st.markdown("<h2 align='center'> 🧬 Scaling & Transformation </h2>", unsafe_allow_html=True)

    # What is Scaling & Transformation?
    new_line()
    st.markdown(":green[**Data Scaling**] is the process of adjusting the range of feature values to a common scale. This is crucial because features with larger ranges can disproportionately influence the model, introducing **bias**. Scaling ensures that all features contribute equally to the model's performance.")
    st.markdown(":green[**Data Transformation**] involves converting data to a specific distribution. It is especially important for handling features with skewed distributions or outliers. Transformations help normalize the data, reducing **bias** and improving the model's accuracy.")

    new_line()

    # Why we should scale the data?
    st.markdown("##### 📏 Why Should We Scale the Data?")
    st.markdown("Scaling is essential for algorithms sensitive to feature magnitudes. For instance, the K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) algorithms rely on distance metrics, which are affected by the scale of features. Scaling ensures that each feature contributes equally to the distance calculations.")
    new_line()

    # Why we should transform the data?
    st.markdown("##### ➰ Why Should We Transform the Data?")
    st.markdown("Transformation is crucial for algorithms sensitive to data distribution. Linear Regression and Logistic Regression models assume normally distributed features for optimal performance. Transforming data can stabilize variance and make features more normally distributed, enhancing the model's effectiveness.")

    st.divider()

    # How to scale data
    st.subheader("🧮 Scaling Methods")
    st.markdown("Common scaling methods include:")

    st.markdown("1. Min-Max Scaling")
    st.markdown("2. Standard Scaling")
    st.markdown("3. Robust Scaling")
    st.markdown("4. Max Absolute Scaling")

    st.markdown("We will explore each method, how to implement it in Python, and its use cases.")
    new_line()

    # Min-Max Scaling
    st.markdown("##### Min-Max Scaling")
    st.markdown("Min-Max Scaling, also known as normalization, adjusts feature values to a specified range, usually [0, 1]. It is effective for features without outliers and those that are normally distributed. Min-Max Scaling is computed as follows:")
    
    col1, col2 = st.columns(2)
    col1.latex(r''' x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}} ''')
    col2.latex(r''' Z \in [0, 1] ''')
    
    new_line()

    st.code("""from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['feature'] = scaler.fit_transform(df[['feature']])""", language="python")
    new_line()

    # Standard Scaling
    st.markdown("##### Standard Scaling")
    st.markdown("Standard Scaling, or standardization, scales features to have zero mean and unit variance. This method is ideal for features with a normal distribution or when dealing with outliers. It standardizes the feature as follows:")
    
    col1, col2 = st.columns(2)
    col1.latex(r''' x_{scaled} = \frac{x - \mu}{\sigma} ''')
    col2.latex(r''' Z \in [-1, 1] ''')
    
    new_line()

    st.code("""from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['feature'] = scaler.fit_transform(df[['feature']])""", language="python")
    new_line()

    # Robust Scaling
    st.markdown("##### Robust Scaling")
    st.markdown("Robust Scaling is designed to handle outliers by using the median and interquartile range (IQR). It is useful when the data contains significant outliers. The scaling is computed as follows:")
    
    col1, col2 = st.columns(2)
    col1.latex(r''' x_{scaled} = \frac{x - \text{median}}{\text{IQR}} ''')
    col2.latex(r''' Z \in [-1, 1] ''')
    
    new_line()

    st.code("""from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df['feature'] = scaler.fit_transform(df[['feature']])""", language="python")
    new_line()

    # Max Absolute Scaling
    st.markdown("##### Max Absolute Scaling")
    st.markdown("Max Absolute Scaling scales features by their maximum absolute value. It is suitable for data with a range of positive and negative values. The scaling is computed as follows:")
    
    col1, col2 = st.columns(2)
    col1.latex(r''' x_{scaled} = \frac{x}{x_{max}} ''')
    col2.latex(r''' Z \in [-1, 1] ''')
    
    new_line()

    st.code("""from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df['feature'] = scaler.fit_transform(df[['feature']])""", language="python")
    new_line()
    st.divider()
    new_line()

    # How to transform data
    st.subheader("🧬 Transformation Methods")
    st.markdown("Common transformation methods include:")

    st.markdown("1. Log Transformation")
    st.markdown("2. Square Root Transformation")
    st.markdown("3. Cube Root Transformation")
    st.markdown("4. Box-Cox Transformation")

    st.markdown("We'll explore each method, how to implement it in Python, and its applications.")
    new_line()

    # Log Transformation
    st.markdown("##### Log Transformation")
    st.markdown("Log Transformation is used to stabilize variance and handle right-skewed distributions. It is effective for data with a large spread or outliers. The transformation is calculated as follows:")
    
    col1, col2 = st.columns(2)
    col1.latex(r''' x_{transformed} = \log(x) ''')
    col2.latex(r''' Z \in [-1, 1] ''')
    
    new_line()

    st.code("""import numpy as np
df['feature'] = np.log(df['feature'])""", language="python")
    new_line()

    # Square Root Transformation
    st.markdown("##### Square Root Transformation")
    st.markdown("Square Root Transformation reduces right-skewed distributions by taking the square root of feature values. It is useful for data with moderate skewness and is calculated as follows:")
    
    col1, col2 = st.columns(2)
    col1.latex(r''' x_{transformed} = \sqrt{x} ''')
    col2.latex(r''' Z \in [-1, 1] ''')
    
    new_line()

    st.code("""import numpy as np
df['feature'] = np.sqrt(df['feature'])""", language="python")
    new_line()

    # Cube Root Transformation
    st.markdown("##### Cube Root Transformation")
    st.markdown("Cube Root Transformation is used to handle skewed data by taking the cube root of feature values. It is effective for data with severe skewness and is computed as follows:")
    
    col1, col2 = st.columns(2)
    col1.latex(r''' x_{transformed} = \sqrt[3]{x} ''')
    col2.latex(r''' Z \in [-1, 1] ''')
    
    new_line()

    st.code("""import numpy as np
df['feature'] = np.cbrt(df['feature'])""", language="python")
    new_line()

    # Box-Cox Transformation
    st.markdown("##### Box-Cox Transformation")
    st.markdown("Box-Cox Transformation stabilizes variance and makes the data more normally distributed. It is used for data with positive values and is computed as follows:")
    
    col1, col2 = st.columns(2)
    col1.latex(r''' x_{transformed} = \frac{x^{\lambda} - 1}{\lambda} ''')
    col2.latex(r''' Z \in [-1, 1] ''')
    
    new_line()

    st.code("""from scipy.stats import boxcox
df['feature'] = boxcox(df['feature'] + 1)[0]""", language="python")  # Adding 1 to avoid log(0) issue
    new_line()
    new_line()

    # Perform Scaling & Transformation on the Dataset
    st.divider()
    st.markdown("#### Select Dataset to Perform Scaling & Transformation on")
    dataset = st.selectbox("Select Dataset", ["Select", "Iris", "Titanic", "Wine Quality"], key="scaling_transformation_data")

    if dataset == "Iris":
        from sklearn.datasets import load_iris
        import pandas as pd
        
        df = load_iris()
        df = pd.DataFrame(df.data, columns=df.feature_names)
        df['target'] = load_iris().target
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Numerical Features:")
        st.markdown("The numerical features in the dataset are scaled and transformed:")
        numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
        st.write(numerical_features.tolist())

        # Scale the data
        st.markdown("### Scaling Example:")
        scaler_type = st.selectbox("Select Scaling Method", ["Select", "Min-Max", "Standard", "Robust", "Max Absolute"], key="scaling_method")
        
        if scaler_type != "Select":
            scaler_mapping = {
                "Min-Max": MinMaxScaler(),
                "Standard": StandardScaler(),
                "Robust": RobustScaler(),
                "Max Absolute": MaxAbsScaler()
            }
            scaler = scaler_mapping[scaler_type]
            df[numerical_features] = scaler.fit_transform(df[numerical_features])
            st.markdown(f"Data has been scaled using {scaler_type} Scaling")
            st.write(df)

        # Transform the data
        st.markdown("### Transformation Example:")
        transform_type = st.selectbox("Select Transformation Method", ["Select", "Log", "Square Root", "Cube Root", "Box-Cox"], key="transformation_method")
        
        if transform_type != "Select":
            if transform_type == "Box-Cox":
                for feature in numerical_features:
                    df[feature] = boxcox(df[feature] + 1)[0]  # Adding 1 to avoid log(0) issue
            else:
                transform_mapping = {
                    "Log": np.log,
                    "Square Root": np.sqrt,
                    "Cube Root": np.cbrt
                }
                transformation = transform_mapping[transform_type]
                for feature in numerical_features:
                    df[feature] = transformation(df[feature])
            st.markdown(f"Data has been transformed using {transform_type} Transformation")
            st.write(df)

    elif dataset == "Titanic":
        import pandas as pd
        import seaborn as sns

        # Use an online source for Titanic dataset
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        try:
            df = pd.read_csv(url)
            st.markdown("#### The Dataset")
            st.write(df)
        except Exception as e:
            st.error(f"Error loading Titanic dataset: {e}")

    # Visualization section
    st.subheader("📊 Visualization of Scaled & Transformed Data")

    # Choose which plot to display
    plot_type = st.selectbox("Select Plot Type", ["None", "Histogram", "Box Plot"], key="plot_type")

    if plot_type != "None":
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if dataset == "Iris":
            fig, ax = plt.subplots()
            if plot_type == "Histogram":
                df[numerical_features].hist(ax=ax, bins=20)
            elif plot_type == "Box Plot":
                df[numerical_features].plot(kind='box', ax=ax)
            st.pyplot(fig)

        elif dataset == "Titanic":
            if plot_type == "Histogram":
                fig, ax = plt.subplots()
                df[['Age', 'Fare']].hist(ax=ax, bins=20)
                st.pyplot(fig)
            elif plot_type == "Box Plot":
                fig, ax = plt.subplots()
                sns.boxplot(x='Pclass', y='Fare', data=df, ax=ax)
                st.pyplot(fig)








# Feature Engineering
with tabs[5]:

    new_line()
    st.markdown("<h2 align='center'> 💡 Feature Engineering </h1>", unsafe_allow_html=True)

    # What is Feature Engineering?
    new_line()
    st.markdown("Feature Engineering is the process of performing operations on features themselves. Features can contain valuable information that, if extracted and added to the data, can enhance the model's performance. Feature Engineering is crucial as it can significantly boost the accuracy of the model. It can be categorized into three types: **📈 Feature Extraction**, **🔄 Feature Transformation**, and **🎯 Feature Selection**.", unsafe_allow_html=True)
    new_line()

    # Feature Extraction
    st.markdown("#### 📈 Feature Extraction")
    st.markdown("Feature Extraction involves extracting useful information from the features themselves. This process can increase model accuracy and can be divided into two types: **📊 Numerical Feature Extraction** and **🔠 Categorical Feature Extraction**.", unsafe_allow_html=True)
    new_line()

    # Numerical Feature Extraction
    st.markdown("##### 📊 Numerical Feature Extraction")
    st.markdown("Numerical Feature Extraction involves extracting information from numerical features. It can be categorized into **📏 Scaling** and **🧬 Transformation**.", unsafe_allow_html=True)
    
    # Example
    st.markdown("###### Examples")
    st.markdown("- Extracting `Age` into `Age` and `Age Group`, where `Age Group` is `Age` divided by 10.")
    st.markdown("- Combining `walking distance`, `swimming distance`, and `driving distance` into a single feature `total distance` by summing these distances.")

    # Categorical Feature Extraction
    st.markdown("##### 🔠 Categorical Feature Extraction")
    st.markdown("Categorical Feature Extraction involves extracting information from categorical features and can be divided into **🔷 Ordinal Feature Extraction** and **🔶 Nominal Feature Extraction**.", unsafe_allow_html=True)
    
    # Example
    st.markdown("###### Examples")
    st.markdown("- Extracting `Title` from the `Name` feature, resulting in values like `Mr`, `Mrs`, `Miss`, `Master`, and `Other`.")
    st.markdown("- Combining `SibSp` and `Parch` into `Family Size`.")

    new_line()
    st.divider()

    # Feature Transformation  
    st.markdown("#### 🔄 Feature Transformation")
    st.markdown("Feature Transformation involves transforming feature values using mathematical equations to follow a specific logic.", unsafe_allow_html=True)
    
    # Example
    st.markdown("###### Examples")
    st.markdown("- Transforming `song_duration_ms` to `song_duration_min` by dividing by 3600.")
    st.markdown(":green[Note:] Feature transformations can scale the data to a smaller range, as seen in the transformation of `song_duration_ms` to `song_duration_min`.", unsafe_allow_html=True)
    st.divider()

    # Feature Selection
    st.markdown("#### 🎯 Feature Selection")
    st.markdown("Feature Selection involves selecting the most important features for the model. This is crucial if you have many features, as not all are relevant. In the next section, we'll discuss methods to select the most important features from a model.", unsafe_allow_html=True)
    
    # Example
    st.markdown("###### Examples")
    st.markdown("- **Orphan columns**: Features with very high unique values like `id` that are not useful for the model.")
    st.markdown("- Categorical features with too many unique values relative to the number of rows.")

    st.markdown("##### 🤫 Secret way for applying feature selection")
    st.markdown("You can build a model and select the most important features based on the model's importance scores. Models like `Decision Tree`, `Random Forest`, `XGBoost`, `LightGBM`, and `CatBoost` have built-in feature importance capabilities.", unsafe_allow_html=True)
    st.markdown("###### Example")
    st.code("""from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
model.feature_importances_""", language="python")
    st.markdown("The `model.feature_importances_` attribute provides the importance of each feature. Higher values indicate more important features.")

    st.divider()

    # Apply Feature Engineering on the Dataset
    st.markdown("#### Select Dataset to Apply Feature Engineering on")
    dataset = st.selectbox("Select Dataset", ["Select", "Iris", "Titanic", "Wine Quality"], key = "feature_engineering_data")

    if dataset == "Iris":
        import seaborn as sns
        df = sns.load_dataset('iris')
        st.markdown("#### The Dataset")
        st.write(df)
        st.write("Feature Engineering is not applicable here as the features are crucial for the model. No features will be added or modified.")
        new_line()
        congratulation("feature_engineering_iris")

    if dataset == 'Titanic':
        import pandas as pd
        import seaborn as sns
        
        df = sns.load_dataset('titanic')
        st.markdown("#### The Dataset")
        st.write(df)
        
        st.markdown("#### Feature Extraction")
        st.markdown("- Extract `Title` from `Name`.")
        st.code("""df['Title'] = df['name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]""", language="python")
        new_line()
        st.markdown("- Combine `SibSp` and `Parch` into `Family Size`.")
        st.code("""df['Family Size'] = df['sibsp'] + df['parch']""", language="python")

        st.markdown("#### Feature Transformation")
        st.markdown("- Transform `Age` into age groups by dividing by 10.")
        st.code("""df['Age'] = df['age'] // 10""", language="python")
        new_line()

        st.markdown("#### Feature Selection")
        st.markdown("- Drop `PassengerId` and `Name` as they are **Orphan columns**.")
        st.code("""df.drop(['passenger_id', 'name'], axis=1, inplace=True)""", language="python")
        new_line()
        congratulation("feature_engineering_titanic")

    if dataset == "Wine Quality":
        from sklearn.datasets import load_wine
        
        df = load_wine(as_frame=True).frame
        st.markdown("#### The Dataset")
        st.write(df)
        
        st.markdown("#### Feature Extraction")
        st.markdown("- Calculate the ratio of alcohol concentration to volatile acidity.")
        st.code("""df['alcohol concentration per acidity level'] = df['alcohol'] / df['volatile acidity']""", language="python")
        st.markdown("- Compute the ratio between total and free sulfur dioxide.")
        st.code("""df['total sulfur dioxide to free sulfur dioxide ratio'] = df['total sulfur dioxide'] / df['free sulfur dioxide']""", language="python")
        new_line()

        st.markdown("#### Feature Selection")
        st.markdown("Select the most relevant features for the model.")
        st.code("""df = df[['alcohol', 'flavanoids', 'color_intensity', 'total_phenols', 'od280/od315_of_diluted_wines', 'proline', 'hue', 'malic_acid', 'target']]""", language="python")
        new_line()
        congratulation("feature_engineering_wine")







# Splitting Data
with tabs[6]:
    new_line()
    st.markdown("<h2 align='center'> ✂️ Splitting The Data </h2>", unsafe_allow_html=True)
    new_line()

    # What is Splitting The Data?
    st.markdown("Splitting the data is the process of dividing the data into three parts: **Training Data**, **Validation Data**, and **Testing Data**. This step is crucial in the machine learning process because you want to evaluate the model on unseen data. Here’s how you typically split the data:")
    st.markdown("1. **Training Data:** Used to train the model. This dataset should have the highest number of rows, typically 60% to 80% of the total data.")
    st.markdown("2. **Validation Data:** Used for hyperparameter tuning. This dataset should have the lowest number of rows, usually 10% to 20% of the total data.")
    st.markdown("3. **Testing Data:** Used for final model evaluation. This dataset should have the second highest number of rows, generally 20% to 30% of the total data.")

    # Train and Test Split
    st.markdown("#### Train and Test Split")
    st.markdown("If you only need to split your data into training and test sets, you can use the following code:")
    st.table(pd.DataFrame([["80%", "20%"]], columns=["Train", "Test"], index=["Split Size"]))
    st.code("""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)""", language="python")
    new_line()

    # Train, Validation, and Test Split
    st.markdown("#### Train, Validation, and Test Split")
    st.markdown("If you need to split your data into training, validation, and test sets, use the following code:")
    st.table(pd.DataFrame([["70%", "15%", "15%"]], columns=["Train", "Validation", "Test"], index=["Split Size"]))
    st.code("""from sklearn.model_selection import train_test_split
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=0.5)""", language="python")
    new_line()

    st.markdown("**:green[1. NOTE:]** Always split the data into features (`X`) and target (`y`).")
    st.markdown("**:green[2. NOTE:]** For very large datasets, testing and validation sets do not need to exceed 5% of the total data. For example, with 100,000 rows, limit the testing and validation sets to a maximum of 5,000 rows each, as large evaluation sets are not necessary for model assessment.")

    # Apply Splitting The Data on the Dataset
    st.divider()
    st.markdown("#### Select Dataset to Apply Splitting The Data")
    dataset = st.selectbox("Select Dataset", ["Select", "Iris", "Titanic", "Wine Quality"], key="splitting_data")

    if dataset == "Iris":
        from sklearn.datasets import load_iris

        df = load_iris()
        df = pd.DataFrame(df.data, columns=df.feature_names)
        df['target'] = load_iris().target
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### Dataset Shape")
        st.code("""df.shape""", language="python")
        st.write(df.shape)

        st.markdown("#### Splitting The Data")
        st.markdown("We will split the data into training and testing sets, with the training set containing 80% and the testing set containing 20% of the rows.")
        st.code("""from sklearn.model_selection import train_test_split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)""", language="python")
        new_line()

        congratulation("splitting_iris")

    if dataset == 'Titanic':
        import seaborn as sns
        
        df = sns.load_dataset("titanic")
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### Dataset Shape")
        st.code("""df.shape""", language="python")
        st.write(df.shape)

        st.markdown("#### Splitting The Data")
        st.markdown("We will split the data into training, validation, and testing sets, with 70% for training, 15% for validation, and 15% for testing.")
        st.code("""from sklearn.model_selection import train_test_split
X = df.drop('survived', axis=1)
y = df['survived']
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=0.5)""", language="python")
        new_line()

        congratulation("splitting_titanic")

    if dataset == "Wine Quality":
        from sklearn.datasets import load_wine

        df = load_wine()
        df = pd.DataFrame(df.data, columns=df.feature_names)
        df['target'] = load_wine().target
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### Dataset Shape")
        st.code("""df.shape""", language="python")
        st.write(df.shape)

        st.markdown("#### Splitting The Data")
        st.markdown("We will split the data into training, validation, and testing sets, with 70% for training, 15% for validation, and 15% for testing.")
        st.code("""from sklearn.model_selection import train_test_split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=0.5)""", language="python")
        new_line()

        congratulation("splitting_wine")





















# Building Machine Learning Models
with tabs[7]:
    
        new_line()
        st.markdown("<h2 align='center'> 🧠 Building Machine Learning Models </h1>", unsafe_allow_html=True)
        new_line()

        

        # Introduction to Building Machine Learning Models
        st.markdown(""" Machine learning models play a crucial role in predicting outcomes and making informed decisions based on data. Building an effective machine learning model requires a systematic approach that encompasses various steps, including data preparation, model selection, training, evaluation, and deployment.

Throughout this section, we will cover **Regression Models** for predicting **Numerical Targets** (continuous values) and **Classification Models** for **Categorizing Data** (discrete data) into classes. You will learn about linear regression, decision trees, random forests, support vector machines, neural networks, and more. Additionally, we will delve into evaluating model performance, selecting optimal models, tuning hyperparameters, and deploying models in real-world scenarios.

""", unsafe_allow_html=True)

        # Tabs
        tab_titles_ml = ["  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 💫 Regression Models  󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 ", "  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 🪀 Classification Models  󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 "]
        ml_tabs = st.tabs(tab_titles_ml)

        # Regression Models
        with ml_tabs[0]:

                st.write('\n\n\n\n\n')

                st.markdown("""
                ### 💫 Overview of Regression Models

        Regression models are a fundamental class of machine learning models used for predicting numerical values. In this section, we will explore two main types of regression models: linear regression and non-linear regression.

        #### ✏️ Linear Regression

        Linear regression is a widely used regression technique that models the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the independent variables and the dependent variable. The goal of linear regression is to find the best-fit line that minimizes the difference between the predicted and actual values.

        There are two types of linear regression:

        - **Simple Linear Regression**: In simple linear regression, we have one independent variable (feature) and one dependent variable (target). The model fits a straight line to the data to predict the target variable.

        - **Multiple Linear Regression**: Multiple linear regression extends simple linear regression to include multiple independent variables. It considers the linear relationship between multiple features and the target variable.

        #### ⛓ Non-Linear Regression

        Non-linear regression models capture more complex relationships between the independent variables and the dependent variable. Unlike linear regression, non-linear regression does not assume a linear relationship. It allows for curves, exponential functions, polynomial functions, and other non-linear patterns.

        Non-linear regression can be useful when the relationship between variables is not linear and requires a more flexible model to capture the underlying patterns in the data. Various algorithms, such as decision trees, support vector regression, and neural networks, can be used to perform non-linear regression.

        
        <br> 
                        
        ---
                        
        <br>

        """, unsafe_allow_html=True)
                
                # Regression Algorithms
                st.markdown("### 🧪 Regression Algorithms")
                st.markdown("There are numerous regression algorithms available, each with its strengths and limitations. Some popular algorithms include: **Linear Regression**, **Decision Trees**, **Random Forest**, **Support Vector Regression (SVR)**, **K Nearest Neighbors (KNN)**, and **XGBoost**. We will dive into each of these algorithms in detail in the following section.")
                st.write("\n")

                # expander 
                with st.expander("🧪 Regression Algorithms"):
                        st.write("\n")

                        # Regression Algorithms Tabs
                        tabs_reg_aglo = [" 📏 Linear Regression", " 🍁 Decision Tree", " 🌳 Random Forest", " ⛑️ Support Vector Regression", " 🏘️ K Nearest Neighbors", " 💥 XGBoost"]
                        reg_algo_tabs = st.tabs(tabs_reg_aglo)

                        # Linear Regression
                        with reg_algo_tabs[0]:
                                st.markdown("""
        <br>

        ## 📏 Linear Regression


        Linear Regression is a widely used and versatile algorithm for predicting numerical values. It models the relationship between the independent variables (features) and the dependent variable (target) by fitting a linear equation to the data. The goal is to find the best-fit line that minimizes the difference between the predicted and actual values.

        --- 

        ### Examples


        Linear Regression can be applied to various real-world scenarios, such as:

        - **Housing Prices**: Predicting house prices based on features like area, number of rooms, location, etc.
        - **Stock Market Analysis**: Forecasting stock prices based on historical data and relevant factors.
        - **Demand Forecasting**: Estimating future demand for a product based on past sales and market trends.

        --- 

        ### Plots


        Linear Regression is often visualized using scatter plots and regression lines. Scatter plots show the distribution of data points, while the regression line represents the linear relationship between the features and the target variable.

        ---

        ### Abilities


        Linear Regression offers several benefits:

        - **Interpretability**: The linear equation's coefficients provide insights into the relationship between the features and the target variable.
        - **Simplicity**: Linear Regression is easy to understand and implement, making it suitable for both beginners and experts.
        - **Efficiency**: The training and prediction process is computationally efficient, allowing for quick model development.

        ---


        ### Pros and Cons


        **Pros:**
        - Linear Regression performs well when the relationship between the features and the target is linear.
        - It provides interpretability, allowing you to understand the impact of each feature on the target.
        - Linear Regression is computationally efficient and can handle large datasets.

        **Cons:**
        - Linear Regression assumes a linear relationship, which may not be suitable for datasets with complex non-linear relationships.
        - It is sensitive to outliers, which can significantly impact the model's performance.
        - Linear Regression is limited to modeling continuous numerical variables and may not be suitable for categorical or discrete targets.

        --- 

        ### Code:

        ```python
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        model.predict(X_test)
        ```

        ---

        In summary, Linear Regression is a powerful algorithm for predicting numerical values by fitting a linear equation to the data. It is widely used and offers interpretability, simplicity, and efficiency. However, it is important to consider its assumptions and limitations when applying it to real-world datasets.

        """, unsafe_allow_html=True)

                        # Decision Tree
                        with reg_algo_tabs[1]:
                                st.markdown("""

        <br>

        ### 🍁 Decision Tree

        Decision Tree is a versatile algorithm that can be used for both classification and regression tasks. In the context of regression, Decision Trees create a tree-like model of decisions and their possible consequences. It partitions the data based on the features to predict the numerical target variable.

        ---

        #### Examples

        Decision Trees can be applied to various real-world scenarios, such as:

        - **Medical Diagnosis**: Predicting a patient's blood pressure based on symptoms, age, and other medical factors.
        - **Crop Yield Prediction**: Estimating the yield of a crop based on environmental factors, soil quality, and cultivation techniques.
        - **Insurance Premium Estimation**: Determining the appropriate insurance premium for a policyholder based on risk factors like age, occupation, and health status.

        ---

        #### Plots

        Decision Trees can be visualized as tree-like structures, where each internal node represents a decision based on a feature, each branch represents an outcome, and each leaf node represents the predicted value. The depth of the tree determines the complexity of the model.


        ---

        #### Abilities

        Decision Trees offer several benefits:

        - **Interpretability**: Decision Trees provide a clear and interpretable representation of decision-making logic.
        - **Handling Non-linearity**: Decision Trees can capture non-linear relationships between features and the target variable.
        - **Feature Importance**: Decision Trees can identify important features for prediction, aiding feature selection.

        ---

        #### Pros and Cons

        **Pros:**
        - Decision Trees can handle both numerical and categorical features.
        - They can capture non-linear relationships between features and the target variable.
        - Decision Trees are computationally efficient and can handle large datasets.

        **Cons:**
        - Decision Trees are prone to overfitting, especially when the tree becomes too deep.
        - They can be sensitive to small variations in the training data, leading to different tree structures.
        - Decision Trees may not generalize well to unseen data if the training data is not representative.

        ---

        ### Code

        ```python
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        model.predict(X_test)
        ```

        --- 

        In summary, Decision Trees are powerful regression algorithms that create tree-like models to predict numerical values. They offer interpretability, handle non-linearity, and provide feature importance information. However, caution should be exercised to prevent overfitting and ensure generalizability.


        """, unsafe_allow_html=True)
                        
                        # Random Forest
                        with reg_algo_tabs[2]:
                                st.markdown("""
        <br>

        ### 🌳 Random Forest

        Random Forest is a powerful ensemble algorithm that combines multiple Decision Trees to create a robust regression model. It belongs to the family of ensemble methods, which aim to improve prediction performance by aggregating the predictions of multiple individual models.

        ---

        #### Ensemble Algorithms

        Ensemble methods work by combining the predictions of multiple models, often referred to as base models or weak learners, to make a final prediction. These base models are trained on different subsets of the training data or with different features to introduce diversity in their predictions. The ensemble then combines these predictions using various techniques to arrive at a final prediction.

        ---

        #### Characteristics of Random Forest

        Random Forest has the following characteristics:

        - **Multiple Decision Trees**: Random Forest consists of a collection of Decision Trees, each trained on a random subset of the training data.
        - **Bootstrap Aggregation**: The subsets of data for training each Decision Tree are created through a process called bootstrap aggregation or bagging. This involves random sampling of the training data with replacement.
        - **Feature Randomness**: In addition to data randomness, Random Forest also introduces feature randomness. Each Decision Tree is trained on a random subset of features, which helps to decorrelate the trees and increase diversity.

        ---

        #### Strengths of Random Forest

        Random Forest offers several strengths as a regression algorithm:

        - **Improved Generalization**: By combining predictions from multiple Decision Trees, Random Forest reduces overfitting and improves generalization performance.
        - **Robustness to Outliers and Noisy Data**: Random Forest is less sensitive to outliers and noisy data compared to individual Decision Trees.
        - **Feature Importance**: Random Forest provides a measure of feature importance, which helps in identifying the most influential features in the prediction.

        ---

        #### Other Factors in Ensemble Methods

        Ensemble methods, including Random Forest, exhibit the following factors:

        - **Bias-Variance Tradeoff**: Ensemble methods aim to strike a balance between bias and variance. Individual models with low bias and high variance can be combined to obtain a lower overall variance while maintaining reasonable bias.
        - **Parallelizability**: Ensemble methods can be easily parallelized, allowing for efficient training and prediction on large datasets.
        - **Model Diversity**: The performance of an ensemble relies on the diversity of the individual models. The base models should be different from each other in terms of the data they are trained on or the features they use.

        ---

        ### Code

        ```python
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        model.predict(X_test)
        ```

        ---

        In summary, Random Forest is an ensemble algorithm that combines multiple Decision Trees to create a robust regression model. It leverages the diversity of individual trees to improve generalization and provides feature importance information. Ensemble methods, including Random Forest, aim to strike a balance between bias and variance while benefiting from model diversity.


        """, unsafe_allow_html=True)

                        # Support Vector Regression
                        with reg_algo_tabs[3]:
                                st.markdown("""
        <br>

        ### ⛑️ Support Vector Regression

        Support Vector Regression (SVR) is a powerful regression algorithm that utilizes the principles of Support Vector Machines to perform regression tasks. Similar to its classification counterpart, SVR aims to find a hyperplane that best fits the data points while maximizing the margin.

        ---

        #### Characteristics of Support Vector Regression

        SVR has the following characteristics:

        - **Kernel Trick**: SVR employs the kernel trick, allowing it to operate in high-dimensional feature spaces without explicitly calculating the transformations. This enables SVR to capture complex nonlinear relationships between the features and the target variable.
        - **Margin Maximization**: SVR seeks to find a hyperplane that maintains a maximum margin while tolerating a certain amount of error, known as the epsilon-tube. Data points outside this epsilon-tube are considered outliers.
        - **Support Vectors**: SVR uses a subset of training data points called support vectors, which lie on or within the margin or the epsilon-tube. These support vectors heavily influence the position and orientation of the regression hyperplane.

        ---

        #### Strengths of Support Vector Regression

        Support Vector Regression offers several strengths as a regression algorithm:

        - **Flexibility**: SVR can effectively model both linear and nonlinear relationships between features and the target variable by utilizing different kernel functions, such as linear, polynomial, radial basis function (RBF), or sigmoid.
        - **Robustness to Outliers**: SVR is robust to outliers, as it focuses on maximizing the margin and is less influenced by individual data points lying outside the margin or the epsilon-tube.
        - **Regularization**: SVR incorporates a regularization parameter, C, which controls the tradeoff between minimizing the training error and the complexity of the model. This allows for controlling overfitting and improving generalization.

        ---

        #### Considerations when using Support Vector Regression

        When working with Support Vector Regression, it's important to consider the following:

        - **Feature Scaling**: Feature scaling, such as normalization or standardization, is crucial when using SVR. SVR is sensitive to the scale of the features, and unscaled features may lead to suboptimal performance.
        - **Model Complexity**: SVR's performance is highly dependent on the choice of hyperparameters, including the kernel function, regularization parameter C, and kernel-specific parameters. Proper tuning of these hyperparameters is essential for achieving optimal performance.
        - **Computational Complexity**: SVR's training time can be relatively higher compared to some other regression algorithms, especially for large datasets. Additionally, the memory requirements for storing support vectors can be significant.

        ---

        ### Code

        ```python
        from sklearn.svm import SVR
        model = SVR()
        model.fit(X_train, y_train)
        model.predict(X_test)
        ```

        --- 

        In summary, Support Vector Regression (SVR) is a flexible regression algorithm that can effectively model linear and nonlinear relationships. It utilizes the kernel trick, maximizes the margin, and employs support vectors to influence the regression hyperplane. SVR is robust to outliers, provides regularization, but requires proper feature scaling and hyperparameter tuning.


        """, unsafe_allow_html=True)
                        
                        # K Nearest Neighbors
                        with reg_algo_tabs[4]:
                                st.markdown("""
        <br>

        ### 🏘️ K Nearest Neighbors (KNN)

        K Nearest Neighbors (KNN) is a simple yet powerful non-parametric algorithm used for both classification and regression tasks. In the context of regression, KNN predicts the target variable of a new data point by considering the average or weighted average of its K nearest neighbors.

        --- 

        #### Characteristics of K Nearest Neighbors

        KNN exhibits the following characteristics:

        - **Lazy Learning**: KNN is often referred to as a "lazy algorithm" because it doesn't have a traditional training phase. Instead, it stores the entire training dataset and performs computations only when making predictions on new data points. This makes the training time relatively short but can lead to longer testing times.
        - **Distance-based Similarity**: KNN determines the proximity between data points based on a distance metric, commonly Euclidean distance. The K nearest neighbors of a given data point are identified based on their distance to that point.
        - **Non-Parametric**: KNN makes no assumptions about the underlying data distribution, making it a non-parametric algorithm. It doesn't require any assumptions about the functional form of the relationship between the features and the target variable.

        ---

        #### Strengths of K Nearest Neighbors

        K Nearest Neighbors offers several strengths as a regression algorithm:

        - **Flexibility**: KNN can handle both linear and non-linear relationships between features and the target variable. It is capable of capturing complex patterns in the data, making it suitable for a wide range of regression tasks.
        - **Interpretability**: KNN provides transparency in the decision-making process. Predictions are made based on the actual values of neighboring data points, allowing for easy interpretation of results.
        - **Non-Parametric Nature**: KNN's non-parametric nature makes it more robust to outliers and less sensitive to skewed data distributions compared to parametric regression algorithms.

        ---

        #### Considerations when using K Nearest Neighbors

        When working with K Nearest Neighbors, it's important to consider the following:

        - **Feature Scaling**: Feature scaling is essential when using KNN, as it relies on the distance metric to identify neighbors. Features with larger scales can dominate the distance calculation, leading to biased results. Therefore, it's recommended to scale the features before applying KNN.
        - **Choosing the Value of K**: The choice of the parameter K, representing the number of nearest neighbors, is critical. A small value of K may lead to overfitting, while a large value may lead to underfitting. It's important to experiment with different values of K and choose the optimal value through cross-validation or other evaluation techniques.
        - **Computational Complexity**: KNN's testing time can be relatively high, especially for large datasets, as it requires calculating distances between the new data point and all training data points. Therefore, efficient data structures and algorithms, such as KD-trees or Ball-trees, can be employed to speed up the nearest neighbor search process.

        ---

        ### Code

        ```python
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(k=5) # k is the number of neighbors (we will talk about it later)
        model.fit(X_train, y_train)
        model.predict(X_test)
        ```

        ---

        In summary, K Nearest Neighbors (KNN) is a flexible and interpretable regression algorithm. Its non-parametric nature and ability to capture complex patterns make it a suitable choice for various regression tasks. However, careful consideration should be given to feature scaling, choosing the value of K, and the computational complexity associated with larger datasets.


        """, unsafe_allow_html=True)
                        
                        # XGBoost
                        with reg_algo_tabs[5]:
                                st.markdown("""
        <br>

        ### 💥 XGBoost

        XGBoost (Extreme Gradient Boosting) is a popular and powerful machine learning algorithm known for its efficiency and effectiveness in both classification and regression tasks. It is based on the concept of gradient boosting, a type of ensemble learning technique.

        ---

        #### Characteristics of XGBoost

        XGBoost exhibits the following characteristics:

        - **Ensemble Learning**: XGBoost is an ensemble learning algorithm that combines the predictions of multiple weak models, known as decision trees, to make accurate predictions. It uses a boosting technique to sequentially train new models that focus on the errors made by the previous models, thereby improving overall prediction performance.
        - **Boosting Algorithm**: XGBoost belongs to the family of boosting algorithms, where each subsequent model in the ensemble is trained to correct the mistakes of the previous models. This iterative process allows XGBoost to gradually improve the overall predictive power by combining weak learners into a strong ensemble.
        - **Gradient Optimization**: XGBoost employs gradient-based optimization techniques to find the optimal values of model parameters. It uses gradient descent algorithms to minimize a specified loss function, resulting in better model fitting and increased predictive accuracy.

        ---

        #### Strengths of XGBoost

        XGBoost offers several strengths as a regression algorithm:

        - **High Accuracy**: XGBoost is renowned for its high prediction accuracy and performance. It leverages the ensemble of decision trees and gradient optimization techniques to make accurate predictions on a wide range of regression problems.
        - **Feature Importance**: XGBoost provides insights into feature importance, allowing users to understand the relative importance of different features in the prediction process. This information can be valuable for feature selection and understanding the underlying relationships in the data.
        - **Regularization Techniques**: XGBoost offers various regularization techniques, such as L1 and L2 regularization, which can help prevent overfitting and improve the model's generalization capability.
        - **Handling Missing Data**: XGBoost has built-in capabilities to handle missing data, eliminating the need for preprocessing steps such as imputation. It can effectively handle missing values during the training and prediction phases.

        ---

        #### Considerations when using XGBoost

        When working with XGBoost, it's important to consider the following:

        - **Parameter Tuning**: XGBoost has several hyperparameters that can significantly impact its performance. It's essential to tune these parameters carefully to achieve optimal results. Techniques such as grid search and random search can be used to find the best combination of hyperparameters.
        - **Computational Complexity**: XGBoost can be computationally expensive, especially for large datasets and complex models. It's important to consider the available computational resources and training time requirements when using XGBoost.
        - **Interpretability**: As an ensemble of decision trees, the interpretability of XGBoost may be lower compared to simpler regression algorithms. However, techniques such as feature importance can provide insights into the model's behavior.

        ---

        ### Code

        ```python
        # You need to install XGBoost first using the following command:
        # pip install xgboost
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.fit(X_train, y_train)
        model.predict(X_test)
        ```

        ---
        In summary, XGBoost is a powerful regression algorithm that leverages ensemble learning and gradient optimization techniques. Its high accuracy, feature importance analysis, and built-in regularization capabilities make it a popular choice for regression tasks. Careful parameter tuning and consideration of computational resources are essential for optimal performance.


        """, unsafe_allow_html=True)               



                # Regression Evaluation Metrics
                st.markdown("""
                
<br> 
                
---
                
<br> 

## 💯 Regression Evaluation Metrics 
                
<br> 

Evaluation metrics are used to assess the performance of machine learning models. In this section, we will explore various evaluation metrics for regression models, such as mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), and R-squared (R2).
The Purpose of Evaluation Metrics is to evaluate the performance of the model. We use the evaluation metrics to compare between different models and choose the best one. We also use the evaluation metrics to evaluate the model on the testing set. The evaluation metrics are different from one problem to another. For example, we use different evaluation metrics for classification problems and regression problems. In this section, we will talk about the evaluation metrics for regression problems.

<br>

""", unsafe_allow_html=True)

                # Expander
                with st.expander("💯 Regression Evaluation Metrics"):
                        
                        st.write("\n")

                        # Evaluation tabs
                        tab_titles_eval = [" 🥪 Mean Absolute Error (MAE)" , " 🌀 Mean Squared Error (MSE)", " 🌱 Root Mean Squared Error (RMSE)" , " 🎯 R-squared (R2)"]
                        eval_tabs = st.tabs(tab_titles_eval)

                        # MAE
                        with eval_tabs[0]:
                        
                                st.markdown("""
                        ## 🥪 MAE (Mean Absolute Error)

        The Mean Absolute Error (MAE) is a commonly used evaluation metric in regression tasks. It measures the average absolute difference between the predicted and actual values of a regression model. The MAE provides a simple and interpretable measure of the model's performance.

        ### Equation

        The MAE is calculated by taking the average of the absolute differences between the predicted values (y_pred) and the actual values (y_true) for a set of data points:
        """, unsafe_allow_html=True)

                                st.latex(r''' MAE = \frac{1}{n} \sum_{i=1}^{n} |y_{true} - y_{pred}| ''')
                        
                                st.markdown("""


        where:
        - $MAE$: Mean Absolute Error
        - $n$: Number of data points in the dataset
        - $Σ$: Summation symbol
        - $y_{true}$: Actual values
        - $y_{pred}$: Predicted values

        ### Usage

        MAE is used to assess the overall accuracy of a regression model. It is particularly useful when the magnitude of errors is essential and needs to be measured in the original units of the target variable. MAE provides a straightforward interpretation of the average absolute error.

        ### Interpretation

        A lower MAE value indicates better performance, as it represents a smaller average difference between the predicted and actual values. It measures the average magnitude of errors without considering their direction, making it less sensitive to outliers.

        ### Pros and Cons

        #### Pros:
        - **Intuitive Interpretation**: MAE is easy to interpret as it represents the average absolute difference between predicted and actual values.
        - **Robust to Outliers**: MAE is less affected by outliers since it treats all errors with equal importance.
        - **Same Scale as the Target Variable**: MAE is in the same units as the target variable, making it easy to relate to the problem domain.

        #### Cons:
        - **Lack of Sensitivity to Error Magnitude**: MAE treats all errors equally, regardless of their magnitude. It may not adequately penalize large errors if precise estimation of error magnitude is required.
        - **Does Not Provide Directional Information**: MAE does not indicate the direction of errors, making it difficult to identify whether the model tends to overestimate or underestimate the target variable.

        The MAE metric is a valuable tool for evaluating the performance of regression models, providing an intuitive measure of the average absolute error. However, it is essential to consider the specific requirements of the problem and the trade-offs between different evaluation metrics.

        ### Code

        ```python
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_true, y_pred)
        print("MAE:", mae)
        ```

                                """, unsafe_allow_html=True)        
                        
                        # MSE
                        with eval_tabs[1]:
                        
                                st.markdown("""
                        ## 🌀 MSE (Mean Squared Error)

        The Mean Squared Error (MSE) is a commonly used evaluation metric in regression tasks. It measures the average of the squared differences between the predicted and actual values of a regression model. The MSE provides a measure of the average squared error, which gives more weight to larger errors.

        ### Equation

        The MSE is calculated by taking the average of the squared differences between the predicted values (y_pred) and the actual values (y_true) for a set of data points:

        """, unsafe_allow_html= True)

                                st.latex(r''' MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{true} - y_{pred})^2 ''')
                        
                                st.markdown("""


        where:
        - $MSE$: Mean Squared Error
        - $n$: Number of data points in the dataset
        - $Σ$: Summation symbol
        - $y_{true}$: Actual values
        - $y_{pred}$: Predicted values

        ### Usage

        MSE is used to assess the overall accuracy of a regression model. It is particularly useful when larger errors need to be penalized more compared to smaller errors. The MSE provides a measure of the average squared error.

        ### Interpretation

        A lower MSE value indicates better performance, as it represents a smaller average squared difference between the predicted and actual values. MSE measures the average magnitude of errors and penalizes larger errors more than MAE.

        ### Pros and Cons

        #### Pros:
        - **Sensitive to Large Errors**: MSE gives more weight to larger errors due to the squaring operation, making it more sensitive to outliers or extreme values.
        - **Mathematically Convenient**: Squaring the errors makes the metric mathematically convenient for optimization algorithms, as it is differentiable and enables gradient-based optimization.
        - **Same Scale as the Target Variable**: MSE is in the squared units of the target variable, which can be useful for comparing against the variance of the target variable.

        #### Cons:
        - **Lack of Intuitive Interpretation**: MSE is not as easily interpretable as MAE since it is in squared units of the target variable.
        - **Large Errors are Heavily Penalized**: MSE heavily penalizes large errors due to the squaring operation, which may not be desirable in certain applications.

        The MSE metric is commonly used to evaluate the performance of regression models, giving more weight to larger errors due to the squaring operation. However, it is important to consider the specific requirements of the problem and the trade-offs between different evaluation metrics.

        ### Code

        ```python
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_true, y_pred)
        print("MSE:", mse)
        ```


        """, unsafe_allow_html=True)
                        
                        # RMSE
                        with eval_tabs[2]:
                                
                                st.markdown("""
                                ## 🌱 RMSE (Root Mean Squared Error)

        The Root Mean Squared Error (RMSE) is a popular evaluation metric in regression tasks. It is an extension of the Mean Squared Error (MSE) that addresses the issue of the MSE being in squared units. RMSE provides a measure of the average magnitude of the errors in the same unit as the target variable.

        ### Equation

        RMSE is calculated by taking the square root of the average of the squared differences between the predicted values (y_pred) and the actual values (y_true) for a set of data points:

        """, unsafe_allow_html=True)

                                st.latex(r''' RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{true} - y_{pred})^2} ''')
                                
                                st.markdown("""

        where:

        - $RMSE$: Root Mean Squared Error
        - $n$: Number of data points in the dataset
        - $Σ$: Summation symbol
        - $y_{true}$: Actual values
        - $y_{pred}$: Predicted values
        - $\sqrt{}$: Square root

        ### Usage

        RMSE is widely used to assess the performance of regression models, especially when the error values need to be interpreted in the same unit as the target variable. It provides a measure of the average magnitude of the errors.

        ### Interpretation

        RMSE measures the square root of the average squared difference between the predicted and actual values. A lower RMSE value indicates better performance, as it represents a smaller average magnitude of the errors.

        ### Pros and Cons

        #### Pros:
        - **Interpretability**: RMSE is more easily interpretable than MSE since it is in the same unit as the target variable.
        - **Same Scale as the Target Variable**: RMSE provides a measure of the average magnitude of errors in the same unit as the target variable, which enhances interpretability.
        - **Sensitive to Large Errors**: RMSE, like MSE, gives more weight to larger errors due to the squaring operation.

        #### Cons:
        - **Lack of Intuitive Interpretation**: While RMSE is in the same unit as the target variable, its interpretation may still require domain knowledge and context.
        - **Large Errors are Heavily Penalized**: RMSE, like MSE, heavily penalizes large errors due to the squaring operation.

        The RMSE metric is widely used in regression tasks to evaluate the performance of models, providing an interpretable measure of the average magnitude of errors in the same unit as the target variable.

        ### Code

        ```python
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        print("RMSE:", rmse)
        ```

        """, unsafe_allow_html=True)

                        # R2 Score     
                        with eval_tabs[3]:
                                
                                st.markdown("""
                                ## 🎯 R-squared (R2)

        The R2 Score, also known as the coefficient of determination, is a commonly used evaluation metric for regression tasks. It measures the proportion of the variance in the dependent variable that can be explained by the independent variables.

        ### Equation

        R2 Score is calculated using the following formula:

        """, unsafe_allow_html=True)

                                st.latex(r''' R^2 = 1 - \frac{SS_{res}}{SS_{tot}} ''')
                                
                                st.markdown("""

        where:

        - $R^2$: R2 Score
        - $SS_{res}$: Sum of squared residuals
        - $SS_{tot}$: Total sum of squares

        ### Usage

        R2 Score is used to assess how well a regression model fits the given data. It provides an indication of the proportion of the variance in the dependent variable that can be explained by the independent variables.

        ### Interpretation

        The R2 Score ranges between 0 and 1. Here's how to interpret the R2 Score:

        - **R2 Score = 1**: The model perfectly predicts the target variable.
        - **R2 Score = 0**: The model fails to capture any relationship between the independent and dependent variables.
        - **R2 Score < 0**: The model performs worse than a horizontal line (the mean of the target variable).

        ### Pros and Cons

        #### Pros:
        - **Interpretability**: R2 Score provides a measure of how well the model fits the data, ranging from 0 to 1.
        - **Relative Comparison**: R2 Score allows for the comparison of different models based on their performance.
        - **Normalization**: R2 Score is normalized and does not depend on the scale of the target variable.

        #### Cons:
        - **Dependence on Model Complexity**: R2 Score may not accurately reflect the quality of the model if the model is too simple or too complex.
        - **Does Not Capture Overfitting**: R2 Score alone may not be sufficient to detect overfitting or the generalizability of the model.

        The R2 Score is a valuable metric to assess the goodness of fit of a regression model, indicating the proportion of the variance in the dependent variable that can be explained by the independent variables.

        ### Code

        ```python
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        print("R2 Score:", r2)
        ```

        """, unsafe_allow_html=True)
                                                        


        # Classification Models
        with ml_tabs[1]:
                st.write('\n\n\n\n\n')

                st.markdown("""

                ### 🪀 Overview of Classification Models

        Classification models are a class of machine learning models used for predicting categorical or discrete class labels. In this section, we will provide an overview of classification models and their key concepts.

        #### 🍴 Binary Classification

        Binary classification is a type of classification task where the goal is to predict one of two possible classes. The model learns from labeled data and assigns new instances to one of the two classes based on the learned patterns. Common algorithms used for binary classification include logistic regression, support vector machines (SVM), and decision trees.

        #### 🧊 Multiclass Classification

        Multiclass classification involves predicting one class out of three or more possible classes. It extends binary classification to handle multiple classes. The model learns from labeled data with multiple classes and assigns new instances to one of the classes. Algorithms like random forest, k-nearest neighbors (KNN), and neural networks are commonly used for multiclass classification.


        <br>

        ---

        <br>

        ### 🛹 Classification Algorithms

        There are numerous classification algorithms available, each with its strengths and limitations. Some popular algorithms include **Logistic Regression**, **Decision Trees**, **Random Forest**, **Support Vector Machines (SVM)**, **K Nearest Neighbors (KNN)**, and **XGBoost**.
        
        <br> 

         """, unsafe_allow_html=True)
                

                # expander
                with st.expander("🛹 Classification Algorithms"):
                      
                        st.write("\n")

                        # Classification Algorithms Tabs
                        tabs_class_aglo = [" 📦 Logistic Regression", " 🍁 Decision Tree", " 🌳 Random Forest", " ⛑️ Support Vector Machine", " 🏘️ K Nearest Neighbors", " 💥 XGBoost"]
                        class_algo_tabs = st.tabs(tabs_class_aglo)

                        # Logistic Regression
                        with class_algo_tabs[0]:
                                st.markdown("""

                                ## 📦 Logistic Regression

Logistic Regression is a popular algorithm for binary classification tasks. It models the relationship between the dependent variable and one or more independent variables by estimating the probabilities using a logistic function.

### How It Works

Logistic Regression works by fitting a logistic curve to the training data, which allows it to predict the probability of an instance belonging to a particular class. It uses the logistic function, also known as the sigmoid function, to map the input values to a range between 0 and 1.

### Equation

The logistic function used in Logistic Regression is given by the following equation:

""", unsafe_allow_html=True)

                                st.latex(r''' f(x) = \frac{1}{1 + e^{-x}} ''')

                                st.markdown("""

where:

- $f(x)$: Logistic function
- $e$: Euler's number
- $x$: Input value


<br> 

### Pros and Cons

#### Pros:
- **Interpretability**: Logistic Regression provides interpretable coefficients that indicate the influence of each feature on the prediction.
- **Efficiency**: Logistic Regression is computationally efficient and can handle large datasets.
- **Works well with linearly separable data**: Logistic Regression performs well when the decision boundary between classes is linear.

#### Cons:
- **Assumption of linearity**: Logistic Regression assumes a linear relationship between the independent variables and the log-odds of the dependent variable.
- **Limited to binary classification**: Logistic Regression is primarily used for binary classification tasks and may not perform well for multi-class problems without modifications.


<br>

### Code Example

Here's an example code snippet for implementing Logistic Regression using the scikit-learn library in Python:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.predict(X_test)
```

""", unsafe_allow_html=True)
           
                        # Decision Tree
                        with class_algo_tabs[1]:
                              
                                st.markdown("""

                                ## 🍁 Decision Tree

Decision Tree is a popular algorithm for classification tasks. It builds a tree-like model of decisions and their possible consequences based on the features of the data. Each internal node represents a feature, each branch represents a decision rule, and each leaf node represents the outcome or class label.

### How It Works

Decision Tree works by recursively splitting the data based on the values of the features to create homogeneous subsets. The splits are made based on certain criteria, such as Gini impurity or information gain, to maximize the homogeneity or purity of the subsets with respect to the target variable.

<br>

### Pros and Cons

#### Pros:
- **Interpretability**: Decision Trees provide intuitive interpretations as they mimic human decision-making processes.
- **Handling both numerical and categorical data**: Decision Trees can handle both numerical and categorical features without requiring extensive preprocessing.
- **Feature importance**: Decision Trees can rank the importance of features based on their contribution to the splits.

#### Cons:
- **Overfitting**: Decision Trees have a tendency to overfit the training data, leading to poor generalization on unseen data. Techniques like pruning can be applied to alleviate this issue.
- **Instability**: Decision Trees are sensitive to small changes in the data and can produce different trees with different splits.
- **Bias towards features with more levels**: Decision Trees with categorical features tend to favor features with more levels or categories.

<br>

### Code Example

Here's an example code snippet for implementing Decision Tree using the scikit-learn library in Python:

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
model.predict(X_test)
```

""", unsafe_allow_html=True)

                        # Random Forest
                        with class_algo_tabs[2]:
                              
                                st.markdown("""

                                ## 🌳 Random Forest

Random Forest is a popular ensemble algorithm for classification tasks. It combines multiple decision trees to create a more robust and accurate model. Each decision tree in the Random Forest is built on a random subset of the data and features.

### How It Works

Random Forest works by creating an ensemble of decision trees. Each tree is trained on a random subset of the data through a process called bootstrap aggregating or "bagging." Additionally, for each split in the tree, a random subset of features is considered, reducing the correlation between trees. The final prediction is made by aggregating the predictions of all the individual trees.

### Ensemble and Boosting

Random Forest is an ensemble algorithm because it combines multiple weak learners (decision trees) to create a strong learner. Ensemble methods leverage the diversity of individual models to improve the overall prediction accuracy and reduce overfitting.

<br>

### Pros and Cons

#### Pros:
- **High Accuracy**: Random Forest tends to achieve high accuracy due to the combination of multiple decision trees.
- **Robustness**: Random Forest is resistant to overfitting and performs well on a variety of datasets.
- **Feature Importance**: Random Forest can provide feature importance scores, indicating the contribution of each feature in the classification task.

#### Cons:
- **Computational Complexity**: Random Forest can be computationally expensive, especially when dealing with large datasets or a large number of trees.
- **Lack of Interpretability**: The individual trees in the Random Forest are not easily interpretable, unlike a single decision tree.

<br>

### Code Example

Here's an example code snippet for implementing Random Forest using the scikit-learn library in Python:

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
model.predict(X_test)
```

""", unsafe_allow_html=True)

                        # Support Vector Machine
                        with class_algo_tabs[3]:
                              
                                st.markdown("""

                                ## ⛑️ Support Vector Machine

Support Vector Machine (SVM) is a powerful supervised learning algorithm used for classification tasks. It finds an optimal hyperplane that best separates the different classes in the feature space.

### How It Works

SVM works by mapping the input data into a high-dimensional feature space and finding a hyperplane that maximally separates the classes. The hyperplane is determined by support vectors, which are the data points closest to the decision boundary. SVM can handle both linear and non-linear classification tasks using different kernel functions, such as linear, polynomial, and radial basis function (RBF) kernels.

### Strengths and Applications

- **Effective in High-Dimensional Spaces**: SVM performs well even in cases where the number of dimensions is greater than the number of samples. This makes it suitable for tasks with a large number of features.
- **Ability to Handle Non-Linear Data**: By using kernel functions, SVM can effectively handle non-linear classification problems by mapping the data into a higher-dimensional space.
- **Robust to Outliers**: SVM is less sensitive to outliers compared to other classification algorithms.

<br>

### Pros and Cons

#### Pros:
- **Strong Generalization**: SVM aims to find the best decision boundary with the largest margin, which often leads to good generalization performance on unseen data.
- **Effective with High-Dimensional Data**: SVM can handle high-dimensional data efficiently.
- **Flexibility with Kernels**: SVM allows the use of different kernel functions to capture complex relationships between features.

#### Cons:
- **Computationally Expensive**: SVM can be computationally expensive, especially when dealing with large datasets.
- **Sensitive to Noise**: SVM performance can be affected by noisy data, so it's important to preprocess the data and handle outliers carefully.
- **Difficult Interpretability**: SVM can be challenging to interpret, especially in high-dimensional spaces.

<br>

### Code Example

Here's an example code snippet for implementing SVM using the scikit-learn library in Python:

```python
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
model.predict(X_test)
```

""", unsafe_allow_html=True)

                        # K Nearest Neighbors
                        with class_algo_tabs[4]:
                                  
                                    st.markdown("""
        
                                    ## 🏘️ K Nearest Neighbors


K-Nearest Neighbors (KNN) is a simple yet effective supervised learning algorithm used for classification tasks. It classifies new data points based on the majority class of its k nearest neighbors in the feature space.

### How It Works

KNN works by measuring the distance between the new data point and the existing data points in the training set. It then selects the k nearest neighbors based on the chosen distance metric (e.g., Euclidean distance) and assigns the class label based on the majority vote among those neighbors.

### Strengths and Applications

- **Simplicity and Intuition**: KNN is easy to understand and implement, making it a popular choice for beginners in machine learning.
- **Non-Parametric and Lazy Learning**: KNN makes no assumptions about the underlying data distribution and does not require explicit model training. It learns from the data at the prediction stage, making it a lazy learning algorithm.
- **Ability to Handle Non-Linear Data**: KNN can effectively classify non-linear data by considering local patterns and relationships.

<br>

### Pros and Cons

#### Pros:
- **Simple Implementation**: KNN is straightforward to implement, making it an accessible algorithm for classification tasks.
- **No Training Phase**: KNN does not require an explicit training phase, as it learns from the data during prediction.
- **Non-Parametric**: KNN makes no assumptions about the underlying data distribution, giving it flexibility in handling diverse datasets.

#### Cons:
- **Computationally Expensive**: KNN can be computationally expensive, especially when dealing with large datasets or high-dimensional feature spaces.
- **Sensitive to Noise and Irrelevant Features**: KNN is sensitive to noisy data and irrelevant features, which can impact its classification accuracy.
- **Determining Optimal K**: Choosing the appropriate value of k (number of neighbors) can be challenging and may require experimentation.

<br>

### Code Example

Here's an example code snippet for implementing KNN using the scikit-learn library in Python:

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
model.predict(X_test)
```

""", unsafe_allow_html=True)
                        
                        # XGBoost
                        with class_algo_tabs[5]:
                                        
                                st.markdown("""
        
                                        ## 💥 XGBoost

XGBoost (Extreme Gradient Boosting) is a powerful and popular machine learning algorithm known for its exceptional performance in classification tasks. It is an ensemble learning method that combines multiple weak classifiers (decision trees) to create a strong predictive model.

### How It Works

XGBoost works by iteratively adding decision trees to improve the predictive performance. It builds trees in a sequential manner, where each subsequent tree tries to correct the errors made by the previous trees. It uses gradient boosting, a technique that minimizes a loss function by optimizing the gradient of the loss with respect to the model predictions.

### Strengths and Applications

- **High Predictive Accuracy**: XGBoost is known for its exceptional predictive accuracy and is widely used in various machine learning competitions and real-world applications.
- **Handles Complex Relationships**: XGBoost can capture complex patterns and interactions in the data, making it suitable for datasets with intricate relationships.
- **Regularization and Feature Importance**: XGBoost incorporates regularization techniques to prevent overfitting and provides feature importance scores, aiding in feature selection.

<br>

### Pros and Cons

#### Pros:
- **Highly Accurate Predictions**: XGBoost achieves state-of-the-art performance in many machine learning tasks due to its strong modeling capabilities.
- **Handles Complex Data**: XGBoost can effectively handle high-dimensional data and capture complex relationships between features.
- **Regularization and Control Overfitting**: XGBoost incorporates regularization techniques to prevent overfitting and improve generalization.

#### Cons:
- **Computationally Expensive**: XGBoost can be computationally expensive, especially when dealing with large datasets and complex models.
- **Sensitive to Hyperparameters**: XGBoost requires careful tuning of hyperparameters to achieve optimal performance, which can be a time-consuming process.
- **Requires Sufficient Data**: XGBoost typically requires a sufficient amount of data to train an accurate model and may not perform well with small datasets.

<br>

### Code Example

Here's an example code snippet for implementing XGBoost using the XGBoost library in Python:

```python
# You need to install XGBoost first using the following command:
# pip install xgboost
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
model.predict(X_test)
```

""", unsafe_allow_html=True)

                # Classification Evaluation Metrics
                st.markdown("""

<br>

---

<br>

### 💯 Evaluation Metrics


To assess the performance of classification models, various evaluation metrics are used. Some commonly used metrics include accuracy, precision, recall, and F1-score. These metrics provide insights into the model's ability to correctly classify instances and its overall performance. <br> <br>
The Purpose of Evaluation Metrics is to evaluate the performance of the model. We use the evaluation metrics to compare between different models and choose the best one. We also use the evaluation metrics to evaluate the model on the testing set. The evaluation metrics are different from one problem to another. For example, we use different evaluation metrics for classification problems and regression problems. In this section, we will talk about the evaluation metrics for classification problems. <br> <br> 
Some Metrics has high importance than others. For example, in the case of imbalanced data, we use the F1-score instead of accuracy. Also, in Medical problems, the recall is more important than precision.     
        """ , unsafe_allow_html=True)
                
                st.write("\n")
                # Expander
                with st.expander("💯 Classification Evaluation Metrics"):
                      
                        st.write("\n")


                        # Evaluation tabs
                        tab_titles_eval = [" 󠁪󠁪 󠁪 󠁪 󠁪 󠁪💢 Confusion Matrix 󠁪 󠁪 󠁪 󠁪 󠁪", " 󠁪 󠁪 󠁪 󠁪 󠁪🎯 Accuracy 󠁪 󠁪 󠁪 󠁪 󠁪 " , "󠁪 󠁪 󠁪 󠁪 󠁪 🌡️ Precision 󠁪 󠁪 󠁪 󠁪 󠁪", " 󠁪 󠁪 󠁪 󠁪 󠁪 📲 Recall 󠁪 󠁪 󠁪 󠁪 󠁪" , " 󠁪 󠁪 󠁪 󠁪 󠁪 ⚾ F1-score 󠁪 󠁪 󠁪 󠁪 󠁪 󠁪 "]
                        eval_tabs = st.tabs(tab_titles_eval)
                        

                        # Confusion Matrix
                        with eval_tabs[0]:
                                
                                st.markdown("""

## 󠁪󠁪󠁪󠁪󠁪💢 Confusion Matrix

The Confusion Matrix is a performance measurement for classification models that summarizes the results of the predictions made by the model on a set of test data. It provides a detailed breakdown of the model's performance by counting the true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions.

<br>

### Understanding TP, TN, FP, and FN

- **True Positive (TP)**: The number of positive instances correctly predicted by the model as positive. These are the cases where the model predicted the class correctly.
- **True Negative (TN)**: The number of negative instances correctly predicted by the model as negative. These are the cases where the model predicted the absence of the class correctly.
- **False Positive (FP)**: The number of negative instances incorrectly predicted by the model as positive. These are the cases where the model predicted the presence of the class, but it was not present in reality.
- **False Negative (FN)**: The number of positive instances incorrectly predicted by the model as negative. These are the cases where the model predicted the absence of the class, but it was present in reality.

<br>

### Example Confusion Matrix

|         | Predicted Negative | Predicted Positive |
|---------|-------------------|-------------------|
| **Actual Negative** | True Negative (TN) | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP) |

<br>

Here's an example confusion matrix to illustrate how TP, TN, FP, and FN are arranged in a tabular form:


|         | Predicted Negative | Predicted Positive |
|---------|-------------------|-------------------|
| **Actual Negative** | 90 | 10 |
| **Actual Positive** | 15 | 85 |

<br>

In this example, we have a binary classification problem with two classes: Negative and Positive. The model correctly predicted 90 instances as Negative (TN), incorrectly predicted 10 instances as Positive (FP), incorrectly predicted 15 instances as Negative (FN), and correctly predicted 85 instances as Positive (TP).

The Confusion Matrix provides valuable insights into the performance of the classification model, allowing us to calculate various evaluation metrics such as accuracy, precision, recall, and F1-score.

<br>

### Code

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)
```



""", unsafe_allow_html=True)

                        # Accuracy
                        with eval_tabs[1]:
                              
                                st.markdown("""

                                ## 󠁪󠁪󠁪󠁪󠁪🎯 Accuracy

Accuracy is a widely used evaluation metric for classification models that measures the overall correctness of the predictions made by the model. It calculates the ratio of correctly classified instances to the total number of instances in the dataset.

### Equation

Accuracy is calculated using the following equation:

""", unsafe_allow_html=True)
                                
                                st.latex(r''' Accuracy = \frac{TP + TN}{TP + TN + FP + FN} ''')

                                st.markdown("""

where:

- $TP$: True Positive
- $TN$: True Negative
- $FP$: False Positive
- $FN$: False Negative

<br>


### Usage and Interpretation

Accuracy is commonly used to assess the performance of a classification model, especially when the classes in the dataset are balanced (approximately equal number of instances for each class). It provides an overall measure of how well the model predicts both positive and negative instances.

### Pros and Cons

**Pros:**
- Provides a straightforward and intuitive evaluation of model performance.
- Suitable for balanced datasets or when the cost of misclassification for both classes is similar.
- Easy to interpret and communicate to stakeholders.

**Cons:**
- Accuracy alone may not be a reliable measure when dealing with imbalanced datasets, where one class dominates the others in terms of the number of instances.
- It does not provide information about the specific types of errors the model is making (e.g., false positives or false negatives).
- Accuracy may give misleading results when applied to datasets with varying class distributions or when the cost of misclassification differs significantly between classes.

Accuracy is a useful metric for initial assessment of a classification model's performance, but it should be complemented with other evaluation metrics, especially when dealing with imbalanced datasets or when the costs of different types of errors are not equal.

### Code

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)
```

""", unsafe_allow_html=True)


                        # Precision
                        with eval_tabs[2]:
                               
                                st.markdown("""

                                ## 󠁪󠁪󠁪󠁪󠁪🌡️ Precision

Precision is an evaluation metric for classification models that measures the proportion of correctly predicted positive instances out of the total instances predicted as positive. It focuses on the accuracy of the positive predictions made by the model.

### Equation

Precision is calculated using the following equation:

""", unsafe_allow_html=True)

                                st.latex(r''' Precision = \frac{TP}{TP + FP} ''')

                                st.markdown("""

where:

- $TP$: True Positive
- $FP$: False Positive

<br>


### Usage and Interpretation

Precision is particularly useful when the goal is to minimize false positives. It provides insights into the model's ability to accurately classify positive instances and avoid false positives.

A high precision value indicates that the model has a low rate of falsely predicting positive instances, making it valuable in scenarios where false positives are costly or undesirable.

### Pros and Cons

**Pros:**
- Precision provides a specific measure of the model's accuracy in predicting positive instances.
- It focuses on minimizing false positives, making it suitable for applications where the cost of false positives is high.
- Useful for situations where the positive class is of higher importance or interest.

**Cons:**
- Precision does not take into account the instances that were incorrectly predicted as negative (false negatives).
- It may not provide a complete picture of the model's performance, especially when the goal is to minimize false negatives.
- Precision alone does not consider the true negatives and may not reflect the overall accuracy of the model.

Precision should be considered in conjunction with other evaluation metrics, such as recall or F1-score, to gain a comprehensive understanding of the model's performance in classification tasks.

### Code

```python
from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred)
print("Precision:", precision)
```

""", unsafe_allow_html=True)
                                
                        # Recall
                        with eval_tabs[3]:

                                st.markdown("""
        
                                        ## 󠁪󠁪󠁪󠁪󠁪📲 Recall

Recall is an evaluation metric for classification models that measures the proportion of correctly predicted positive instances out of the total actual positive instances. It focuses on the model's ability to correctly identify positive instances.

### Equation

Recall is calculated using the following equation:

""", unsafe_allow_html=True)

                                st.latex(r''' Recall = \frac{TP}{TP + FN} ''')

                                st.markdown("""

where:

- $TP$: True Positive
- $FN$: False Negative

<br>


### Usage and Interpretation

Recall is particularly useful when the goal is to minimize false negatives. It provides insights into the model's ability to capture positive instances and avoid false negatives.

A high recall value indicates that the model has a low rate of falsely predicting negative instances, making it valuable in scenarios where false negatives are costly or undesirable.

### Pros and Cons

**Pros:**
- Recall provides a specific measure of the model's ability to capture positive instances.
- It focuses on minimizing false negatives, making it suitable for applications where the cost of false negatives is high.
- Useful for situations where the positive class is of higher importance or interest.

**Cons:**
- Recall does not take into account the instances that were incorrectly predicted as positive (false positives).
- It may not provide a complete picture of the model's performance, especially when the goal is to minimize false positives.
- Recall alone does not consider the true negatives and may not reflect the overall accuracy of the model.

Recall should be considered in conjunction with other evaluation metrics, such as precision or F1-score, to gain a comprehensive understanding of the model's performance in classification tasks.

### Code

```python
from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred)
print("Recall:", recall)
```

""", unsafe_allow_html=True)

                        # F1 Score
                        with eval_tabs[4]:
                               
                                st.markdown("""

                                ## 󠁪󠁪󠁪󠁪󠁪⚾ F1-score

The F1-score is an evaluation metric for classification models that combines both precision and recall into a single measure. It provides a balance between precision and recall, making it a useful metric when both false positives and false negatives need to be minimized.

### Equation

The F1-score is calculated using the following equation:

""", unsafe_allow_html=True)
                                
                                st.latex(r''' F1-score = 2 \times \frac{Precision \times Recall}{Precision + Recall} ''')

                                st.markdown("""

where:

- $Precision$: Precision
- $Recall$: Recall

<br>


### Usage and Interpretation

The F1-score is particularly useful when there is an uneven class distribution or when false positives and false negatives have different impacts on the problem. It provides a single metric that considers both precision and recall, giving a balanced measure of the model's performance.

A high F1-score indicates that the model has both high precision and high recall, meaning it can effectively identify positive instances while minimizing false positives and false negatives.

### Pros and Cons

**Pros:**
- The F1-score provides a single metric that balances both precision and recall.
- It is useful in scenarios where there is an imbalance between classes or when false positives and false negatives have different consequences.
- The F1-score is a robust measure for evaluating model performance, especially in classification tasks.

**Cons:**
- The F1-score does not consider true negatives and may not reflect the overall accuracy of the model.
- It may not be the best choice when the relative importance of precision and recall varies based on the specific problem.

The F1-score should be considered alongside other evaluation metrics, such as accuracy, precision, and recall, to gain a comprehensive understanding of the model's performance in classification tasks.


### Code

```python
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)
print("F1-score:", f1)
```

""", unsafe_allow_html=True)
                                



















