import streamlit as st
import requests

def load_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff;
    }
    .main-title {
        color: #1e90ff;
        font-size: 72px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .tagline {
        color: #4169e1;
        font-size: 28px;
        font-style: italic;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-header {
        color: #1e90ff;
        font-size: 36px;
        font-weight: bold;
        margin-top: 40px;
        margin-bottom: 20px;
        text-align: center;
    }
    .feature-box {
        background-color: #e6f2ff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.3s ease-in-out;
    }
    .feature-box:hover {
        transform: scale(1.05);
    }
    .feature-title {
        color: #4169e1;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .app-button {
        background-color: white;
        color: white;
        font-size: 12px;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 25px;
        text-align: center;
        margin: 5px;
        display: inline-block;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    .app-button:hover {
        background-color: #FFD35A;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .problem-statement {
        background-color: #ffd700;
        border-radius: 10px;
        padding: 20px;
        margin: 30px 0;
        text-align: center;
        font-size: 20px;
        color: #333;
    }
    .circular-image {
        border-radius: 50%;
        overflow: hidden;
        width: 120px;  /* Adjust the size as needed */
        height: 120px; /* Adjust the size as needed */
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px solid #1e90ff; /* Optional border */
    }
    .circular-image img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    </style>
    """, unsafe_allow_html=True)

def new_line():
    st.markdown("<br>", unsafe_allow_html=True)

# Define a function to load the Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Create the About Us page
def about_page():
    load_css()  # Load custom CSS
    
    # Title Page
    st.markdown("<h1 class='main-title'> 🔎 About Us - Data Vue</h1>", unsafe_allow_html=True)
    new_line()

    # About the Project
    st.markdown("""
    <div class='tagline'>Welcome to Data Vue!</div>
    <p>Welcome to Data Vue! This application is designed to provide a comprehensive platform for data analysis and machine learning.
    Whether you're a beginner or an experienced data scientist, our tool is tailored to simplify your workflow.</p>
    """, unsafe_allow_html=True)
    new_line()

    st.markdown("<h2 class='section-header'>👤 Meet the Team</h2>", unsafe_allow_html=True)

    # Define the team members with online image URLs
    team_members = [
        {
            "name": "M Jawad",
            "role": "Sir",
            "image": "https://media.licdn.com/dms/image/v2/D4D03AQGhdbU8hITDEA/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1709276424296?e=1730332800&v=beta&t=3RwehLr_ZcXAVxe2EZKHN5oaDPBxlaa_Jr9LjrSgFdo",  # Replace with the actual URL
            "linkedin": "https://www.linkedin.com/in/muhammad-jawad-86507b201/",
            "github": "https://github.com/mj-awad17",
        },
        {
            "name": "M Ibrahim",
            "role": "",
            "image": "https://media.licdn.com/dms/image/v2/D4D03AQFSX9z8C2gRTg/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1722410662074?e=1730332800&v=beta&t=meNgmw_h6m4cv_FzEP3fOZI-tjdaUGXcNPimiYMfDnQ",  # Replace with the actual URL
            "linkedin": "https://www.linkedin.com/in/muhammad-ibrahim-qasmi-9876a1297/",
            "github": "https://github.com/muhammadibrahim313",
        },
        {
            "name": "Alisha Ashraf",
            "role": "",
            "image": "https://media.licdn.com/dms/image/v2/D4E03AQE8YF_XiyirPQ/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1720500608337?e=1730332800&v=beta&t=cXDZ22_tBS_pFcyedLvNvnm5jvkqtmzPri2KNudD6J4",  # Replace with the actual URL
            "linkedin": "https://www.linkedin.com/in/alisha-ashraf-b73404301",
            "github": "https://github.com/AlishaAshraf",
        },
        {
            "name": "Phool Fatima",
            "role": "",
            "image": "https://media.licdn.com/dms/image/v2/C4D03AQH_YB-K8letfQ/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1658415696189?e=1730332800&v=beta&t=RHPdqVHpH5QpJ31HduZcqPs6kztEauBP3vX0VqLxF2w",  # Replace with the actual URL
            "linkedin": "https://www.linkedin.com/in/phool-fatima-602a7a23b/",
            "github": "https://github.com/fatima5655",
        },
        {
            "name": "Ahmad Fakhar",
            "role": "",
            "image": "https://avatars.githubusercontent.com/u/155258276?v=4",  # Replace with the actual URL
            "linkedin": "https://www.linkedin.com/in/ahmad-fakhar-357742258/",
            "github": "https://github.com/Ahmad-Fakhar",
        }
    ]

    # Create columns for each team member
    cols = st.columns(len(team_members))

    # Populate each column with a team member's details
    for col, member in zip(cols, team_members):
        with col:
            # Display the image using st.image with customized styling
            st.markdown(f"""
                <div class='circular-image'>
                    <img src='{member["image"]}' />
                </div>
                <div class='feature-box'>
                    <strong>{member['name']}</strong><br>
                    <em>{member['role']}</em><br>
                    <a href="{member['linkedin']}" target="_blank" class="app-button">LinkedIn</a> | 
                    <a href="{member['github']}" target="_blank" class="app-button">GitHub</a>
                </div>
                """, unsafe_allow_html=True
            )

    new_line()

    # What this app does with the main, quickml, and study_time pages
    st.markdown("<h2 class='section-header'>What This App Does</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='feature-box'>
        <h3>👉 Click DataVue</h3>
        <p>This section is the main page of the *Click DataVue* web app. It provides the customizability to build Machine Learning models by selecting and applying the Data Preparation techniques that fit your data. Also, you can try different Machine Learning models and tune the hyperparameters to get the best model.</p>
    </div>
    <div class='feature-box'>
        <h3>🚀 Quick DataVue</h3>
        <p>Data Vue is a tab that allows you to build a model quickly with just a few clicks. This tab is designed for people who are new to Machine Learning and want to build a model quickly without having to go through the entire process of Exploratory Data Analysis, Data Cleaning, Feature Engineering, etc. It is just a quick way to build a model for testing purposes.</p>
    </div>
    <div class='feature-box'>
        <h3>📚 Study Data Vue</h3>
        <p>The StudyML tab is designed to help you understand the key concepts of building machine learning models. This tab has 7 sections, each section talking about a specific concept in building machine learning models. With each section, you will have the ability to apply the concepts of these sections on multiple datasets. The code, the explanation, and everything you need to understand is in this tab.</p>
    </div>
    """, unsafe_allow_html=True)
    new_line()

    # Why Data Vue?
    st.markdown("<h2 class='section-header'>✨ Why Choose DataVue?</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='feature-box'>
        <ul>
            <li><strong>User-Friendly Interface:</strong> Data Vue offers an intuitive and easy-to-use interface, making machine learning accessible to users of all skill levels.</li>
            <li><strong>Efficiency and Speed:</strong> With Data Vue, you can quickly build, train, and evaluate machine learning models, reducing the time and effort required.</li>
            <li><strong>Comprehensive Learning Resources:</strong> The StudyML tab provides detailed explanations, code examples, and visualizations to enhance your understanding of machine learning concepts.</li>
            <li><strong>Flexible and Customizable:</strong> Data Vue supports a wide range of algorithms and allows you to fine-tune model parameters to meet your specific requirements.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    new_line()

    # Project Information
    st.markdown("<h2 class='section-header'>✨ Project Information</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='feature-box'>
        Data Vue is continuously evolving. We welcome contributions and feedback from the community.
        If you're interested in collaborating or have suggestions, feel free to reach out to us.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    about_page()
