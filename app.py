import streamlit as st
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Fake News Detector", layout="wide")

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# Initialize session state for page tracking
if "page" not in st.session_state:
    st.session_state.page = "home"

# Function to switch page
def switch_to_input():
    st.session_state.page = "input"

# Custom CSS for styling
st.markdown("""
    <style>
        /* Set background to gray */
        [data-testid="stAppViewContainer"] {
            background-color: #f0f0f0; 
        }
        [data-testid="stSidebar"] {
            background-color: #d9d9d9;
        }
        /* Style buttons */
        div.stButton > button:first-child {
            background-color: #28A745; /* Green for Predict */
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 8px;
        }
        div.stButton > button:last-child {
            background-color: #007BFF !important; /* Blue for Check News */
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

# Display Home Page
if st.session_state.page == "home":
    col1, col2 = st.columns([2, 1])  # Layout for text & image
    
    with col1:
        st.title("Fake News Detection")
        st.write("""Fake news is a false information presented as legitimate news which is often with the intent to manipulate public opinion or generate engagement for political benefit. It can spread rapidly across social media and online platforms for making difficult for people to distinguish between real and fake content. This platform help to distinguish between real and fake news and allow individuals to rely on verified and trustworthy sources. This platform assist to journalists and researchers in validating news articles quickly and efficiently.""")

        if st.button("Predict", key="predict_btn"):
            switch_to_input()

    with col2:
        st.image("1.png", use_column_width=True)

# Display Input Page
elif st.session_state.page == "input":
    st.title("Fake News Detection")
    st.markdown("<h3 style='text-align: center; color: blue;'>Enter your Whole news</h3>", unsafe_allow_html=True)
    
    # Display Word Limit Quote
    st.markdown("<p class='word-limit'>Maximum 500 words</p>", unsafe_allow_html=True)
    inputn = st.text_area("", key="inputn", placeholder="Type or paste your news article here...")

    # Blue Check News Button
    if st.button("Check News", key="check_news_btn"):
        if inputn.strip():
            transform_input = vectorizer.transform([inputn])
            prediction = model.predict(transform_input)
            confidence = model.predict_proba(transform_input).max() * 100

            if prediction[0] == 1:
                st.success(f"The News is **Real!** ({confidence:.2f}% confidence)")
            else:
                st.error(f" The News is **Fake!** ({confidence:.2f}% confidence)")
        else:
            st.warning("⚠ Please enter some text to analyze.")