import streamlit as st
import pickle
import re
import string
import matplotlib.pyplot as plt
import numpy as np

# Page Config — Dark Theme!

st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS — Pro Look!

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stButton>button {
        background: linear-gradient(90deg, #FF6B6B, #FF8E53);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #FF8E53, #FF6B6B);
        transform: scale(1.02);
    }
    .result-fake {
        background: linear-gradient(135deg, #FF6B6B, #FF4444);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .result-real {
        background: linear-gradient(135deg, #56CCF2, #2F80ED);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-card {
        background: #1E2130;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Models Load

@st.cache_resource
def load_models():
    with open("../models/fake_news_model.pkl", 'rb') as f:
        model = pickle.load(f)
    with open("../models/tfidf_vectorizer.pkl", 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_models()


# Text Cleaning

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Sidebar

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/news.png", width=80)
    st.title("🔍 Fake News Detector")
    st.markdown("---")
    st.markdown("""
    ### How it works?
    1. 📝 Paste news article
    2. 🤖 AI analyzes text
    3. ✅ Get instant result
    
    ### Model Info
    - **Algorithm:** XGBoost
    - **Accuracy:** 99.75%
    - **Features:** TF-IDF (10K)
    - **Training Data:** 44K articles
    """)
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.info("Always verify news from trusted sources like BBC, Reuters, AP News.")


# Main Header

st.markdown("""
<h1 style='text-align: center; 
           background: linear-gradient(90deg, #FF6B6B, #56CCF2);
           -webkit-background-clip: text;
           -webkit-text-fill-color: transparent;
           font-size: 48px;'>
    🔍 Fake News Detector AI
</h1>
<p style='text-align: center; color: #888; font-size: 18px;'>
    Powered by XGBoost + TF-IDF | 99.75% Accuracy
</p>
""", unsafe_allow_html=True)

st.markdown("---")


# Input Section

col1, col2 = st.columns([2, 1])

with col1:
    news_text = st.text_area(
        "📰 Paste News Article Here:",
        height=300,
        placeholder="Paste the complete news article here...\n\nTip: Longer articles give more accurate results!"
    )
    
    # Character counter
    char_count = len(news_text)
    word_count = len(news_text.split()) if news_text else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Characters", char_count)
    c2.metric("Words", word_count)
    c3.metric("Min Words Needed", "20")

with col2:
    st.markdown("### 📊 Quick Stats")
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #56CCF2;'>44,898</h3>
        <p style='color: #888;'>Training Articles</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #FF6B6B;'>99.75%</h3>
        <p style='color: #888;'>Model Accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #F7971E;'>10,000</h3>
        <p style='color: #888;'>TF-IDF Features</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# Predict Button

if st.button("🔍 ANALYZE NEWS", use_container_width=True):
    
    if word_count < 20:
        st.warning("⚠️ Please enter at least 20 words for accurate prediction!")
    
    else:
        with st.spinner("🤖 AI is analyzing the article..."):
            
            # Clean + Transform
            cleaned = clean_text(news_text)
            vectorized = tfidf.transform([cleaned])
            
            # Predict
            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0]
            fake_prob = probability[0] * 100
            real_prob = probability[1] * 100
            confidence = max(fake_prob, real_prob)

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        

        # Main Result
    
        col3, col4 = st.columns(2)
        
        with col3:
            if prediction == 0:
                st.markdown("""
                <div class='result-fake'>
                    ❌ FAKE NEWS DETECTED!
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='result-real'>
                    ✅ REAL NEWS
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
                 st.metric("Confidence", f"{confidence:.1f}%")
                 st.progress(float(confidence/100))
        
        
        # Probability Bars
    
        st.markdown("### 🎯 Probability Breakdown")
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown("**❌ Fake Probability**")
            st.progress(float(fake_prob/100))
            st.markdown(f"**{fake_prob:.1f}%**")
        
        with col6:
            st.markdown("**✅ Real Probability**")
            st.progress(float(real_prob/100))
            st.markdown(f"**{real_prob:.1f}%**")
        
        
        
        # Recommendations
        
        st.markdown("---")
        if prediction == 0:
            st.error("""
            ### ⚠️ Warning — Fake News Detected!
            - 🔍 Verify from BBC, Reuters, AP News
            - 📱 Don't share on social media
            - 🤔 Check the original source
            - 📅 Verify the date of the article
            - 👤 Check author credentials
            """)
        else:
            st.success("""
            ### ✅ Article appears credible!
            - Still verify from original source
            - 📰 Cross-check with other outlets
            - 🔗 Check source website reputation
            """)