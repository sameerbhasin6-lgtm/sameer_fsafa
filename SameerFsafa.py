import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from textstat import textstat
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import numpy as np
from pypdf import PdfReader
import re
from collections import Counter

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Forensic AI Commander", page_icon="🕵️‍♂️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #4e8cff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .high-risk { border-left: 5px solid #ff4b4b !important; background-color: #fff5f5 !important; }
    .good-metric { border-left: 5px solid #00cc96 !important; background-color: #f0fff4 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. ADVANCED EXTRACTION (Page-wise) ---
@st.cache_data
def extract_text_by_page(file, start_p=1, end_p=None):
    """Extracts text page-by-page to allow trend analysis."""
    pages_data = []
    try:
        reader = PdfReader(file)
        total_pages = len(reader.pages)
        if end_p is None or end_p > total_pages: end_p = total_pages
        
        for i in range(start_p - 1, end_p):
            text = reader.pages[i].extract_text()
            if text:
                pages_data.append({"page": i + 1, "text": text})
                
        return pages_data, total_pages
    except Exception as e:
        st.error(f"Error: {e}")
        return [], 0

# --- 2. ANALYTICS ENGINE ---
def analyze_page_metrics(pages_data):
    """Calculates Fog Index and Sentiment for EVERY page individually."""
    results = []
    full_text = ""
    
    for p in pages_data:
        txt = p['text']
        full_text += txt + " "
        
        # Metrics per page
        try: fog = textstat.gunning_fog(txt)
        except: fog = 0
        
        # Sentiment (-1 to +1)
        blob = TextBlob(txt)
        sentiment = blob.sentiment.polarity
        
        results.append({"Page": p['page'], "Fog Index": fog, "Sentiment": sentiment})
        
    return pd.DataFrame(results), full_text

def get_financial_entities(text):
    """Regex to find Money (Crores, Lakhs, Millions, etc.)."""
    # Matches: Rs. 500, $ 10.5 million, 50 crore, etc.
    pattern = r'(Rs\.|INR|\$|€|£)\s?\d+(?:,\d+)*(?:\.\d+)?\s?(?:million|billion|trillion|crore|lakh|cr|mn)?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return Counter(matches).most_common(100)

def perform_topic_modeling(text, n_topics=3):
    """Uses LDA to find hidden themes in the text."""
    # Remove boilerplate
    stopwords = list(STOPWORDS) + ['company', 'year', 'financial', 'notes', 'december', 'march', 'ended', 'amount', 'value']
    
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words=stopwords)
    dtm = vectorizer.fit_transform([text])
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    
    topics = {}
    feature_names = vectorizer.get_feature_names_out()
    for index, topic in enumerate(lda.components_):
        topics[f"Topic {index+1}"] = [feature_names[i] for i in topic.argsort()[-5:]]
        
    return topics

# --- SIDEBAR ---
with st.sidebar:
    st.title("🕵️‍♂️ Forensic Settings")
    uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type="pdf")
    
    st.markdown("---")
    st.caption("Scan Settings")
    use_all = st.checkbox("Scan Entire Document", value=False)
    start_p, end_p = 1, 50
    if not use_all:
        c1, c2 = st.columns(2)
        start_p = c1.number_input("Start Page", 1, value=40)
        end_p = c2.number_input("End Page", 1, value=80)
    else: end_p = None

# --- MAIN APP ---
if uploaded_file:
    with st.spinner("Running Multi-Layer Forensic Scan..."):
        # 1. Extraction
        pages_data, total_pgs = extract_text_by_page(uploaded_file, start_p, end_p)
        
        if pages_data:
            # 2. Analysis
            df_trends, full_text = analyze_page_metrics(pages_data)
            financials = get_financial_entities(full_text)
            
            # Global Metrics
            avg_fog = df_trends["Fog Index"].mean()
            avg_sent = df_trends["Sentiment"].mean()
            
            # --- DASHBOARD HEADER ---
            st.title("Forensic Commander Dashboard")
            st.caption(f"Analyzing {len(pages_data)} pages | Total Words: {len(full_text.split()):,}")
            
            # --- ROW 1: KEY METRICS ---
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                risk_label = "High Complexity" if avg_fog > 18 else "Readable"
                color = "high-risk" if avg_fog > 18 else "good-metric"
                st.markdown(f'<div class="metric-card {color}"><h3>Avg Fog Index</h3><h2>{avg_fog:.1f}</h2><p>{risk_label}</p></div>', unsafe_allow_html=True)
            
            with col2:
                # Sentiment Logic: Too positive (>0.2) in a neutral doc is suspicious
                sent_label = "Neutral"
                if avg_sent > 0.15: sent_label = "Overly Positive (Salesy)"
                elif avg_sent < -0.05: sent_label = "Negative Tone"
                st.markdown(f'<div class="metric-card"><h3>Sentiment Score</h3><h2>{avg_sent:.2f}</h2><p>{sent_label}</p></div>', unsafe_allow_html=True)
                
            with col3:
                # Quantify Money mentions
                st.markdown(f'<div class="metric-card"><h3>Financial Figures</h3><h2>{len(financials)}</h2><p>Unique Amounts Found</p></div>', unsafe_allow_html=True)
            
            with col4:
                # Complex Words Count
                complex_words = [w for w in full_text.split() if textstat.syllable_count(w) >= 3]
                st.markdown(f'<div class="metric-card"><h3>Complex Words</h3><h2>{len(complex_words)}</h2><p>Total Count</p></div>', unsafe_allow_html=True)

            st.markdown("---")

            # --- ROW 2: THE "HEARTBEAT" CHART (New Feature) ---
            st.subheader("📈 The 'Risk Heartbeat': Complexity Flow")
            st.write("This chart tracks the Fog Index per page. **Spikes indicate pages where language suddenly becomes dense (hiding something).**")
            
            fig_trend = px.line(df_trends, x="Page", y="Fog Index", title="Readability Complexity by Page", markers=True)
            fig_trend.add_hline(y=18, line_dash="dash", line_color="red", annotation_text="Danger Zone (>18)")
            fig_trend.update_layout(hovermode="x unified")
            st.plotly_chart(fig_trend, use_container_width=True)

            # --- ROW 3: DEEP DIVE TABS ---
            tab1, tab2, tab3 = st.tabs(["💰 Money Trail", "🧠 Topic Modeling", "☁️ Complex Cloud"])
            
            with tab1:
                st.subheader("Extracted Financial Values")
                st.write("The algorithm scanned for currency patterns (Rs, $, Cr, Mn).")
                if financials:
                    df_fin = pd.DataFrame(financials, columns=["Amount Pattern", "Frequency"])
                    st.dataframe(df_fin, use_container_width=True, height=300)
                else:
                    st.info("No standard currency formats found.")

            with tab2:
                st.subheader("Hidden Themes (Unsupervised Learning)")
                st.write("Using Latent Dirichlet Allocation (LDA) to group words into 3 main topics.")
                try:
                    topics = perform_topic_modeling(full_text)
                    c1, c2, c3 = st.columns(3)
                    for i, (topic, words) in enumerate(topics.items()):
                        with [c1, c2, c3][i]:
                            st.info(f"**{topic}**")
                            st.write(", ".join(words))
                except:
                    st.warning("Not enough text to perform Topic Modeling.")

            with tab3:
                st.subheader("Complex Word Cloud")
                if complex_words:
                    wc = WordCloud(background_color="white", colormap="Reds", height=300).generate(" ".join(complex_words))
                    fig_wc, ax = plt.subplots()
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig_wc)

            # --- ROW 4: EXPORT REPORT ---
            st.markdown("---")
            csv = df_trends.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Forensic Report (CSV)", data=csv, file_name="forensic_audit_report.csv", mime="text/csv")

else:
    st.info("Please upload a file to activate the Commander Dashboard.")
