import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from textstat import textstat
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import numpy as np
from pypdf import PdfReader
import re
from collections import Counter

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Forensic AI Auditor",
    page_icon="⚖️",
    layout="wide"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4e8cff;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .high-risk {
        border-left: 5px solid #ff4b4b !important;
        background-color: #fff1f1 !important;
    }
    .stDataFrame { border: 1px solid #e0e0e0; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. OPTIMIZED TEXT EXTRACTION ---
@st.cache_data
def extract_text_fast(file, start_page=1, end_page=None):
    """Extracts text using pypdf (Fast)."""
    text = ""
    try:
        reader = PdfReader(file)
        total_pages = len(reader.pages)
        if end_page is None or end_page > total_pages:
            end_page = total_pages
        
        # Simple extraction loop
        for i in range(start_page - 1, end_page):
            text += reader.pages[i].extract_text() + "\n"
            
        return text, total_pages
    except Exception as e:
        return "", 0

# --- 2. METRICS & COMPLEXITY ENGINE ---
def calculate_forensic_metrics(text):
    if not text: return None
    
    # A. Readability
    try: fog_index = textstat.gunning_fog(text)
    except: fog_index = 0
    
    # B. Tokenization & Syllable Counting
    words = re.findall(r'\w+', text.lower())
    total_words = len(words) if words else 1
    
    # Identify Complex Words (>= 3 syllables)
    # Optimization: Check unique words first, then map counts
    unique_words = set(words)
    complex_map = {w: textstat.syllable_count(w) for w in unique_words}
    complex_words_list = [w for w in words if complex_map.get(w, 0) >= 3]
    
    # C. Passive Voice Detection (Heuristic)
    # Pattern: "was/were/is/are" + verb ending in "ed"
    passive_matches = re.findall(r'\b(am|are|is|was|were|be|been|being)\b\s+\w+ed\b', text.lower())
    passive_count = len(passive_matches)
    
    # D. Dictionary Analysis
    uncertainty_list = ['approximate', 'contingency', 'fluctuate', 'indefinite', 'uncertain', 'estimate', 'assuming', 'might', 'could', 'pending']
    litigious_list = ['claim', 'legal', 'proceeding', 'litigation', 'petition', 'damages', 'class action', 'liability', 'court']
    
    u_count = sum(1 for w in words if w in uncertainty_list)
    l_count = sum(1 for w in words if w in litigious_list)

    return {
        "fog_index": fog_index,
        "uncertainty_score": (u_count / total_words) * 1000,
        "litigious_score": (l_count / total_words) * 1000,
        "passive_score": (passive_count / total_words) * 1000,
        "total_words": total_words,
        "complex_words": complex_words_list, # List of actual words
        "clean_text": text
    }

def calculate_similarity(text1, text2):
    """Cosine Similarity between two texts."""
    # Truncate to first 5000 words to speed up comparison if files are huge
    t1 = " ".join(text1.split()[:5000])
    t2 = " ".join(text2.split()[:5000])
    vectorizer = CountVectorizer().fit_transform([t1, t2])
    return cosine_similarity(vectorizer.toarray())[0][1]

# --- SIDEBAR ---
with st.sidebar:
    st.title("📂 Forensic Input")
    uploaded_file_curr = st.file_uploader("Current Year Report (PDF)", type="pdf")
    uploaded_file_prev = st.file_uploader("Previous Year Report (For Similarity)", type="pdf")
    
    st.markdown("---")
    st.write("⚙️ **Settings**")
    use_all_pages = st.checkbox("Analyze Full Document?", value=False)
    
    start_p, end_p = 1, 50
    if not use_all_pages:
        col_s, col_e = st.columns(2)
        start_p = col_s.number_input("Start Page", 1, value=50)
        end_p = col_e.number_input("End Page", 1, value=100)
    else: end_p = None

    fog_limit = st.slider("Fog Threshold", 10, 25, 18)

# --- MAIN APP ---
if uploaded_file_curr:
    with st.spinner("Analyzing linguistic patterns..."):
        # 1. Process Current File
        text_curr, _ = extract_text_fast(uploaded_file_curr, start_p, end_p)
        metrics = calculate_forensic_metrics(text_curr)
        
        # 2. Process Previous File (If uploaded)
        similarity_score = None
        if uploaded_file_prev:
            text_prev, _ = extract_text_fast(uploaded_file_prev, start_p, end_p)
            if text_prev:
                similarity_score = calculate_similarity(text_curr, text_prev)

        # --- DASHBOARD UI ---
        st.title("📊 Forensic Analysis Dashboard")
        
        # ROW 1: METRICS
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            is_high = metrics['fog_index'] > fog_limit
            st.markdown(f'<div class="metric-card {"high-risk" if is_high else ""}"><h3>Fog Index</h3><h2>{metrics["fog_index"]:.1f}</h2><p>Complexity</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><h3>Passive Voice</h3><h2>{metrics["passive_score"]:.1f}</h2><p>Per 1k Words</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><h3>Uncertainty</h3><h2>{metrics["uncertainty_score"]:.1f}</h2><p>Per 1k Words</p></div>', unsafe_allow_html=True)
        with c4:
            sim_val = f"{similarity_score*100:.1f}%" if similarity_score else "N/A"
            color_class = "high-risk" if (similarity_score and similarity_score < 0.8) else ""
            st.markdown(f'<div class="metric-card {color_class}"><h3>YoY Similarity</h3><h2>{sim_val}</h2><p>Consistency</p></div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # ROW 2: COMPLEX WORD CLOUD & TABLE
        st.subheader("☁️ Obfuscation Analysis: Complex Words Only")
        st.caption("Visualizing words with 3+ syllables that contribute to the Fog Index.")
        
        col_cloud, col_table = st.columns([2, 1])
        
        with col_cloud:
            # Generate cloud ONLY from complex words list
            if metrics['complex_words']:
                complex_text = " ".join(metrics['complex_words'])
                wc = WordCloud(background_color="white", height=350, width=600, colormap="Reds", stopwords=STOPWORDS).generate(complex_text)
                
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("No complex words found.")

        with col_table:
            # Create a frequency table
            if metrics['complex_words']:
                counts = Counter(metrics['complex_words'])
                df_complex = pd.DataFrame(counts.most_common(20), columns=["Word", "Count"])
                st.write("**Top Complex Words**")
                st.dataframe(df_complex, height=300, use_container_width=True)

        # ROW 3: REGRESSION BENCHMARK
        st.markdown("---")
        st.subheader("📈 Fraud Risk Regression Model")
        
        # Benchmark logic
        np.random.seed(42)
        bench_fog = np.random.normal(16, 3, 50)
        bench_risk = (bench_fog * 1.5) + np.random.normal(0, 5, 50)
        df_bench = pd.DataFrame({"Fog Index": bench_fog, "Risk Score": bench_risk})
        
        model = LinearRegression()
        model.fit(df_bench[["Fog Index"]], df_bench["Risk Score"])
        pred_risk = model.predict([[metrics['fog_index']]])[0]
        
        fig_reg = px.scatter(df_bench, x="Fog Index", y="Risk Score", opacity=0.3, title="Industry Benchmark (Simulated)")
        line_x = np.linspace(df_bench["Fog Index"].min(), df_bench["Fog Index"].max(), 100).reshape(-1, 1)
        fig_reg.add_traces(go.Scatter(x=line_x.flatten(), y=model.predict(line_x), mode='lines', name='Trend Line'))
        fig_reg.add_traces(go.Scatter(x=[metrics['fog_index']], y=[pred_risk], mode='markers', marker=dict(color='red', size=15), name='Your File'))
        
        st.plotly_chart(fig_reg, use_container_width=True)

else:
    st.info("Upload a PDF to start.")
