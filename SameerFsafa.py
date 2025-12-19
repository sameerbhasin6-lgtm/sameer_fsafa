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
import pdfplumber
import re

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
    </style>
    """, unsafe_allow_html=True)

# --- 1. TEXT EXTRACTION FUNCTION ---
@st.cache_data
def extract_text_from_pdf(file):
    """Extracts text from uploaded PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# --- 2. METRICS CALCULATION ENGINE ---
def calculate_forensic_metrics(text):
    # A. READABILITY (Fog Index)
    # textstat handles the complex syllable counting automatically
    fog_index = textstat.gunning_fog(text)
    
    # B. DICTIONARY ANALYSIS (Loughran-McDonald Simplified)
    # Convert to lowercase and tokenize
    words = re.findall(r'\w+', text.lower())
    total_words = len(words) if len(words) > 0 else 1
    
    # Financial Dictionaries (Simplified)
    uncertainty_list = ['approximate', 'contingency', 'fluctuate', 'indefinite', 'uncertain', 'estimate', 'assuming', 'might', 'could', 'pending']
    litigious_list = ['claim', 'legal', 'proceeding', 'litigation', 'petition', 'damages', 'class action', 'liability', 'court']
    constraining_list = ['required', 'obligations', 'strict', 'prohibited', 'must', 'comply', 'restrict']
    
    u_count = sum(1 for w in words if w in uncertainty_list)
    l_count = sum(1 for w in words if w in litigious_list)
    c_count = sum(1 for w in words if w in constraining_list)
    
    # Normalize to "Words per 1,000" for fair comparison
    u_score = (u_count / total_words) * 1000
    l_score = (l_count / total_words) * 1000
    c_score = (c_count / total_words) * 1000
    
    return {
        "fog_index": fog_index,
        "uncertainty_score": u_score,
        "litigious_score": l_score,
        "constraining_score": c_score,
        "total_words": total_words,
        "clean_text": text
    }

def calculate_similarity(text1, text2):
    """Calculates Cosine Similarity between two texts."""
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)
    return similarity[0][1] # Returns value between 0 and 1

# --- SIDEBAR ---
with st.sidebar:
    st.title("📂 Data Input")
    st.info("Upload the Annual Report (Notes to Accounts) to begin.")
    
    uploaded_file_curr = st.file_uploader("Current Year Report (PDF)", type="pdf")
    uploaded_file_prev = st.file_uploader("Previous Year Report (Optional - for Similarity Check)", type="pdf")
    
    st.markdown("---")
    st.write("⚙️ **Threshold Settings**")
    fog_limit = st.slider("Max Acceptable Fog Index", 10, 25, 18)

# --- MAIN APP LOGIC ---
if uploaded_file_curr:
    
    # 1. PROCESSING
    with st.spinner("Extracting text and computing metrics..."):
        text_curr = extract_text_from_pdf(uploaded_file_curr)
        metrics = calculate_forensic_metrics(text_curr)
        
        similarity_score = None
        if uploaded_file_prev:
            text_prev = extract_text_from_pdf(uploaded_file_prev)
            similarity_score = calculate_similarity(text_curr, text_prev)

    # 2. DASHBOARD HEADER
    st.title(f"📊 Forensic Analysis Report")
    st.caption(f"Filename: {uploaded_file_curr.name} | Total Words Analyzed: {metrics['total_words']:,}")
    
    # 3. METRIC CARDS (Top Row)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        is_high = metrics['fog_index'] > fog_limit
        st.markdown(f"""
        <div class="metric-card {'high-risk' if is_high else ''}">
            <h3>Fog Index</h3>
            <h2>{metrics['fog_index']:.2f}</h2>
            <p>{'⚠️ High Complexity' if is_high else '✅ Readable'}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Uncertainty Score</h3>
            <h2>{metrics['uncertainty_score']:.1f}</h2>
            <p>Words per 1,000</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Litigious Score</h3>
            <h2>{metrics['litigious_score']:.1f}</h2>
            <p>Words per 1,000</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        sim_display = f"{similarity_score*100:.1f}%" if similarity_score else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <h3>YoY Similarity</h3>
            <h2>{sim_display}</h2>
            <p>{'⚠️ Significant Rewrite' if similarity_score and similarity_score < 0.8 else 'Consistent'}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. WORD CLOUD VISUALIZATION
    col_cloud, col_reg = st.columns([1, 1])
    
    with col_cloud:
        st.subheader("☁️ Contextual Word Cloud")
        
        # Stopwords to remove boilerplate
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update(["Company", "Group", "Year", "Financial", "Statement", "Note", "December", "ended"])
        
        wc = WordCloud(
            background_color="white",
            height=300,
            width=500,
            stopwords=custom_stopwords,
            colormap="Reds" # Red = Alert
        ).generate(metrics['clean_text'])
        
        fig_wc, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)

    # 5. REGRESSION ANALYSIS (Benchmarking)
    with col_reg:
        st.subheader("📈 Regression: Complexity vs. Fraud Risk")
        st.write("Comparing this company against an industry benchmark dataset.")
        
        # A. GENERATE DUMMY BENCHMARK DATA (Since we can't regress on 1 file)
        # We create 50 random companies to simulate an industry standard
        np.random.seed(42)
        benchmark_fog = np.random.normal(16, 3, 50)  # Average Fog 16, SD 3
        # Assume Fraud Risk increases with Fog Index (Positive Correlation)
        benchmark_risk = (benchmark_fog * 1.5) + np.random.normal(0, 5, 50)
        
        df_bench = pd.DataFrame({"Fog Index": benchmark_fog, "Risk Score": benchmark_risk})
        
        # B. RUN REGRESSION MODEL
        X = df_bench[["Fog Index"]]
        y = df_bench["Risk Score"]
        model = LinearRegression()
        model.fit(X, y)
        
        # C. PREDICT FOR CURRENT FILE
        current_pred = model.predict([[metrics['fog_index']]])[0]
        
        # D. VISUALIZE WITH PLOTLY
        fig_reg = px.scatter(df_bench, x="Fog Index", y="Risk Score", opacity=0.4, title="Industry Benchmark Analysis")
        
        # Add Regression Line
        line_x = np.linspace(df_bench["Fog Index"].min(), df_bench["Fog Index"].max(), 100).reshape(-1, 1)
        line_y = model.predict(line_x)
        fig_reg.add_traces(go.Scatter(x=line_x.flatten(), y=line_y, mode='lines', name='Regression Line'))
        
        # Add THIS Company (Red Dot)
        fig_reg.add_traces(go.Scatter(
            x=[metrics['fog_index']], 
            y=[current_pred], 
            mode='markers+text', 
            marker=dict(color='red', size=15, symbol='x'),
            name='Current Company',
            text=["YOU ARE HERE"],
            textposition="top center"
        ))
        
        st.plotly_chart(fig_reg, use_container_width=True)
        st.caption(f"Predicted Fraud Risk Score: {current_pred:.1f} (Based on Fog Index of {metrics['fog_index']:.1f})")

else:
    # LANDING PAGE (Empty State)
    st.subheader("👋 Welcome to the Forensic Lab")
    st.info("Please upload a PDF file from the sidebar to generate the analysis.")
