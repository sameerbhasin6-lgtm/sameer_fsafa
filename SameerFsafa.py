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

# --- 1. OPTIMIZED TEXT EXTRACTION (FAST) ---
@st.cache_data
def extract_text_fast(file, start_page=1, end_page=None):
    """
    Extracts text using pypdf (Faster than pdfplumber).
    Includes a progress bar for user feedback.
    """
    text = ""
    try:
        reader = PdfReader(file)
        total_pages = len(reader.pages)
        
        # Determine valid page range
        if end_page is None or end_page > total_pages:
            end_page = total_pages
        
        # Progress bar setup
        progress_text = "Scanning document..."
        my_bar = st.progress(0, text=progress_text)
        
        # Extract Loop
        for i in range(start_page - 1, end_page):
            page = reader.pages[i]
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
            
            # Update progress bar
            percent_complete = int(((i - (start_page - 1)) / (end_page - (start_page - 1))) * 100)
            my_bar.progress(percent_complete, text=f"Scanning page {i+1} of {end_page}...")
            
        my_bar.empty() # Clear bar when done
        return text, total_pages
        
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return "", 0

# --- 2. METRICS CALCULATION ENGINE ---
def calculate_forensic_metrics(text):
    if not text:
        return None
        
    # A. READABILITY (Fog Index)
    try:
        fog_index = textstat.gunning_fog(text)
    except:
        fog_index = 0 # Fallback if text is empty
    
    # B. DICTIONARY ANALYSIS
    words = re.findall(r'\w+', text.lower())
    total_words = len(words) if len(words) > 0 else 1
    
    uncertainty_list = ['approximate', 'contingency', 'fluctuate', 'indefinite', 'uncertain', 'estimate', 'assuming', 'might', 'could', 'pending']
    litigious_list = ['claim', 'legal', 'proceeding', 'litigation', 'petition', 'damages', 'class action', 'liability', 'court']
    constraining_list = ['required', 'obligations', 'strict', 'prohibited', 'must', 'comply', 'restrict']
    
    u_count = sum(1 for w in words if w in uncertainty_list)
    l_count = sum(1 for w in words if w in litigious_list)
    c_count = sum(1 for w in words if w in constraining_list)
    
    # Normalize per 1,000 words
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

# --- SIDEBAR ---
with st.sidebar:
    st.title("📂 Data Input")
    st.info("Upload Annual Reports. Tip: Select page ranges to speed up analysis.")
    
    uploaded_file_curr = st.file_uploader("Current Year Report (PDF)", type="pdf")
    
    # PAGE SELECTOR (Crucial for performance)
    st.markdown("---")
    st.write("📄 **Page Scope**")
    use_all_pages = st.checkbox("Analyze Full Document?", value=False)
    
    start_p, end_p = 1, 50 # Defaults
    if not use_all_pages:
        st.caption("Select the 'Notes to Accounts' pages to avoid analyzing marketing images.")
        start_p = st.number_input("Start Page", min_value=1, value=50)
        end_p = st.number_input("End Page", min_value=1, value=100)
    else:
        end_p = None # Signal to read all
        
    st.markdown("---")
    fog_limit = st.slider("Max Acceptable Fog Index", 10, 25, 18)

# --- MAIN APP LOGIC ---
if uploaded_file_curr:
    
    # 1. PROCESSING
    with st.spinner("Processing... (This may take a moment)"):
        # Use the NEW fast extractor
        text_curr, total_pgs = extract_text_fast(uploaded_file_curr, start_p, end_p)
        
        if text_curr:
            metrics = calculate_forensic_metrics(text_curr)
            
            # 2. DASHBOARD HEADER
            st.title(f"📊 Forensic Analysis Report")
            pg_range_label = f"Pages {start_p}-{end_p}" if not use_all_pages else f"All {total_pgs} Pages"
            st.caption(f"Filename: {uploaded_file_curr.name} | Scanned: {pg_range_label} | Words: {metrics['total_words']:,}")
            
            # 3. METRIC CARDS
            col1, col2, col3 = st.columns(3)
            
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

            st.markdown("---")

            # 4. WORD CLOUD & REGRESSION
            col_cloud, col_reg = st.columns([1, 1])
            
            with col_cloud:
                st.subheader("☁️ Contextual Word Cloud")
                custom_stopwords = set(STOPWORDS)
                custom_stopwords.update(["Company", "Group", "Year", "Financial", "Statement", "Note", "December", "ended", "March", "crore", "lakhs"])
                
                wc = WordCloud(background_color="white", height=300, width=500, stopwords=custom_stopwords, colormap="Reds").generate(metrics['clean_text'])
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)

            with col_reg:
                st.subheader("📈 Regression Benchmark")
                # Dummy Industry Data
                np.random.seed(42)
                benchmark_fog = np.random.normal(16, 3, 50)
                benchmark_risk = (benchmark_fog * 1.5) + np.random.normal(0, 5, 50)
                df_bench = pd.DataFrame({"Fog Index": benchmark_fog, "Risk Score": benchmark_risk})
                
                # Regression
                X = df_bench[["Fog Index"]]
                y = df_bench["Risk Score"]
                model = LinearRegression()
                model.fit(X, y)
                current_pred = model.predict([[metrics['fog_index']]])[0]
                
                # Plot
                fig_reg = px.scatter(df_bench, x="Fog Index", y="Risk Score", opacity=0.4)
                line_x = np.linspace(df_bench["Fog Index"].min(), df_bench["Fog Index"].max(), 100).reshape(-1, 1)
                line_y = model.predict(line_x)
                fig_reg.add_traces(go.Scatter(x=line_x.flatten(), y=line_y, mode='lines', name='Industry Trend'))
                fig_reg.add_traces(go.Scatter(x=[metrics['fog_index']], y=[current_pred], mode='markers+text', marker=dict(color='red', size=15, symbol='x'), name='Your File', text=["YOU"], textposition="top center"))
                st.plotly_chart(fig_reg, use_container_width=True)
        else:
            st.warning("No text extracted. Try a different page range or file.")

else:
    st.subheader("👋 Welcome to the Forensic Lab")
    st.info("Upload a PDF to begin analysis.")
