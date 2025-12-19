import streamlit as st
import pandas as pd
import plotly.express as px
import random
from textstat import textstat

# --- PAGE CONFIGURATION (UI/UX) ---
st.set_page_config(
    page_title="Forensic Auditor AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Beautiful" UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .red-flag {
        background-color: #ffe6e6;
        padding: 5px;
        border-radius: 5px;
        border-left: 5px solid #ff4b4b;
        color: #b30000;
        font-weight: 500;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.title("🔍 Forensic AI")
    st.write("Upload Annual Reports to detect 'Creative Accounting' patterns.")
    
    uploaded_file = st.file_uploader("Upload 10-K / Annual Report (PDF)", type="pdf")
    
    st.markdown("---")
    st.header("⚙️ Sensitivity Analysis")
    fog_threshold = st.slider("Fog Index Threshold", 10, 25, 18, help="Texts above this score are considered 'Hard to Read'.")
    similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.75, help="Flag notes that changed more than this % from last year.")
    
    st.markdown("---")
    st.caption("Developed for Financial Statement Analysis Project")

# --- MOCK DATA GENERATOR (Simulating your backend) ---
def analyze_text(text):
    """
    In a real app, this would call your NLP pipeline.
    Here, we simulate results for demonstration.
    """
    fog_score = random.uniform(12, 22)  # Random score for demo
    uncertainty_words = random.randint(5, 50)
    
    # Fake extracted snippets
    suspicious_snippets = [
        "The company estimates that potential liabilities could be material, but cannot be reasonably estimated at this time.",
        "Revenue recognition policies were adjusted to better reflect the economic reality of long-term contracts.",
        "Certain off-balance sheet arrangements have been utilized to manage liquidity needs."
    ]
    
    return fog_score, uncertainty_words, suspicious_snippets

# --- MAIN DASHBOARD ---
if uploaded_file is None:
    # LANDING PAGE STATE
    st.header("👋 Welcome to the Forensic Auditor Dashboard")
    st.markdown("### How to use this tool:")
    st.markdown("""
    1. **Upload** a PDF Annual Report in the sidebar.
    2. **Wait** for the AI to extract and analyze 'Notes to Accounts'.
    3. **Review** the Risk Scoreometer and highlighted Red Flags.
    """)
    
    # Demo Chart (Landing Page Visual)
    st.markdown("### 📈 Recent Scans")
    demo_df = pd.DataFrame({
        "Company": ["Enron Corp", "Wirecard", "Satyam", "Apple", "Microsoft"],
        "Risk Score": [95, 88, 92, 12, 15],
        "Status": ["Critical", "Critical", "Critical", "Safe", "Safe"]
    })
    fig = px.bar(demo_df, x="Company", y="Risk Score", color="Status", 
                 color_discrete_map={"Critical": "#ff4b4b", "Safe": "#00cc96"})
    st.plotly_chart(fig, use_container_width=True)

else:
    # RESULTS DASHBOARD STATE
    with st.spinner('Extracting text and analyzing linguistic patterns...'):
        # In real life, you would pass the file to your PDF extractor here
        # raw_text = extract_text(uploaded_file)
        
        # Simulating analysis
        fog_score, uncertainty_count, snippets = analyze_text("dummy text")
        
        # Logic to determine Risk Level
        risk_score = min(100, (fog_score * 2) + (uncertainty_count * 0.5))
        risk_color = "normal" if risk_score < 50 else "off" # Streamlit metric delta color logic
    
    st.title("📊 Analysis Report: " + uploaded_file.name)
    
    # 1. KPI ROW
    col1, col2, col3 = st.columns(3)
    col1.metric("Gunning Fog Index", f"{fog_score:.1f}", f"{fog_score - fog_threshold:.1f} vs Limit", delta_color="inverse")
    col2.metric("Uncertainty Words", uncertainty_count, "High Frequency")
    col3.metric("Overall Risk Score", f"{risk_score:.0f}/100", "CRITICAL" if risk_score > 70 else "SAFE")
    
    st.markdown("---")

    # 2. TABS FOR DEEP DIVE
    tab1, tab2, tab3 = st.tabs(["🚩 Red Flags", "📉 Trend Analysis", "📝 Raw Data"])
    
    with tab1:
        st.subheader("Detected Linguistic Anomalies")
        st.write("The following sentences in 'Notes to Accounts' were flagged for high complexity or hedging:")
        
        for snippet in snippets:
            st.markdown(f'<div class="red-flag">⚠️ "{snippet}"</div>', unsafe_allow_html=True)
            st.caption(f"Reason: High hedging frequency • Readability Grade: {random.randint(18,25)}")

    with tab2:
        st.subheader("Year-over-Year Linguistic Shift")
        # Mocking time-series data
        years = [2020, 2021, 2022, 2023, 2024]
        fog_trend = [14, 15, 14.5, 19.2, 21.5] # Sudden jump in complexity
        
        trend_df = pd.DataFrame({"Year": years, "Fog Index": fog_trend})
        
        line_fig = px.line(trend_df, x="Year", y="Fog Index", markers=True, title="Complexity Spike Detected in 2023")
        line_fig.add_hline(y=fog_threshold, line_dash="dash", line_color="red", annotation_text="Risk Threshold")
        st.plotly_chart(line_fig, use_container_width=True)
        
        st.info("💡 **Insight:** A sudden spike in the Fog Index (as seen in 2023) often correlates with periods of poor financial performance that management is trying to obscure.")

    with tab3:
        st.text_area("Extracted Text Content (Preview)", "This is where the raw text from the PDF would appear for manual review...", height=300)