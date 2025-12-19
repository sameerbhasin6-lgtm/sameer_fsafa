import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from textstat import textstat
import re

# ... [Keep your existing Page Config and CSS] ...

# --- HELPER FUNCTION: CLEAN & QUANTIFY ---
def analyze_quantitative_metrics(text):
    # 1. Calculate Real Fog Index
    fog_score = textstat.gunning_fog(text)
    
    # 2. Uncertainty Dictionary (A mini version of Loughran-McDonald)
    uncertainty_lexicon = ['approximate', 'contingency', 'dependent', 'fluctuate', 'indefinite', 'unclear', 'estimate', 'assuming', 'potential']
    litigious_lexicon = ['claim', 'legal', 'proceeding', 'litigation', 'petition', 'damages', 'class action']
    
    # Count occurrences
    word_list = re.findall(r'\w+', text.lower())
    total_words = len(word_list) if len(word_list) > 0 else 1
    
    u_count = sum(1 for w in word_list if w in uncertainty_lexicon)
    l_count = sum(1 for w in word_list if w in litigious_lexicon)
    
    # Return metrics
    return fog_score, u_count, l_count, total_words

# --- MAIN APP LOGIC ---
# ... [Keep your Sidebar upload code] ...

if uploaded_file is not None:
    # ... [Keep your PDF extraction logic] ...
    
    # SIMULATED TEXT (Replace this with real extracted text later)
    # I am putting a "dummy" heavy financial text here for the demo
    extracted_text = """
    The Company estimates that the potential liabilities related to the pending litigation could be material, 
    but the outcome implies significant uncertainty and cannot be reasonably estimated at this time due to fluctuating market conditions.
    The approximate value of the intangible assets is dependent on future cash flow assumptions which are indefinite.
    Management believes the claims are without merit, but a contingency provision has been created.
    """ * 20  # Repeating to make it long enough for word cloud
    
    # ANALYZE
    fog, u_count, l_count, total_words = analyze_quantitative_metrics(extracted_text)
    
    st.title(f"📊 Quantitative Analysis: {uploaded_file.name}")
    
    # 1. METRICS ROW
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gunning Fog Index", f"{fog:.2f}", "Complexity Score")
    col2.metric("Uncertainty Words", u_count, f"{(u_count/total_words)*100:.1f}% Density")
    col3.metric("Litigious Words", l_count, "Legal Risks")
    col4.metric("Total Word Count", total_words, "Volume")
    
    st.markdown("---")
    
    # 2. WORD CLOUD SECTION
    st.subheader("☁️ Linguistic Word Cloud")
    st.caption("Visualizing the most frequent terms in the Notes to Accounts.")
    
    # Create WordCloud
    # We add "The", "Company", etc to stopwords so they don't clutter the chart
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(["Company", "Year", "financial", "statements", "note"])
    
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        stopwords=custom_stopwords,
        colormap='Reds'  # Red color scheme for "Risk" vibe
    ).generate(extracted_text)
    
    # Display using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
    
    # 3. REGRESSION / CORRELATION PREVIEW
    st.markdown("---")
    st.subheader("📈 Regression Helper: Extracted Variables")
    st.write("These are the quantitative variables you would export to run your Regression Analysis (e.g., in Excel or SPSS).")
    
    # Create a DataFrame that looks like an exportable dataset
    data_export = {
        "Metric": ["Fog Index (X1)", "Uncertainty Count (X2)", "Litigious Count (X3)", "Word Count (Control)"],
        "Value": [fog, u_count, l_count, total_words],
        "Interpretation": [
            "High (>18) indicates obfuscation.",
            "High count indicates hedging/vagueness.",
            "High count indicates legal trouble.",
            "Control variable for document size."
        ]
    }
    st.dataframe(pd.DataFrame(data_export), use_container_width=True)
    
    st.download_button(
        "📥 Download Data for Regression",
        data=pd.DataFrame(data_export).to_csv().encode('utf-8'),
        file_name='forensic_variables.csv',
        mime='text/csv'
    )
