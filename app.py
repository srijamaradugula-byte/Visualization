import streamlit as st
import pdfplumber
import docx
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk

# --- Setup ---
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# --- Text Extraction ---
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# --- Word Frequency ---
def get_word_frequencies(text):
    words = re.findall(r'\w+', text.lower())
    filtered_words = [word for word in words if word not in STOPWORDS and len(word) > 2]
    return Counter(filtered_words)

# --- Streamlit UI ---
st.set_page_config(page_title="Text Analysis App", layout="wide")
st.title("📊 Text Analysis App")
st.write("Upload a PDF or DOCX file to visualize word frequencies.")

uploaded_file = st.file_uploader("📁 Choose a file", type=["pdf", "docx"])

if uploaded_file:
    file_type = "PDF" if uploaded_file.name.endswith(".pdf") else "DOCX"
    st.success(f"Uploaded {file_type} file: {uploaded_file.name}")

    # Extract text
    text = extract_text_from_pdf(uploaded_file) if file_type == "PDF" else extract_text_from_docx(uploaded_file)

    st.subheader("📄 Extracted Text Preview")
    st.text_area("Preview", text[:3000] + "..." if len(text) > 3000 else text, height=200)

    # Word frequency analysis
    word_freq = get_word_frequencies(text)
    df = pd.DataFrame(word_freq.items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)
    top_words = df.head(20)

    st.markdown(f"*Total Words (after filtering):* {df['Frequency'].sum()}")

    # --- WordCloud ---
    st.subheader("☁ WordCloud")
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(word_freq))
    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    # --- Histogram ---
    st.subheader("📈 Word Frequency Histogram")
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(df["Frequency"], bins=30, color="skyblue", edgecolor="black")
    ax_hist.set_title("Distribution of Word Frequencies")
    ax_hist.set_xlabel("Frequency")
    ax_hist.set_ylabel("Count")
    st.pyplot(fig_hist)

    # --- Bar Chart ---
    st.subheader("📊 Top 20 Words - Bar Chart")
    fig_bar, ax_bar = plt.subplots()
    sns.barplot(x="Frequency", y="Word", data=top_words, ax=ax_bar, palette="viridis")
    ax_bar.set_title("Most Frequent Words")
    st.pyplot(fig_bar)
