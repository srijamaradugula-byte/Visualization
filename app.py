import streamlit as st
import pdfplumber
import docx
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
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
st.write("Upload a PDF or DOCX file to visualize word frequencies and explore word relationships.")

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
    top_words = word_freq.most_common(20)
    df = pd.DataFrame(top_words, columns=["Word", "Frequency"])

    st.markdown(f"*Total Words (after filtering):* {sum(word_freq.values())}")

    # --- WordCloud ---
    st.subheader("☁ WordCloud")
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    # --- Bar Chart ---
    st.subheader("📊 Top 20 Words - Bar Chart")
    fig_bar, ax_bar = plt.subplots()
    sns.barplot(x="Frequency", y="Word", data=df, ax=ax_bar, palette="viridis")
    ax_bar.set_title("Most Frequent Words")
    st.pyplot(fig_bar)

    # --- Word Co-occurrence Matrix (Confusion Matrix Style) ---
    st.subheader("🔄 Word Co-occurrence Matrix")

    # Use top 20 words for matrix
    top_words_list = [word for word, _ in top_words]

    # Vectorize sentences
    sentences = re.split(r'[.!?]', text)
    vectorizer = CountVectorizer(vocabulary=top_words_list, lowercase=True, stop_words='english')
    X = vectorizer.fit_transform(sentences)

    # Co-occurrence matrix
    co_matrix = (X.T @ X).toarray()
    np.fill_diagonal(co_matrix, 0)  # Remove self-co-occurrence

    # Display heatmap
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(co_matrix, xticklabels=top_words_list, yticklabels=top_words_list, cmap="Reds", annot=True, fmt="d", ax=ax_cm)
    ax_cm.set_title("Word Co-occurrence Matrix")
    st.pyplot(fig_cm)
