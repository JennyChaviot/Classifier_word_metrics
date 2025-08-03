import streamlit as st
import pandas as pd
from collections import Counter

# ---------- File upload ----------
st.header("Classifier Word Metrics – % of Words per Post ID")
uploaded = st.file_uploader("Upload CSV(s)", type="csv", accept_multiple_files=False)
if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)

# ---------- Column selectors ----------
with st.expander("Preview data"):
    st.dataframe(df.head())

text_col   = st.selectbox("Text column", df.columns, index=0)
gt_col     = st.selectbox("Ground-truth 0/1 column", df.columns, index=1)
id_col_opt = st.selectbox("(Optional) ID column for aggregation", ["<none – keep each row>"] + list(df.columns))

# ---------- Dictionary setup ----------
dict_mode = st.radio("Keyword Dictionary", ["Generate from data", "Provide custom dictionary"])
custom_dict_box = st.empty()

if dict_mode == "Provide custom dictionary":
    raw_dict = custom_dict_box.text_area("One classifier per line  ➜  name: word1, word2, …")
    classifiers = {}
    for line in raw_dict.splitlines():
        if ":" in line:
            name, words = line.split(":", 1)
            classifiers[name.strip()] = [w.strip() for w in words.split(",") if w.strip()]
else:
    top_n = st.slider("Top-N most frequent words", 5, 50, 20)
    pos_text = " ".join(df.loc[df[gt_col] == 1, text_col])
    freqs = Counter(pos_text.lower().split())
    classifiers = {"auto_top_words": [w for w, _ in freqs.most_common(top_n)]}
    st.write(pd.DataFrame(freqs.most_common(top_n), columns=["word", "count"]))

if not classifiers:
    st.warning("⚠ No valid classifiers detected.")
    st.stop()

# ---------- Metric computation ----------
def pct_of_words(text, keywords):
    words = text.lower().split()
    return sum(w in keywords for w in words) / len(words) if words else 0

for name, kws in classifiers.items():
    df[f"%_{name}"] = df[text_col].apply(lambda t: pct_of_words(t, kws))

if id_col_opt != "<none – keep each row>":
    agg_cols = [c for c in df.columns if c.startswith("%_")]
    df = df.groupby(id_col_opt)[agg_cols].mean().reset_index()

st.subheader("Metrics")
st.dataframe(df)
