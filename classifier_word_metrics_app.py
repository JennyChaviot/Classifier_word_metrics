# classifier_word_metrics_app.py
# Streamlit app – % of classifier words per Instagram post (≈75 LOC)

import streamlit as st, pandas as pd, nltk, re
from nltk.tokenize import wordpunct_tokenize        # regex-based → no Punkt download

# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # page setup & a touch of “classic luxury”
    st.set_page_config(page_title="Classifier Word Metrics – % of Words per Post ID",
                       layout="centered")
    st.markdown(
        "<style>h1{font-family:serif;}"
        ".stButton>button{background:#DC143C;color:white;font-weight:600;}</style>",
        unsafe_allow_html=True)
    st.title("Classifier Word Metrics – % of Words per Post ID")

    # ── STEP 1 – upload CSV ──────────────────────────────────────────────────
    st.markdown("**Step 1 – Upload CSV(s)** "
                "(any file that has *both* an **ID** column and a **caption/text** column).")
    files = st.file_uploader("Drag one or more .csv files here", type="csv", accept_multiple_files=True)

    df = None
    if files:
        # accepted header aliases (case-insensitive, whitespace ignored)
        cap_aliases = {"caption", "statement", "cleaned", "text", "content"}
        id_aliases  = {"shortcode", "id", "postid", "post id"}

        for f in files:
            d = pd.read_csv(f)
            d.columns = d.columns.str.strip().str.lower()             # normalise headers
            cap_col = next((c for c in d.columns if c in cap_aliases), None)
            id_col  = next((c for c in d.columns if c in id_aliases),  None)
            if cap_col and id_col:                                    # found a usable file
                df = d.rename(columns={cap_col: "caption",
                                       id_col:  "shortcode"})[["shortcode", "caption"]]
                break

        if df is None:
            st.error("⚠️ Couldn’t find both an ID column and a caption/text column "
                     "in the uploaded file(s).")

    else:
        st.info("↖️ Upload a CSV to continue.")

    # ── STEP 2 – keyword dictionary ─────────────────────────────────────────
    dict_default = "luxury: diamante, couture, gala\ncasual: jeans, sneakers, chill"
    dict_text = st.text_area("**Step 2 – Keyword Dictionary** "
                             "(one classifier per line → `name: word1, word2, …`).",
                             value=dict_default, height=140)

    # ── STEP 3 – generate metrics ───────────────────────────────────────────
    if st.button("Generate Metrics", type="primary"):
        if df is None:
            st.error("⚠️ Please upload a compatible CSV first.")
            return

        # parse dictionary
        classifiers: dict[str, set[str]] = {}
        for line in dict_text.strip().splitlines():
            if ":" in line:
                name, kws = line.split(":", 1)
                kws = [k.strip().lower() for k in kws.split(",") if k.strip()]
                if kws:
                    classifiers[name.strip()] = set(kws)
        if not classifiers:
            st.error("⚠️ No valid classifiers detected."); return

        # tokeniser (regex, punctuation stripped)
        def tokens(txt: str) -> list[str]:
            words = wordpunct_tokenize(str(txt).lower())
            return [re.sub(r"\W+", "", w) for w in words if re.sub(r"\W+", "", w)]

        # count + aggregate
        rows = []
        for _, r in df.iterrows():
            tok = tokens(r["caption"])
            total = len(tok)
            counts = {n: sum(w in kws for w in tok) for n, kws in classifiers.items()}
            rows.append({"shortcode": r["shortcode"], "total_words": total, **counts})

        res = (pd.DataFrame(rows)
               .groupby("shortcode").sum(numeric_only=True).reset_index())
        for n in classifiers:
            res[f"%_{n}"] = res[n] / res["total_words"] * 100

        # show + download
        st.subheader("Preview"); st.dataframe(res.head())
        st.download_button("Download full CSV",
                           res.to_csv(index=False).encode(),
                           "classifier_metrics.csv", mime="text/csv")

    # ── minimalist requirements.txt for reference ───────────────────────────
    with st.expander("requirements.txt"):
        st.code("streamlit\npandas\nnltk", language="text")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
