# classifier_word_metrics_app.py
# Streamlit single-page app – % of classifier words per Instagram post
# Dependencies: streamlit, pandas, nltk   •   <150 LOC

import streamlit as st, pandas as pd, nltk, re
from nltk.tokenize import wordpunct_tokenize   # regex-based → no Punkt download

# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # Page config & minimal luxury styling
    st.set_page_config(page_title="Classifier Word Metrics – % of Words per Post ID",
                       layout="centered")
    st.markdown(
        "<style>h1{font-family:serif;} "
        ".stButton>button{background:#DC143C;color:white;font-weight:600;}"
        "</style>",
        unsafe_allow_html=True)
    st.title("Classifier Word Metrics – % of Words per Post ID")

    # ── STEP 1 : Upload CSV ───────────────────────────────────────────────────
    st.markdown("**Step 1 – Upload CSV(s)**  "
                "(raw Instagram posts **or** a file that already contains "
                "`caption, shortcode, username, likes, comments`).")
    files = st.file_uploader("Drag one or more .csv files here",
                             type="csv", accept_multiple_files=True)

    mandatory = {"caption", "shortcode", "username", "likes", "comments"}
    df = raw = other = None
    if files:
        for f in files:
            d = pd.read_csv(f)
            d.columns = d.columns.str.strip().str.lower()       # normalise headers
            has_all = mandatory.issubset(d.columns)
            raw   = d if has_all and raw   is None else raw     # prioritise raw-post file
            other = d if not has_all and other is None else other
        df = raw if raw is not None else other
        if df is not None and not mandatory.issubset(df.columns):
            miss = ", ".join(mandatory.difference(df.columns))
            st.error(f"⚠️ Uploaded file is missing column(s): {miss}")
            df = None                                           # stop later steps

    # ── STEP 2 : Keyword dictionary ───────────────────────────────────────────
    st.markdown("**Step 2 – Define keyword dictionary**  "
                "(one classifier per line → `name: word1, word2, …`).")
    dict_default = "luxury: diamante, couture, gala\ncasual: jeans, sneakers, chill"
    dict_text = st.text_area("Keyword Dictionary", value=dict_default, height=140)

    # ── STEP 3 : Generate metrics ─────────────────────────────────────────────
    if st.button("Generate Metrics", type="primary"):
        if df is None:
            st.error("⚠️ Please upload at least one compatible CSV first.")
            return

        # 3-A  parse dictionary ------------------------------------------------
        classifiers: dict[str, set[str]] = {}
        for line in dict_text.strip().splitlines():
            if ":" in line:
                name, kws = line.split(":", 1)
                kws = [k.strip().lower() for k in kws.split(",") if k.strip()]
                if kws:
                    classifiers[name.strip()] = set(kws)
        if not classifiers:
            st.error("⚠️ No valid classifiers detected.")
            return

        # 3-B  tokenise captions ----------------------------------------------
        def tokens(text: str) -> list[str]:
            words = wordpunct_tokenize(str(text).lower())
            words = [re.sub(r"\W+", "", w) for w in words]       # strip punctuation
            return [w for w in words if w]

        # 3-C  count & aggregate ----------------------------------------------
        rows: list[dict] = []
        for _, r in df.iterrows():
            tok = tokens(r["caption"])
            total = len(tok)
            counts = {n: sum(w in kws for w in tok) for n, kws in classifiers.items()}
            rows.append({"shortcode": r["shortcode"], "total_words": total, **counts})

        res = (pd.DataFrame(rows)
               .groupby("shortcode").sum(numeric_only=True).reset_index())
        for n in classifiers:
            res[f"%_{n}"] = res[n] / res["total_words"] * 100

        # 3-D  display & download ---------------------------------------------
        st.subheader("Preview");  st.dataframe(res.head())
        st.download_button("Download full CSV",
                           res.to_csv(index=False).encode(),
                           "classifier_metrics.csv", mime="text/csv")

    # ── Minimal requirements.txt (main-page expander) ────────────────────────
    with st.expander("requirements.txt"):
        st.code("streamlit\npandas\nnltk", language="text")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()


