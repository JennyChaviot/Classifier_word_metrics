# Classifier Word Metrics – % of Words per Post ID
# Streamlit, pandas, nltk only • ≈125 LOC

import streamlit as st, pandas as pd, nltk, re

def main() -> None:
    # ───── page & style ─────
    st.set_page_config(page_title="Classifier Word Metrics – % of Words per Post ID",
                       layout="centered")
    st.markdown(
        "<style>h1{font-family:serif;}"
        ".stButton>button{background:#DC143C;color:white;}</style>",
        unsafe_allow_html=True)
    st.title("Classifier Word Metrics – % of Words per Post ID")

    # ───── step 1 – upload ─────
    st.markdown("**Step 1 – Upload CSV(s)** (raw Instagram posts **or** a file whose "
                "columns already include `caption, shortcode, username, likes, comments`).")
    files = st.file_uploader("Drag one or more .csv files here",
                             type="csv", accept_multiple_files=True)

    def is_raw(df: pd.DataFrame) -> bool:
        return {"caption", "shortcode", "username", "likes", "comments"}.issubset(df.columns)

    df = None
    if files:
        raw, other = None, None
        for f in files:
            d = pd.read_csv(f)
            raw   = d if is_raw(d) and raw   is None else raw
            other = d if not is_raw(d) and other is None else other
        df = raw if raw is not None else other

    # ───── step 2 – dictionary ─────
    st.markdown("**Step 2 – Define keyword dictionary**  "
                "(one classifier per line → `name: word1, word2, …`).")
    default_dict = "luxury: diamante, couture, gala\ncasual: jeans, sneakers, chill"
    dict_text = st.text_area("Keyword Dictionary", value=default_dict, height=140)

    # ───── step 3 – generate ─────
    if st.button("Generate Metrics", type="primary"):
        if df is None:
            st.error("⚠️ Upload at least one compatible CSV first.")
            return

        # 3-A  parse dictionary
        classifiers: dict[str, set[str]] = {}
        for ln in dict_text.strip().splitlines():
            if ":" in ln:
                name, kws = ln.split(":", 1)
                kws = [k.strip().lower() for k in kws.split(",") if k.strip()]
                if kws:
                    classifiers[name.strip()] = set(kws)
        if not classifiers:
            st.error("⚠️ No valid classifiers detected.")
            return

        # 3-B  tokenise
        nltk.download("punkt", quiet=True)
        def tokens(txt: str) -> list[str]:
            words = nltk.word_tokenize(str(txt).lower())
            words = [re.sub(r"\W+", "", w) for w in words]
            return [w for w in words if w]

        # 3-C  count & aggregate
        rows = []
        for _, r in df.iterrows():
            tok = tokens(r["caption"])
            total = len(tok)
            counts = {n: sum(t in kws for t in tok) for n, kws in classifiers.items()}
            rows.append({"shortcode": r["shortcode"], "total_words": total, **counts})

        res = (pd.DataFrame(rows)
               .groupby("shortcode").sum(numeric_only=True).reset_index())
        for n in classifiers:
            res[f"%_{n}"] = res[n] / res["total_words"] * 100

        # 3-D  display & download
        st.subheader("Preview")
        st.dataframe(res.head())
        csv = res.to_csv(index=False).encode()
        st.download_button("Download full CSV", csv,
                           file_name="classifier_metrics.csv", mime="text/csv")

    # ───── sidebar requirements ─────
    with st.sidebar.expander("requirements.txt"):
        st.code("streamlit\npandas\nnltk", language="text")

if __name__ == "__main__":
    main()

