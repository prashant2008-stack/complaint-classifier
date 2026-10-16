import pandas as pd, streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

st.set_page_config(page_title="Smart Complaint Classifier", page_icon="📨", layout="wide")
st.title("📨 Smart Complaint Classification")
st.caption("Complaint likho → AI sahi department/category batayega")

# ---------- Data ----------
up = st.sidebar.file_uploader("Apna dataset CSV upload karo (optional)", type="csv")
df = pd.read_csv(up) if up else pd.read_csv("complaints.csv")
text_col = st.sidebar.selectbox("Complaint text column", df.columns, index=list(df.columns).index("complaint") if "complaint" in df.columns else 0)
label_col = st.sidebar.selectbox("Category column", df.columns, index=list(df.columns).index("category") if "category" in df.columns else len(df.columns)-1)
df = df[[text_col, label_col]].dropna()
df.columns = ["text", "label"]

# ---------- Train (cached) ----------
@st.cache_resource
def train(data):
    X_tr, X_te, y_tr, y_te = train_test_split(data.text, data.label, test_size=0.2, random_state=42, stratify=data.label)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    Xtr, Xte = vec.fit_transform(X_tr), vec.transform(X_te)
    models = {"Logistic Regression": LogisticRegression(max_iter=1000), "Naive Bayes": MultinomialNB()}
    res = {}
    for n, m in models.items():
        m.fit(Xtr, y_tr); p = m.predict(Xte)
        res[n] = {"Accuracy": accuracy_score(y_te, p), "F1 (macro)": f1_score(y_te, p, average="macro"),
                  "report": classification_report(y_te, p, output_dict=True, zero_division=0)}
    return vec, models, res

vec, models, res = train(df)
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Model Comparison", "🔑 Top Words", "🗂 Dataset"])

with tab1:
    choice = st.selectbox("Model chuno", list(models), index=0)
    ex = "I returned the item two weeks ago but my refund has not been credited yet."
    text = st.text_area("Complaint likho", ex, height=140)
    if st.button("Classify karo", type="primary") and text.strip():
        m = models[choice]; probs = m.predict_proba(vec.transform([text]))[0]
        order = np.argsort(probs)[::-1]
        st.success(f"Predicted Category: **{m.classes_[order[0]]}**  ({probs[order[0]]*100:.1f}% confidence)")
        st.bar_chart(pd.Series(probs, index=m.classes_).sort_values(ascending=False))

with tab2:
    st.subheader("Do models ki comparison (80/20 split)")
    st.dataframe(pd.DataFrame({n: {k: f"{v:.3f}" for k, v in r.items() if k != "report"} for n, r in res.items()}).T)
    st.bar_chart(pd.DataFrame({n: {k: v for k, v in r.items() if k != "report"} for n, r in res.items()}).T)
    best = max(res, key=lambda n: res[n]["F1 (macro)"])
    st.info(f"Best model (F1 ke hisaab se): **{best}**")

with tab3:
    lr = models["Logistic Regression"]; names = np.array(vec.get_feature_names_out())
    cols = st.columns(3)
    for i, c in enumerate(lr.classes_):
        top = names[np.argsort(lr.coef_[i])[::-1][:8]]
        with cols[i % 3]:
            st.markdown(f"**{c}**"); st.write(", ".join(top))

with tab4:
    st.write(f"{len(df)} complaints, {df.label.nunique()} categories")
    st.bar_chart(df.label.value_counts())
    st.dataframe(df.head(50))
