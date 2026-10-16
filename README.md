# Smart Complaint Classification (PS-01)

**Problem:** Customer complaints ko manually sahi category/department me bhejna slow hai.
**Solution:** TF-IDF + ML (Logistic Regression & Naive Bayes) se complaint ki category automatically predict hoti hai. Streamlit UI me prediction, model comparison aur top words dikhte hain.

## Setup
```
pip install -r requirements.txt
python generate_data.py     # sirf agar dataset nahi hai
streamlit run app.py
```
## Components
- `generate_data.py` – sample dataset (complaints.csv)
- `app.py` – preprocessing (TF-IDF), training, evaluation, UI
- Sidebar se apna CSV upload kar sakte ho (text + category column)

## Features
Predict category + confidence • 2 models ki comparison • Har category ke top words • Dataset overview
