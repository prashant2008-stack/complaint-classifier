# Smart Complaint Classification (PS-01)

An ML-powered web app that automatically classifies customer complaints into the right category or department.

## Problem
Organizations receive a large number of customer complaints through emails, forms and support systems. Sorting them manually into the correct category takes time and delays responses.

## Solution
The app converts complaint text into TF-IDF features and uses machine learning to predict the complaint category. Two classification approaches, **Logistic Regression** and **Naive Bayes**, are trained and compared. A Streamlit interface lets users classify new complaints, compare models and explore the most important words per category.

## Features
- **Predict:** enter a complaint to get the predicted category, confidence % and a probability chart
- **Model Comparison:** accuracy and macro F1-score of both models on a held-out test set (80/20 split)
- **Top Words:** the most influential words for each category (from Logistic Regression coefficients)
- **Dataset Overview:** category distribution and data preview
- **Custom data:** upload your own CSV and choose the text and category columns from the sidebar

## Tech Stack
Python, pandas, scikit-learn, Streamlit

## Setup
```bash
pip install -r requirements.txt
python generate_data.py        # optional: regenerates the sample dataset
python -m streamlit run app.py
```
The app opens at `http://localhost:8501`.

## Project Structure
| File | Purpose |
|---|---|
| `app.py` | Preprocessing (TF-IDF), model training, evaluation and the Streamlit UI |
| `generate_data.py` | Creates the sample dataset `complaints.csv` (600 complaints, 6 categories) |
| `complaints.csv` | Sample dataset (columns: `complaint`, `category`) |
| `requirements.txt` | Python dependencies |

## Workflow
Dataset → TF-IDF vectorization → train Logistic Regression and Naive Bayes → evaluate (accuracy, F1) → predict on new complaints via the UI

## Note on Data
The included dataset is a synthetic sample used for development and demonstration. Results on a real dataset may differ.

## Future Scope
Hindi/Hinglish support, transformer-based models (e.g. BERT), urgency detection, and email/helpdesk integration.

