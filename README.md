# NLP Text Similarity Explorer

A Streamlit web app that uses the **free pretrained model `all-MiniLM-L6-v2`**  
to compute and visualise text/sentence similarity — built for the NLP Lab Quiz.

---

## 🚀 Live App

👉 **[Click here to open the deployed app](https://your-app-name.streamlit.app)**  
*(Replace this link with your Streamlit Community Cloud URL after deployment)*

---

## 🤖 Model Used

| Property | Detail |
|---|---|
| **Model name** | `all-MiniLM-L6-v2` |
| **Source** | [sentence-transformers](https://www.sbert.net/) |
| **Cost** | Free — no API key needed |
| **Preprocessing** | None — raw text fed directly to the model |
| **Training** | None — pretrained weights used as-is |

---

## App Purpose

The app takes user-entered sentences or words (one per line), generates  
dense vector embeddings using the pretrained model, computes cosine  
similarity between every pair, and presents results through:

- **Similarity score table** — exact scores for all pairs  
- **Bar chart** — top similar pairs ranked by score  
- **Heatmap** — full pairwise similarity matrix  
- **2D PCA plot** — geometric view of sentence relationships  
- **Critical thinking analysis** — Paul's 7 standards applied to results  

---

## 📂 Repository Structure

```
nlp_similarity_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🖥️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/nlp-similarity-app.git
cd nlp-similarity-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📋 Strict Rules Followed

| Rule | Status |
|---|---|
| No preprocessing (tokenize/stem/lemmatize) | ✅ |
| No model training | ✅ |
| No paid API | ✅ |
| Free pretrained model only | ✅ |
| Runs from `app.py` | ✅ |
| GitHub has `app.py`, `requirements.txt`, `README.md` | ✅ |

---

## 🧠 Paul's Critical Thinking Standards

| Standard | Implementation |
|---|---|
| **Clarity** | Explains input/output in plain language |
| **Accuracy** | Names the exact model; no unsupported claims |
| **Precision** | Shows exact 4-decimal similarity scores |
| **Relevance** | All graphs directly reflect computed scores |
| **Logic** | Explains why top result makes sense |
| **Significance** | Highlights the most similar pair |
| **Fairness** | States the model's language/domain limitations |

---

## ⚙️ Deployment on Streamlit Community Cloud

1. Push this repository to GitHub (public)  
2. Go to [share.streamlit.io](https://share.streamlit.io)  
3. Click **New app** → select your repo → set `app.py` as main file  
4. Click **Deploy** — the app goes live in ~2 minutes
