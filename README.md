# NLP Text Similarity Explorer

A Streamlit web app that uses the **free pretrained model `all-MiniLM-L6-v2`**  
to compute and visualise text/sentence similarity — built for the NLP Lab Quiz.

---

## 🚀 Live App

👉 **[Click here to open the deployed app](https://pretrained-model-with-app-7nyfthdp8mlul5bsxkunmb.streamlit.app/)**  
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

## UI
<img width="505" height="150" alt="image" src="https://github.com/user-attachments/assets/00977fd8-6339-42c8-85c9-7b62ece459d9" />
<img width="551" height="256" alt="image" src="https://github.com/user-attachments/assets/2bc3bd23-c0b2-4cd0-ab55-e78075cab1a8" />
<img width="619" height="351" alt="image" src="https://github.com/user-attachments/assets/9d64828b-d9cb-463f-96f6-7e88ea13ff76" />
<img width="608" height="339" alt="image" src="https://github.com/user-attachments/assets/878a9037-889a-4936-9b33-7af339b5183b" />
<img width="609" height="357" alt="image" src="https://github.com/user-attachments/assets/1860480e-0b23-423e-8b1b-3c7aaedd653a" />
<img width="602" height="388" alt="image" src="https://github.com/user-attachments/assets/e7f9bfd4-4a03-4961-84d0-0de10b89a1f6" />
