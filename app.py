"""
app.py
======
NLP Similarity Explorer - Complete Assignment Solution

Model: sentence-transformers/all-MiniLM-L6-v2 (Free, Apache 2.0)
No preprocessing · No training · No paid API
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time

# ──────────────────────────────────────────────────────────────
#  Page Configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Similarity Explorer",
    page_icon="🔮",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_DIM = 384

# ──────────────────────────────────────────────────────────────
#  Custom CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: #0b0b14;
    }
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #6C63FF, #A78BFA);
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
    }
    .card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }
    .card h3 {
        color: #f0f0f8;
        margin: 0 0 0.5rem 0;
    }
    .card p {
        color: #c8c8d8;
        margin: 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #5A52D5);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 2rem;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4);
    }
    .footer {
        text-align: center;
        color: #6a6a80;
        font-size: 0.75rem;
        padding: 1rem 0;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  Load Model
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

with st.spinner(f"Loading {MODEL_NAME}..."):
    model = load_model()

# ──────────────────────────────────────────────────────────────
#  Helper Functions
# ──────────────────────────────────────────────────────────────
def similarity_label(score):
    if score >= 0.75:
        return "🔵 Very Similar", "#6C63FF"
    elif score >= 0.55:
        return "🟢 Similar", "#4ECB71"
    elif score >= 0.35:
        return "🟡 Moderately Similar", "#F9A826"
    else:
        return "🔴 Dissimilar", "#FF6B6B"

def compute_paul_scores(primary_score, all_scores, n_texts):
    """Calculate Paul's Critical Thinking Standards scores"""
    scores = {}
    
    # Clarity
    clarity = 30 + (20 if n_texts >= 2 else 0) + (20 if n_texts >= 3 else 0) + (30 if primary_score > 0 else 0)
    scores["Clarity"] = min(clarity, 100)
    
    # Accuracy
    accuracy = 25 + (25 if n_texts >= 2 else 0) + (25 if primary_score > 0 else 0) + (25 if len(all_scores) > 1 else 0)
    scores["Accuracy"] = min(accuracy, 100)
    
    # Precision
    precision = 10 + (40 if primary_score > 0.7 else 25 if primary_score > 0.5 else 0)
    precision += 20 if len(all_scores) >= 2 else 0
    if len(all_scores) >= 2 and max(all_scores) - min(all_scores) > 0.3:
        precision += 20
    scores["Precision"] = min(precision, 100)
    
    # Relevance
    relevance = 25 + (25 if len(all_scores) > 0 else 0) + (25 if primary_score > 0 else 0) + (25 if n_texts >= 2 else 0)
    scores["Relevance"] = min(relevance, 100)
    
    # Logic
    logic = 15 + (30 if primary_score > 0.5 else 0) + (25 if len(all_scores) >= 2 else 0)
    if len(all_scores) >= 2 and max(all_scores) > 0 and min(all_scores) < max(all_scores):
        logic += 30
    scores["Logic"] = min(logic, 100)
    
    # Significance
    significance = 10 + (50 if primary_score > 0.7 else 30 if primary_score > 0.5 else 0)
    significance += 20 if len(all_scores) >= 2 else 0
    if len(all_scores) >= 2 and max(all_scores) > min(all_scores):
        significance += 20
    scores["Significance"] = min(significance, 100)
    
    # Fairness
    fairness = 70 + (15 if n_texts >= 3 else 0) + (15 if len(all_scores) >= 2 else 0)
    scores["Fairness"] = min(fairness, 100)
    
    return scores

# ──────────────────────────────────────────────────────────────
#  Graph Functions
# ──────────────────────────────────────────────────────────────
def graph_bar_chart(top_results):
    """Bar Chart: Top similar words/sentences with similarity scores"""
    if not top_results:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No data", ha='center', va='center', color='white')
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0b0b14')
        return fig
    
    labels = [r['text'][:40] + ("..." if len(r['text']) > 40 else "") for r in top_results]
    scores = [r['score'] for r in top_results]
    colors = ['#6C63FF' if s >= 0.5 else '#FF6B6B' for s in scores]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1], edgecolor='white', height=0.6)
    ax.set_xlabel("Cosine Similarity Score", fontsize=11, color='white')
    ax.set_title("Top Similarity Scores", fontsize=14, fontweight='700', color='white')
    ax.set_xlim(0, 1)
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0b0b14')
    ax.tick_params(colors='white')
    
    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{score:.4f}", va='center', fontsize=9, color='white')
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='0.5 threshold')
    ax.legend(loc='lower right', fontsize=9, facecolor='#1a1a2e', edgecolor='white')
    plt.tight_layout()
    return fig

def graph_heatmap(matrix, texts):
    """Heatmap: Pairwise similarity between selected words/sentences"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(texts)))
    ax.set_yticks(range(len(texts)))
    ax.set_xticklabels([f"T{i+1}" for i in range(len(texts))], color='white')
    ax.set_yticklabels([f"T{i+1}" for i in range(len(texts))], color='white')
    ax.set_title("Pairwise Similarity Heatmap", fontsize=14, fontweight='700', color='white')
    fig.patch.set_facecolor('#0b0b14')
    ax.set_facecolor('#1a1a2e')
    
    for i in range(len(texts)):
        for j in range(len(texts)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha='center', va='center', color='white', fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Similarity Score')
    plt.tight_layout()
    return fig

def graph_pca(embeddings, texts):
    """2D Embedding Plot: PCA showing related terms near each other"""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#6C63FF' if i == 0 else '#4ECB71' for i in range(len(texts))]
    sizes = [150 if i == 0 else 100 for i in range(len(texts))]
    
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes, edgecolors='white', linewidths=1.5)
    ax.set_title("2D PCA Embedding Projection", fontsize=14, fontweight='700', color='white')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10, color='white')
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10, color='white')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0b0b14')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='gray')
    
    for i, label in enumerate(texts):
        label_short = label[:20] + ("..." if len(label) > 20 else "")
        ax.annotate(f"{i+1}. {label_short}", (coords[i, 0], coords[i, 1]),
                   textcoords="offset points", xytext=(8, 5), fontsize=8, color='white')
    
    plt.tight_layout()
    return fig

def graph_paul_radar(scores):
    """Radar chart for Paul's Critical Thinking Standards"""
    if not scores:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "No data", ha='center', va='center', color='white')
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0b0b14')
        return fig
    
    categories = list(scores.keys())
    values = list(scores.values())
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, color='#6C63FF')
    ax.fill(angles, values, alpha=0.25, color='#6C63FF')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color='white')
    ax.set_title("Paul's Critical Thinking Standards", fontsize=14, fontweight='700', color='white')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0b0b14')
    ax.grid(True, color='gray', alpha=0.3)
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────
#  Main App
# ──────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔮 NLP Similarity Explorer</h1>
        <p>Compare text similarity using <strong>all-MiniLM-L6-v2</strong> · Free · No training · No preprocessing</p>
    </div>
    """, unsafe_allow_html=True)

    # Input Section
    st.markdown("### ✍️ Enter Your Texts")
    st.caption("Enter 2 to 6 texts. The model computes cosine similarity between ALL pairs.")

    n_inputs = st.slider("Number of texts to compare", min_value=2, max_value=6, value=3)

    default_texts = [
        "Artificial intelligence is transforming the world.",
        "Machine learning enables computers to learn from data.",
        "Deep learning models are a subset of machine learning.",
        "Natural language processing handles text and speech.",
        "Computer vision allows machines to see and interpret images.",
        "Robotics involves designing intelligent machines.",
    ]

    texts = []
    cols = st.columns(2)
    for i in range(n_inputs):
        with cols[i % 2]:
            default = default_texts[i] if i < len(default_texts) else ""
            txt = st.text_area(
                f"Text {i+1}",
                value=default,
                height=80,
                key=f"text_{i}",
            )
            texts.append(txt)

    st.markdown("---")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        top_k = st.slider("Top-K results to show", min_value=1, max_value=min(5, n_inputs-1), value=min(3, n_inputs-1))
    with col2:
        run_btn = st.button("🚀 Analyze", use_container_width=True)
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            for i in range(n_inputs):
                if f"text_{i}" in st.session_state:
                    del st.session_state[f"text_{i}"]
            st.rerun()

    if run_btn:
        clean_texts = [t.strip() for t in texts if t.strip()]
        if len(clean_texts) < 2:
            st.warning("⚠️ Please enter at least 2 non-empty texts.")
            st.stop()

        # ── Compute Embeddings ──────────────────────────────────
        with st.spinner("Generating embeddings..."):
            start_time = time.time()
            embeddings = model.encode(clean_texts)
            elapsed_ms = (time.time() - start_time) * 1000

        # ── Compute Similarity ──────────────────────────────────
        matrix = cosine_similarity(embeddings)
        primary_score = matrix[0][1]
        label, color = similarity_label(primary_score)

        all_scores = []
        for i in range(len(clean_texts)):
            for j in range(i+1, len(clean_texts)):
                all_scores.append(float(matrix[i][j]))

        top_results = []
        for i in range(1, len(clean_texts)):
            top_results.append({"rank": i, "text": clean_texts[i], "score": float(matrix[0][i])})
        top_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = top_results[:top_k]

        # ── Paul's Standards ────────────────────────────────────
        paul_scores = compute_paul_scores(primary_score, all_scores, len(clean_texts))

        # ── Store in session ────────────────────────────────────
        st.session_state.current = {
            "texts": clean_texts,
            "embeddings": embeddings,
            "matrix": matrix,
            "all_scores": all_scores,
            "top_results": top_results,
            "paul_scores": paul_scores,
            "primary_score": primary_score,
            "elapsed_ms": elapsed_ms,
        }

        # ── Results Display ─────────────────────────────────────

        # Success Banner
        st.markdown(f"""
        <div style="background:rgba(108,99,255,0.12);border:1px solid rgba(108,99,255,0.25);border-radius:12px;padding:1rem;margin:1rem 0;">
            <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
                <span style="font-size:2rem;">✅</span>
                <span style="font-weight:600;color:#f0f0f8;">Analysis Complete</span>
                <span style="color:#8e8ea0;">Primary similarity score:</span>
                <span style="font-weight:700;color:#6C63FF;font-size:1.3rem;">{primary_score:.4f}</span>
                <span style="font-size:0.9rem;color:#a8a8c0;">({label})</span>
                <span style="color:#6a6a80;font-size:0.8rem;">| Time: {elapsed_ms:.0f}ms</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Top Results ──────────────────────────────────────────
        st.markdown("### 🏆 Top Similar Results")

        for i, r in enumerate(top_results, 1):
            lbl, col = similarity_label(r['score'])
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:0.6rem 1rem;margin:0.3rem 0;display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <span style="font-weight:600;color:#6C63FF;">#{i}</span>
                    <span style="color:#f0f0f8;margin-left:0.8rem;">{r['text']}</span>
                </div>
                <div style="display:flex;align-items:center;gap:0.8rem;">
                    <span style="color:#8e8ea0;font-size:0.85rem;">{lbl}</span>
                    <span style="font-weight:700;color:{col};">{r['score']:.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── GRAPH 1: Bar Chart ──────────────────────────────────
        st.markdown("### 📊 Graph 1 — Bar Chart: Top Similar Words/Sentences")
        st.caption("Shows top similar words/sentences with their exact similarity scores.")
        fig_bar = graph_bar_chart(top_results)
        st.pyplot(fig_bar)
        plt.close()

        st.markdown("---")

        # ── GRAPH 2: Heatmap ────────────────────────────────────
        st.markdown("### 🌡️ Graph 2 — Heatmap: Pairwise Similarity Matrix")
        st.caption("Shows pairwise similarity between selected words/sentences.")
        fig_heat = graph_heatmap(matrix, clean_texts)
        st.pyplot(fig_heat)
        plt.close()

        st.markdown("---")

        # ── GRAPH 3: 2D PCA ─────────────────────────────────────
        st.markdown("### 🗺️ Graph 3 — 2D Embedding Plot (PCA)")
        st.caption("PCA projection showing related terms near each other.")
        fig_pca = graph_pca(embeddings, clean_texts)
        st.pyplot(fig_pca)
        plt.close()

        st.markdown("---")

        # ── GRAPH 4: Paul's Radar (Bonus) ──────────────────────
        st.markdown("### 🧠 Paul's Critical Thinking Standards Radar")
        fig_radar = graph_paul_radar(paul_scores)
        st.pyplot(fig_radar)
        plt.close()

        st.markdown("---")

        # ── Paul's Critical Thinking Standards ──────────────────
        st.markdown("### 📋 Paul's Critical Thinking Standards Analysis")

        top = top_results[0] if top_results else None
        bottom = top_results[-1] if top_results else None
        tscore = top['score'] if top else 0
        bscore = bottom['score'] if bottom else 0
        top_text = top['text'] if top else ""
        bottom_text = bottom['text'] if bottom else ""

        standards = {
            "🔵 Clarity": (
                f"**Score: {paul_scores['Clarity']}%**  \n"
                f"The user entered **{len(clean_texts)}** texts. The query was: *'{clean_texts[0]}'*.  \n"
                f"The model converted each text into a **{MODEL_DIM}-dimensional vector**.  \n"
                f"**Cosine similarity** (0=unrelated, 1=identical) was computed for every pair.  \n"
                f"Results are displayed as exact scores, a bar chart, a heatmap, and a PCA plot."
            ),
            "🟢 Accuracy": (
                f"**Score: {paul_scores['Accuracy']}%**  \n"
                f"Model used: **{MODEL_NAME}** from sentence-transformers (Apache 2.0).  \n"
                f"No preprocessing, training, or post-processing was applied.  \n"
                f"Scores are **raw cosine similarities** between embeddings.  \n"
                f"No claims beyond what the model produces are made."
            ),
            "🟡 Precision": (
                f"**Score: {paul_scores['Precision']}%**  \n"
                f"**Highest similarity:** **{tscore:.4f}** between query and *'{top_text}'*.  \n"
                f"**Lowest similarity:** **{bscore:.4f}** between query and *'{bottom_text}'*.  \n"
                f"All scores are shown with **4-decimal precision**."
            ),
            "🟠 Relevance": (
                f"**Score: {paul_scores['Relevance']}%**  \n"
                f"**Graph 1 (Bar Chart):** Ranks pairs by similarity score.  \n"
                f"**Graph 2 (Heatmap):** Shows complete pairwise similarity matrix.  \n"
                f"**Graph 3 (2D PCA):** Shows geometric closeness of embeddings."
            ),
            "🔴 Logic": (
                f"**Score: {paul_scores['Logic']}%**  \n"
                f"The top pair scored **{tscore:.4f}** because both sentences share similar "
                f"semantic meaning in the embedding space.  \n"
                f"The heatmap confirms this relationship visually."
            ),
            "🟣 Significance": (
                f"**Score: {paul_scores['Significance']}%**  \n"
                f"The **most important finding** is the highest-scoring pair "
                f"(score = **{tscore:.4f}**).  \n"
                f"Scores above **0.70** indicate strong semantic similarity."
            ),
            "⚪ Fairness": (
                f"**Score: {paul_scores['Fairness']}%**  \n"
                f"**Limitation:** {MODEL_NAME} is optimised for **English** and may "
                f"produce lower-quality embeddings for other languages, domain-specific "
                f"jargon, or very short single-word inputs.  \n"
                f"It reflects biases present in its training corpus."
            ),
        }

        for standard, explanation in standards.items():
            with st.expander(standard, expanded=True):
                st.markdown(explanation)

        # ── Model Info ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("### ℹ️ Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Model:** `{MODEL_NAME}`")
            st.markdown(f"**Dimension:** {MODEL_DIM}")
            st.markdown(f"**License:** Apache 2.0")
        with col2:
            st.markdown(f"**Framework:** Sentence-Transformers")
            st.markdown(f"**Type:** Free pretrained embedding model")
            st.markdown(f"**Inference Time:** {elapsed_ms:.1f}ms")

    else:
        # Show placeholder when no analysis
        st.info("👆 Enter your texts above and click **Analyze** to see results.")

    # ── Footer ──────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        🔮 NLP Similarity Explorer · Model: all-MiniLM-L6-v2 · Free · No training · No preprocessing
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
