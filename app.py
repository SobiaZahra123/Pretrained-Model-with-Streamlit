import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="NLP Similarity Explorer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 NLP Text Similarity Explorer")
st.markdown(
    "**Model:** `all-MiniLM-L6-v2` — free pretrained sentence embedding model "
    "from [sentence-transformers](https://www.sbert.net/)  \n"
    "**Rule:** No preprocessing · No training · No paid API"
)
st.divider()

# ── Load model 
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

with st.spinner("Loading pretrained model (all-MiniLM-L6-v2)…"):
    model = load_model()

st.success(" Model loaded: all-MiniLM-L6-v2")

# ── Sidebar – input ───────────────────────────────────────────
st.sidebar.header("📝 Input")
st.sidebar.markdown("Enter one sentence/word per line (min 2, max 10).")

default_text = (
    "Artificial intelligence is changing the world\n"
    "Machine learning helps computers learn from data\n"
    "Deep learning uses neural networks\n"
    "Natural language processing handles text\n"
    "Computer vision processes images\n"
    "Robotics involves autonomous machines"
)

raw_input = st.sidebar.text_area(
    "Enter sentences or words (one per line):",
    value=default_text,
    height=220
)

run_btn = st.sidebar.button("▶ Compute Similarity", type="primary")

# ── Main logic ─────────────
if run_btn:

    # parse lines — NO manual preprocessing, model handles it
    sentences = [line.strip() for line in raw_input.strip().split("\n") if line.strip()]

    if len(sentences) < 2:
        st.error("Please enter at least 2 sentences.")
        st.stop()

    if len(sentences) > 10:
        sentences = sentences[:10]
        st.warning("Using first 10 sentences only.")

    st.subheader("📋 Input Sentences")
    for i, s in enumerate(sentences):
        st.markdown(f"**{i+1}.** {s}")

    # ── Embed ─────────────────────────────────────
    with st.spinner("Computing embeddings…"):
        embeddings = model.encode(sentences)

    # cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # top similar pairs (exclude diagonal)
    pairs = []
    n = len(sentences)
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((sentences[i], sentences[j], sim_matrix[i][j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    st.divider()

    # ── Similarity scores table ───────────────────────────────
    st.subheader("📊 Top Similarity Pairs")
    col1, col2 = st.columns(2)
    with col1:
        for idx, (s1, s2, score) in enumerate(pairs[:5]):
            label1 = s1[:40] + "…" if len(s1) > 40 else s1
            label2 = s2[:40] + "…" if len(s2) > 40 else s2
            st.metric(
                label=f"#{idx+1}  {label1}  ↔  {label2}",
                value=f"{score:.4f}"
            )

    st.divider()

    # ── GRAPH 1: Bar Chart – top similar pairs ────────────────
    st.subheader("📈 Graph 1 — Bar Chart: Top Similar Pairs")

    top_n   = min(8, len(pairs))
    labels  = [
        f"{p[0][:20]}…\n↔ {p[1][:20]}…" if len(p[0]) > 20 or len(p[1]) > 20
        else f"{p[0]}\n↔ {p[1]}"
        for p in pairs[:top_n]
    ]
    scores  = [p[2] for p in pairs[:top_n]]
    colors  = ["#2ecc71" if s >= 0.7 else "#3498db" if s >= 0.4 else "#e74c3c" for s in scores]

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.barh(labels[::-1], scores[::-1], color=colors[::-1], edgecolor='white')
    ax1.set_xlabel("Cosine Similarity Score", fontsize=11)
    ax1.set_title("Top Sentence Pairs by Similarity Score", fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1)
    for bar, score in zip(bars, scores[::-1]):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{score:.4f}", va='center', fontsize=9)
    ax1.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, label='0.5 threshold')
    ax1.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    # ── GRAPH 2: Heatmap ──────────────────────────────────────
    st.subheader("🌡️ Graph 2 — Heatmap: Pairwise Similarity Matrix")

    short_labels = [s[:25] + "…" if len(s) > 25 else s for s in sentences]

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=short_labels,
        yticklabels=short_labels,
        linewidths=0.5,
        vmin=0, vmax=1,
        ax=ax2
    )
    ax2.set_title("Pairwise Cosine Similarity Heatmap", fontsize=13, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── GRAPH 3: 2D PCA Embedding Plot ───────────────────────
    st.subheader("🗺️ Graph 3 — 2D PCA Embedding Plot")

    pca   = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    fig3, ax3 = plt.subplots(figsize=(9, 6))
    scatter = ax3.scatter(
        coords[:, 0], coords[:, 1],
        c=range(n), cmap='tab10', s=120, edgecolors='white', linewidths=1.5, zorder=3
    )
    for i, label in enumerate(short_labels):
        ax3.annotate(
            f"{i+1}. {label}",
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points", xytext=(8, 5),
            fontsize=8, color='black'
        )
    ax3.set_title("2D PCA Projection of Sentence Embeddings", fontsize=13, fontweight='bold')
    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=10)
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.divider()

    # ── Paul's Critical Thinking Standards ───────────────────
    st.subheader("🧠 Critical Thinking Analysis (Paul's Standards)")

    top   = pairs[0]
    bot   = pairs[-1]
    t1    = top[0][:50]
    t2    = top[1][:50]
    b1    = bot[0][:50]
    b2    = bot[1][:50]
    tscore = top[2]
    bscore = bot[2]

    standards = {
        "🔵 Clarity": (
            f"The user entered {n} sentences. The model converted each into a "
            f"384-dimensional vector. Cosine similarity (0=unrelated, 1=identical) "
            f"was computed for every pair and displayed as scores, a heatmap, and a 2D map."
        ),
        "🟢 Accuracy": (
            f"Model used: **all-MiniLM-L6-v2** (sentence-transformers). "
            f"No claims beyond what the model produces are made. "
            f"Scores are raw cosine similarities — no post-processing applied."
        ),
        "🟡 Precision": (
            f"Highest similarity: **{tscore:.4f}** between *'{t1}'* and *'{t2}'*. "
            f"Lowest similarity: **{bscore:.4f}** between *'{b1}'* and *'{b2}'*. "
            f"Exact 4-decimal scores are shown throughout, not vague labels."
        ),
        "🟠 Relevance": (
            f"All three graphs directly reflect the computed similarity values. "
            f"The bar chart ranks pairs, the heatmap shows the full matrix, "
            f"and the PCA plot shows geometric closeness matching the scores."
        ),
        "🔴 Logic": (
            f"The top pair scored {tscore:.4f} because both sentences share similar "
            f"semantic meaning in the embedding space. Sentences with overlapping "
            f"concepts cluster together in the PCA plot, confirming the scores."
        ),
        "🟣 Significance": (
            f"The most important finding is the highest-scoring pair "
            f"(score = {tscore:.4f}). Scores above 0.70 indicate strong semantic "
            f"similarity, suggesting the sentences convey closely related ideas."
        ),
        "⚪ Fairness": (
            f"**Limitation:** all-MiniLM-L6-v2 is optimised for English and may "
            f"produce lower-quality embeddings for other languages, domain-specific "
            f"jargon, or very short single-word inputs. It reflects biases present "
            f"in its training corpus."
        ),
    }

    for standard, explanation in standards.items():
        with st.expander(standard, expanded=True):
            st.markdown(explanation)

    st.divider()
    st.caption(
        "App built for NLP Lab Quiz · Model: all-MiniLM-L6-v2 · "
        "No preprocessing · No training · Free pretrained model only"
    )
