import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroSim - Semantic Similarity Explorer",
    page_icon="🧠",
    layout="wide"
)

# ── Simple CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Simple header */
    .header {
        background: #f0f2f6;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #1f77b4;
    }
    .header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    .header p {
        margin: 0.2rem 0 0 0;
        color: #4a4a5a;
        font-size: 0.9rem;
    }
    .header .tags {
        margin-top: 0.5rem;
    }
    .header .tag {
        display: inline-block;
        background: #e9ecef;
        padding: 0.15rem 0.7rem;
        border-radius: 12px;
        font-size: 0.7rem;
        color: #495057;
        margin-right: 0.3rem;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e9ecef;
        margin-bottom: 1.2rem;
    }
    .card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    .card p {
        margin: 0;
        color: #4a4a5a;
        font-size: 0.85rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #f8f9fa;
    }
    
    /* Buttons */
    .stButton > button {
        background: #1f77b4;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.4rem 1.5rem;
        font-size: 0.85rem;
    }
    .stButton > button:hover {
        background: #155a8a;
    }
    
    /* Query box */
    .query-box {
        background: #f8f9fa;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        border-left: 3px solid #1f77b4;
        margin: 0.3rem 0;
    }
    .query-box strong {
        color: #1a1a2e;
    }
    
    /* Candidate list */
    .candidate-item {
        padding: 0.3rem 0;
        color: #4a4a5a;
        font-size: 0.9rem;
        border-bottom: 1px solid #f0f2f6;
    }
    .candidate-item:last-child {
        border-bottom: none;
    }
    
    /* Metrics */
    .metric-box {
        background: #f8f9fa;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-box .value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f77b4;
    }
    .metric-box .label {
        font-size: 0.7rem;
        color: #6c757d;
        margin-top: 0.1rem;
    }
    
    /* Results table */
    .result-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #f0f2f6;
        font-size: 0.9rem;
    }
    .result-row .rank {
        color: #6c757d;
        width: 30px;
    }
    .result-row .text {
        flex: 1;
        color: #1a1a2e;
    }
    .result-row .score {
        font-weight: 600;
        color: #1f77b4;
        width: 80px;
        text-align: right;
    }
    
    .divider {
        border: none;
        border-top: 1px solid #e9ecef;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🧠 NeuroSim</h1>
    <p>A Semantic Similarity Explorer powered by a free pretrained Sentence-Transformer model</p>
    <div class="tags">
        <span class="tag">all-MiniLM-L6-v2</span>
        <span class="tag">Sentence-Transformers</span>
        <span class="tag">3D + 2D Visualizations</span>
        <span class="tag">Free & Pretrained</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

with st.spinner("Loading pretrained model (all-MiniLM-L6-v2)…"):
    model = load_model()

st.success("✅ Model loaded: all-MiniLM-L6-v2")

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    
    if st.button("📥 Load sample data", use_container_width=True):
        sample_text = """Machine learning is fascinating
I enjoy studying artificial intelligence
The weather is nice today
Deep learning models are powerful
I went to the market to buy vegetables"""
        st.session_state.sample_text = sample_text
        st.rerun()
    
    top_n = st.slider("Top-N results to show", min_value=1, max_value=10, value=5)
    
    st.markdown("---")
    st.markdown("""
    **Model:** `all-MiniLM-L6-v2`  
    **Library:** `sentence-transformers`  
    **Type:** Free pretrained embedding model (no training used)
    """)

# ── Main content ──────────────────────────────────────────────

# Step 1 — Enter Text
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Step 1 — Enter Text")
st.markdown("Enter your **query** (first line) and **comparison items** (one per line):")

sample_text = st.session_state.get('sample_text', """Machine learning is fascinating
I enjoy studying artificial intelligence
The weather is nice today
Deep learning models are powerful
I went to the market to buy vegetables""")

text_input = st.text_area(
    "",
    value=sample_text,
    height=160,
    label_visibility="collapsed",
    placeholder="Enter your query (first line) and comparison items (one per line)..."
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    run_btn = st.button("▶ Compute Similarity", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Process input ─────────────────────────────────────────────
if run_btn and text_input.strip():
    lines = [line.strip() for line in text_input.strip().split("\n") if line.strip()]
    
    if len(lines) < 2:
        st.warning("Please enter at least 2 lines (query + comparisons).")
        st.stop()
    
    query = lines[0]
    candidates = lines[1:]
    
    # ── Compute embeddings ─────────────────────────────────────
    with st.spinner("Computing embeddings and similarities..."):
        all_texts = [query] + candidates
        embeddings = model.encode(all_texts)
        
        query_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        results = []
        for i, (cand, sim) in enumerate(zip(candidates, similarities)):
            results.append({
                "Rank": i + 1,
                "Candidate": cand,
                "Similarity Score": sim
            })
        results.sort(key=lambda x: x["Similarity Score"], reverse=True)
        
        sim_matrix = cosine_similarity(embeddings)
    
    # ── Display Query and Candidates ──────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col_q, col_c = st.columns([1, 2])
    with col_q:
        st.markdown("**Query:**")
        st.markdown(f'<div class="query-box"><strong>{query}</strong></div>', unsafe_allow_html=True)
    
    with col_c:
        st.markdown("**Candidates:**")
        for i, c in enumerate(candidates, 1):
            st.markdown(f'<div class="candidate-item">{i}. {c}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── Metrics ─────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="value">{len(candidates)}</div>
            <div class="label">Candidates</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="value">{max(similarities):.3f}</div>
            <div class="label">Highest Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="value">{min(similarities):.3f}</div>
            <div class="label">Lowest Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="value">{np.mean(similarities):.3f}</div>
            <div class="label">Average Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── Top Results ────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### Top {top_n} Results")
    
    for i, r in enumerate(results[:top_n], 1):
        score_pct = r["Similarity Score"] * 100
        bar_width = min(score_pct, 100)
        st.markdown(f"""
        <div class="result-row">
            <span class="rank">#{i}</span>
            <span class="text">{r['Candidate']}</span>
            <span class="score">{r['Similarity Score']:.4f}</span>
        </div>
        <div style="width:100%; height:4px; background:#f0f2f6; border-radius:2px; margin-bottom:0.3rem;">
            <div style="width:{bar_width}%; height:100%; background:#1f77b4; border-radius:2px;"></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── Graph 1: Bar Chart ─────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Similarity Scores")
    
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    top_results = results[:top_n]
    labels = [f"#{r['Rank']} {r['Candidate'][:35]}..." if len(r['Candidate']) > 35 else f"#{r['Rank']} {r['Candidate']}" for r in top_results]
    scores = [r['Similarity Score'] for r in top_results]
    
    colors = ["#1f77b4" if s >= 0.5 else "#a0c4e8" for s in scores]
    bars = ax1.barh(labels[::-1], scores[::-1], color=colors[::-1], height=0.5)
    ax1.set_xlabel("Similarity Score", fontsize=10)
    ax1.set_xlim(0, 1)
    for bar, score in zip(bars, scores[::-1]):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{score:.4f}", va='center', fontsize=9)
    ax1.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── Graph 2: Heatmap ──────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Similarity Matrix (Heatmap)")
    
    short_labels = [t[:12] + "…" if len(t) > 12 else t for t in all_texts]
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        linewidths=0.5,
        vmin=0, vmax=1,
        ax=ax2,
        cbar_kws={'shrink': 0.8}
    )
    ax2.set_title("Pairwise Similarity", fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── Graph 3: 2D PCA ──────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 2D PCA Projection")
    
    pca2 = PCA(n_components=2, random_state=42)
    coords2 = pca2.fit_transform(embeddings)
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    colors_pca = ['#1f77b4' if i == 0 else '#a0c4e8' for i in range(len(all_texts))]
    sizes = [120 if i == 0 else 80 for i in range(len(all_texts))]
    
    ax3.scatter(coords2[:, 0], coords2[:, 1], c=colors_pca, s=sizes, edgecolors='white', linewidths=1.5)
    for i, label in enumerate(short_labels):
        prefix = "Q: " if i == 0 else f"{i}. "
        ax3.annotate(f"{prefix}{label}", (coords2[i, 0], coords2[i, 1]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax3.set_title("2D PCA Projection", fontsize=12)
    ax3.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)", fontsize=9)
    ax3.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)", fontsize=9)
    ax3.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── Graph 4: 3D PCA ──────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 3D PCA Projection")
    st.caption("Rotate: Use sliders below")
    
    col_el, col_az = st.columns(2)
    elev = col_el.slider("Elevation", min_value=0, max_value=90, value=25, step=5, key="elev")
    azim = col_az.slider("Azimuth", min_value=0, max_value=360, value=45, step=5, key="azim")
    
    n_comp = min(3, len(all_texts))
    pca3 = PCA(n_components=n_comp, random_state=42)
    coords3_raw = pca3.fit_transform(embeddings)
    
    if coords3_raw.shape[1] < 3:
        pad = np.zeros((coords3_raw.shape[0], 3 - coords3_raw.shape[1]))
        coords3 = np.hstack([coords3_raw, pad])
        ev = list(pca3.explained_variance_ratio_) + [0.0] * (3 - len(pca3.explained_variance_ratio_))
    else:
        coords3 = coords3_raw
        ev = pca3.explained_variance_ratio_
    
    try:
        cmap = plt.get_cmap('tab10')
    except AttributeError:
        cmap = plt.cm.get_cmap('tab10')
    
    colors3 = [cmap(i % 10) for i in range(len(all_texts))]
    colors3[0] = (0.12, 0.47, 0.71)  # #1f77b4 for query
    
    fig4 = plt.figure(figsize=(9, 7))
    ax4 = fig4.add_subplot(111, projection='3d')
    
    for i in range(len(all_texts)):
        ax4.scatter(coords3[i, 0], coords3[i, 1], coords3[i, 2],
                   color=colors3[i], s=90 if i == 0 else 70,
                   edgecolors='white', linewidths=1)
        ax4.text(coords3[i, 0], coords3[i, 1], coords3[i, 2],
                f" {i+1}", fontsize=8, color='black')
    
    for i in range(len(all_texts)):
        for j in range(i+1, len(all_texts)):
            sim = sim_matrix[i][j]
            if sim >= 0.4:
                ax4.plot([coords3[i, 0], coords3[j, 0]],
                        [coords3[i, 1], coords3[j, 1]],
                        [coords3[i, 2], coords3[j, 2]],
                        color='gray', alpha=sim * 0.6, linewidth=sim * 2)
    
    ax4.set_title("3D PCA Projection", fontsize=12)
    ax4.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=8)
    ax4.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=8)
    ax4.set_zlabel(f"PC3 ({ev[2]*100:.1f}%)", fontsize=8)
    ax4.view_init(elev=elev, azim=azim)
    ax4.grid(True, alpha=0.2)
    
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=colors3[i], label=f"{i+1}") for i in range(len(all_texts))]
    ax4.legend(handles=patches, loc='upper left', fontsize=7, bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()
    
    st.caption("💡 Points close together = similar meaning. Lines connect pairs with similarity ≥ 0.40.")
    st.markdown('</div>', unsafe_allow_html=True)

elif run_btn and not text_input.strip():
    st.warning("Please enter some text to analyze.")

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.75rem; padding: 0.5rem 0;">
    NeuroSim · Built with Streamlit & Sentence-Transformers · Model: all-MiniLM-L6-v2
</div>
""", unsafe_allow_html=True)
