# ================================================================
#  NLP Quiz – Text/Word Similarity App
#  Model  : all-MiniLM-L6-v2  (free, no training, no preprocessing)
#  Deploy : Streamlit Community Cloud
# ================================================================

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
    page_title="NLP Similarity Explorer",
    page_icon="🔍",
    layout="wide"
)

# ── Custom CSS with Interactive Colors ──────────────────────
st.markdown("""
<style>
    /* Main gradient background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Card with interactive hover */
    .card {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(255,255,255,0.3);
        transition: all 0.3s ease;
        margin-bottom: 1.2rem;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(31, 38, 135, 0.25);
    }
    
    /* Header gradient */
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
    }
    .header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }
    .header .tag {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(5px);
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        margin-right: 0.4rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Interactive buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.5);
    }
    .stButton > button:active {
        transform: scale(0.95);
    }
    
    /* Query highlight */
    .query-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 0.8rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.3rem 0;
    }
    .query-box strong {
        color: #667eea;
    }
    
    /* Candidate items */
    .candidate-item {
        padding: 0.5rem 0.8rem;
        margin: 0.2rem 0;
        border-radius: 6px;
        background: #f8f9fa;
        border-left: 3px solid #dee2e6;
        transition: all 0.2s ease;
        color: #2d3436;
        font-size: 0.9rem;
    }
    .candidate-item:hover {
        background: #e9ecef;
        border-left-color: #667eea;
        transform: translateX(4px);
    }
    .candidate-item .num {
        display: inline-block;
        width: 24px;
        height: 24px;
        line-height: 24px;
        text-align: center;
        background: #667eea;
        color: white;
        border-radius: 50%;
        font-size: 0.7rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }
    
    /* Metric boxes */
    .metric-box {
        background: white;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    .metric-box .value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-box .label {
        font-size: 0.7rem;
        color: #6c757d;
        margin-top: 0.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Progress bars */
    .progress-container {
        width: 100%;
        height: 6px;
        background: #e9ecef;
        border-radius: 3px;
        overflow: hidden;
        margin-top: 0.2rem;
    }
    .progress-bar-custom {
        height: 100%;
        border-radius: 3px;
        transition: width 1s ease;
    }
    
    /* Result rows */
    .result-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0f2f6;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }
    .result-row:hover {
        background: #f8f9fa;
        padding-left: 0.5rem;
        border-radius: 4px;
    }
    .result-row .rank {
        color: #6c757d;
        width: 30px;
        font-weight: 600;
    }
    .result-row .text {
        flex: 1;
        color: #2d3436;
    }
    .result-row .score {
        font-weight: 700;
        color: #667eea;
        width: 80px;
        text-align: right;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #2d3436;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
    }
    
    .divider {
        border: none;
        border-top: 2px dashed #dee2e6;
        margin: 1.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 0.75rem;
        padding: 1rem 0;
        border-top: 1px solid #dee2e6;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🔍 NLP Text Similarity Explorer</h1>
    <p>Powered by <strong>all-MiniLM-L6-v2</strong> — free pretrained sentence embedding model</p>
    <div style="margin-top: 0.6rem;">
        <span class="tag">✨ No Preprocessing</span>
        <span class="tag">🎯 No Training</span>
        <span class="tag">🚀 No Paid API</span>
        <span class="tag">📊 Interactive Visuals</span>
        <span class="tag">🧊 3D Visualization</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

with st.spinner("🔄 Loading pretrained model (all-MiniLM-L6-v2)…"):
    model = load_model()

st.success("✅ Model loaded: all-MiniLM-L6-v2")

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✍️ Input")
    st.markdown("Enter one sentence/word per line (min 2, max 10).")
    
    default_text = (
        "Artificial intelligence is changing the world\n"
        "Machine learning helps computers learn from data\n"
        "Deep learning uses neural networks\n"
        "Natural language processing handles text\n"
        "Computer vision processes images\n"
        "Robotics involves autonomous machines"
    )
    
    raw_input = st.text_area(
        "",
        value=default_text,
        height=220,
        label_visibility="collapsed"
    )
    
    run_btn = st.button("▶ Compute Similarity", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    **Model Details:**  
    `all-MiniLM-L6-v2`  
    *Sentence-Transformers*  
    *Free & Pretrained*
    """)

# ── Main logic ────────────────────────────────────────────────
if run_btn:

    sentences = [line.strip() for line in raw_input.strip().split("\n") if line.strip()]

    if len(sentences) < 2:
        st.error("⚠️ Please enter at least 2 sentences.")
        st.stop()

    if len(sentences) > 10:
        sentences = sentences[:10]
        st.warning("⚠️ Using first 10 sentences only.")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📋 Input Sentences")
    
    query = sentences[0]
    candidates = sentences[1:]
    
    col_q, col_c = st.columns([1, 2])
    with col_q:
        st.markdown("**🔵 Query:**")
        st.markdown(f'<div class="query-box"><strong>{query}</strong></div>', unsafe_allow_html=True)
    
    with col_c:
        st.markdown("**📌 Candidates:**")
        for i, c in enumerate(candidates, 1):
            st.markdown(f'<div class="candidate-item"><span class="num">{i}</span>{c}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Embed ─────────────────────────────────────────────────
    with st.spinner("🧠 Computing embeddings..."):
        all_texts = [query] + candidates
        embeddings = model.encode(all_texts)
        
        query_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        sim_matrix = cosine_similarity(embeddings)
        
        results = []
        for i, (cand, sim) in enumerate(zip(candidates, similarities)):
            results.append({
                "Rank": i + 1,
                "Candidate": cand,
                "Similarity Score": sim
            })
        results.sort(key=lambda x: x["Similarity Score"], reverse=True)

    n = len(all_texts)
    
    # ── Metrics ─────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Summary")
    
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
    st.markdown(f"### 🏆 Top {len(results)} Results")
    
    for i, r in enumerate(results, 1):
        score_pct = r["Similarity Score"] * 100
        bar_width = min(score_pct, 100)
        
        # Color based on score
        if r["Similarity Score"] >= 0.7:
            bar_color = "linear-gradient(90deg, #00b894, #00cec9)"
        elif r["Similarity Score"] >= 0.4:
            bar_color = "linear-gradient(90deg, #fdcb6e, #e17055)"
        else:
            bar_color = "linear-gradient(90deg, #fd79a8, #e17055)"
        
        st.markdown(f"""
        <div class="result-row">
            <span class="rank">#{i}</span>
            <span class="text">{r['Candidate']}</span>
            <span class="score">{r['Similarity Score']:.4f}</span>
        </div>
        <div class="progress-container">
            <div class="progress-bar-custom" style="width:{bar_width}%; background:{bar_color};"></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # ── GRAPH 1: Bar Chart ─────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Similarity Scores")

    top_n = min(8, len(results))
    labels = [f"#{r['Rank']} {r['Candidate'][:30]}..." if len(r['Candidate']) > 30 else f"#{r['Rank']} {r['Candidate']}" for r in results[:top_n]]
    scores = [r['Similarity Score'] for r in results[:top_n]]
    
    # Gradient colors
    colors = ["#00b894" if s >= 0.7 else "#fdcb6e" if s >= 0.4 else "#fd79a8" for s in scores]

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.barh(labels[::-1], scores[::-1], color=colors[::-1], edgecolor='white', height=0.6)
    ax1.set_xlabel("Cosine Similarity Score", fontsize=11, color='#2d3436')
    ax1.set_title("Top Similarity Scores", fontsize=14, fontweight='700', color='#2d3436')
    ax1.set_xlim(0, 1)
    ax1.set_facecolor('#f8f9fa')
    ax1.grid(True, alpha=0.2, axis='x')
    
    for bar, score in zip(bars, scores[::-1]):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{score:.4f}", va='center', fontsize=9, fontweight='600')
    ax1.axvline(0.5, color='#636e72', linestyle='--', linewidth=1, alpha=0.5, label='0.5 threshold')
    ax1.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── GRAPH 2: Heatmap ──────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🌡️ Similarity Matrix (Heatmap)")

    short_labels = [s[:15] + "…" if len(s) > 15 else s for s in all_texts]

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=short_labels,
        yticklabels=short_labels,
        linewidths=0.5,
        vmin=0, vmax=1,
        ax=ax2,
        cbar_kws={'shrink': 0.8}
    )
    ax2.set_title("Pairwise Cosine Similarity Heatmap", fontsize=14, fontweight='700')
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── GRAPH 3: 2D PCA ──────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗺️ 2D PCA Projection")

    pca2 = PCA(n_components=2, random_state=42)
    coords2 = pca2.fit_transform(embeddings)

    fig3, ax3 = plt.subplots(figsize=(9, 6))
    
    # Gradient colors for points
    colors_pca = ['#6c5ce7' if i == 0 else '#00b894' if i == 1 else '#fdcb6e' if i == 2 else '#fd79a8' if i == 3 else '#0984e3' for i in range(len(all_texts))]
    sizes = [150 if i == 0 else 100 for i in range(len(all_texts))]

    scatter = ax3.scatter(coords2[:, 0], coords2[:, 1], c=colors_pca, s=sizes, edgecolors='white', linewidths=2, zorder=3)
    
    for i, label in enumerate(short_labels):
        prefix = "🔵 Query: " if i == 0 else f"{i}. "
        ax3.annotate(f"{prefix}{label}", (coords2[i, 0], coords2[i, 1]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8, fontweight='600' if i == 0 else 'normal')
    
    ax3.set_title("2D PCA Projection of Sentence Embeddings", fontsize=14, fontweight='700')
    ax3.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
    ax3.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
    ax3.set_facecolor('#f8f9fa')
    ax3.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── GRAPH 4: 3D PCA ────────────────────────────────────── (ADDED)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧊 3D PCA Projection")
    st.caption("🔄 Rotate the view using the sliders below")

    col_el, col_az = st.columns(2)
    elev = col_el.slider("Elevation angle", min_value=0, max_value=90, value=25, step=5, key="elev")
    azim = col_az.slider("Azimuth angle", min_value=0, max_value=360, value=45, step=5, key="azim")

    # Need at least 3 components; cap at n
    n_comp = min(3, n)
    pca3 = PCA(n_components=n_comp, random_state=42)
    coords3_raw = pca3.fit_transform(embeddings)

    # Pad to 3 columns if fewer sentences than 3
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

    colors3 = [cmap(i % 10) for i in range(n)]
    colors3[0] = (0.4, 0.36, 0.91)  # #6c5ce7 for query

    fig4 = plt.figure(figsize=(10, 7))
    ax4 = fig4.add_subplot(111, projection='3d')

    for i in range(n):
        ax4.scatter(
            coords3[i, 0], coords3[i, 1], coords3[i, 2],
            color=colors3[i], s=120 if i == 0 else 100,
            edgecolors='white', linewidths=1.5, zorder=3
        )
        ax4.text(
            coords3[i, 0], coords3[i, 1], coords3[i, 2],
            f" {i+1}. {short_labels[i]}",
            fontsize=8, color='black', fontweight='bold' if i == 0 else 'normal'
        )

    # Draw lines between points scaled by similarity
    for i in range(n):
        for j in range(i+1, n):
            sim = sim_matrix[i][j]
            if sim >= 0.4:
                ax4.plot(
                    [coords3[i, 0], coords3[j, 0]],
                    [coords3[i, 1], coords3[j, 1]],
                    [coords3[i, 2], coords3[j, 2]],
                    color='gray', alpha=sim * 0.6, linewidth=sim * 2
                )

    ax4.set_title("3D PCA Projection of Sentence Embeddings", fontsize=14, fontweight='700', pad=15)
    ax4.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=9)
    ax4.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=9)
    ax4.set_zlabel(f"PC3 ({ev[2]*100:.1f}%)", fontsize=9)
    ax4.view_init(elev=elev, azim=azim)
    ax4.grid(True, alpha=0.2)

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=colors3[i], label=f"{i+1}. {short_labels[i]}") for i in range(n)]
    ax4.legend(handles=patches, loc='upper left', fontsize=7, bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.markdown("""
    <div class="info-box">
        💡 <strong>How to read the 3D plot:</strong> Points close together in 3D space have similar semantic meaning. 
        Gray lines connect pairs with similarity ≥ 0.40 — thicker and darker lines = higher similarity. 
        Adjust the sliders to rotate the view.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Paul's Critical Thinking Standards ───────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Critical Thinking Analysis")
    st.markdown("*(Paul's Critical Thinking Standards)*")

    top = results[0]
    bot = results[-1]
    t1 = top['Candidate'][:50]
    t2 = query[:50]
    b1 = bot['Candidate'][:50]
    b2 = query[:50]
    tscore = top['Similarity Score']
    bscore = bot['Similarity Score']

    standards = {
        "🔵 Clarity": (
            f"The user entered {n} sentences. The model converted each into a "
            f"384-dimensional vector. Cosine similarity (0=unrelated, 1=identical) "
            f"was computed for every pair and displayed as scores, a heatmap, a 2D PCA plot, and a 3D PCA plot."
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
            f"All four graphs directly reflect the computed similarity values. "
            f"The bar chart ranks pairs, the heatmap shows the full matrix, "
            f"the 2D PCA shows geometric closeness, and the 3D PCA adds a third "
            f"dimension for deeper spatial understanding of relationships."
        ),
        "🔴 Logic": (
            f"The top pair scored {tscore:.4f} because both sentences share similar "
            f"semantic meaning in the embedding space. Sentences with overlapping "
            f"concepts cluster together in both PCA plots, confirming the scores. "
            f"The 3D plot lines visually confirm which pairs are most related."
        ),
        "🟣 Significance": (
            f"The most important finding is the highest-scoring pair "
            f"(score = {tscore:.4f}). Scores above 0.70 indicate strong semantic "
            f"similarity. The 3D plot makes clusters easier to spot than 2D alone."
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
    
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Info Box ──────────────────────────────────────────────
    st.markdown("""
    <div class="info-box">
        💡 <strong>How to interpret:</strong> Scores closer to 1.0 indicate stronger semantic similarity. 
        The <strong>PCA plots</strong> (2D and 3D) show how sentences are positioned in space based on their meaning.
        Closer points = more similar meanings. The <strong>3D view</strong> provides additional depth for understanding relationships.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built for NLP Lab Quiz · Model: all-MiniLM-L6-v2 · 
    No preprocessing · No training · Free pretrained model only
</div>
""", unsafe_allow_html=True)
