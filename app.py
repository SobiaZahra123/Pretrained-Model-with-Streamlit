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

# ── Custom CSS for better UI ─────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header */
    .neuro-header {
        background: linear-gradient(135deg, #4A90D9 0%, #357ABD 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(74, 144, 217, 0.3);
    }
    .neuro-header h1 {
        margin: 0;
        font-weight: 600;
        font-size: 2rem;
        letter-spacing: -0.5px;
    }
    .neuro-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }
    
    /* Badge style */
    .badge {
        display: inline-block;
        background: #e9ecef;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        color: #495057;
        margin-right: 0.4rem;
    }
    
    /* Card style */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e9ecef;
        margin-bottom: 1.5rem;
    }
    .card-title {
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Buttons */
    .stButton > button {
        background: #4A90D9;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #357ABD;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74, 144, 217, 0.4);
    }
    
    /* Metrics */
    .metric-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-box .value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4A90D9;
    }
    .metric-box .label {
        font-size: 0.75rem;
        color: #6c757d;
        margin-top: 0.2rem;
    }
    
    /* Divider */
    .custom-divider {
        border: none;
        border-top: 2px solid #e9ecef;
        margin: 2rem 0;
    }
    
    /* Info box */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #4A90D9;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="neuro-header">
    <h1>🧠 NeuroSim</h1>
    <p>A Semantic Similarity Explorer powered by a free pretrained Sentence-Transformer model</p>
    <div style="margin-top: 0.6rem;">
        <span class="badge">all-MiniLM-L6-v2</span>
        <span class="badge">Sentence-Transformers</span>
        <span class="badge">3D + 2D Visualizations</span>
        <span class="badge">Free & Pretrained</span>
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
    st.markdown("### ⚙️ Settings")
    st.markdown("Enter one item per line. The first line is treated as the **query**, remaining lines are compared against it.")
    
    # Load sample data button
    if st.button("📥 Load sample data", use_container_width=True):
        sample_text = """Machine learning is fascinating
I enjoy studying artificial intelligence
The weather is nice today
Deep learning models are powerful
I went to the market to buy vegetables"""
        st.session_state.sample_text = sample_text
        st.rerun()
    
    st.markdown("---")
    
    # Top-N results
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
st.markdown("### 📝 Step 1 — Enter Text")
st.markdown("Enter your **query** (first line) and **comparison items** (one per line):")

# Get sample text from session state if available
sample_text = st.session_state.get('sample_text', """Machine learning is fascinating
I enjoy studying artificial intelligence
The weather is nice today
Deep learning models are powerful
I went to the market to buy vegetables""")

text_input = st.text_area(
    "",
    value=sample_text,
    height=180,
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
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Results")
    
    # Display query and candidates
    col_q, col_c = st.columns([1, 2])
    with col_q:
        st.markdown("**Query:**")
        st.info(f"💡 {query}")
    with col_c:
        st.markdown("**Candidates:**")
        for i, c in enumerate(candidates, 1):
            st.markdown(f"{i}. {c}")
    
    # ── Compute embeddings ─────────────────────────────────────
    with st.spinner("Computing embeddings and similarities..."):
        # Encode all texts
        all_texts = [query] + candidates
        embeddings = model.encode(all_texts)
        
        # Compute similarities
        query_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # Create results
        results = []
        for i, (cand, sim) in enumerate(zip(candidates, similarities)):
            results.append({
                "Rank": i + 1,
                "Candidate": cand,
                "Similarity Score": sim
            })
        
        results.sort(key=lambda x: x["Similarity Score"], reverse=True)
        
        # Similarity matrix for all pairs (for heatmap)
        sim_matrix = cosine_similarity(embeddings)
    
    # ── Metrics ─────────────────────────────────────────────────
    st.markdown("#### 📈 Summary")
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
    
    # ── Top Results ────────────────────────────────────────────
    st.markdown(f"#### 🏆 Top {top_n} Results")
    
    # Create a clean table
    df = pd.DataFrame(results[:top_n])
    df["Similarity Score"] = df["Similarity Score"].apply(lambda x: f"{x:.4f}")
    
    # Style the dataframe
    st.dataframe(
        df,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Candidate": st.column_config.TextColumn("Candidate", width="large"),
            "Similarity Score": st.column_config.TextColumn("Score", width="medium")
        },
        hide_index=True,
        use_container_width=True
    )
    
    # ── Bar Chart ──────────────────────────────────────────────
    st.markdown("#### 📊 Similarity Scores")
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    top_results = results[:top_n]
    labels = [f"{r['Rank']}. {r['Candidate'][:40]}..." if len(r['Candidate']) > 40 else f"{r['Rank']}. {r['Candidate']}" for r in top_results]
    scores = [r['Similarity Score'] for r in top_results]
    
    colors = ["#4A90D9" if s >= 0.5 else "#A8C4E0" for s in scores]
    bars = ax1.barh(labels[::-1], scores[::-1], color=colors[::-1], edgecolor='white', height=0.6)
    ax1.set_xlabel("Similarity Score", fontsize=11)
    ax1.set_title("Top Results by Similarity to Query", fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1)
    for bar, score in zip(bars, scores[::-1]):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{score:.4f}", va='center', fontsize=9)
    ax1.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='0.5 threshold')
    ax1.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── Graph 2: Heatmap ──────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🌡️ Similarity Matrix (Heatmap)")
    
    short_labels = [t[:15] + "…" if len(t) > 15 else t for t in all_texts]
    
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
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
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── Graph 3: 2D PCA ──────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🗺️ 2D PCA Projection")
    
    pca2 = PCA(n_components=2, random_state=42)
    coords2 = pca2.fit_transform(embeddings)
    
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    colors_pca = ['#4A90D9' if i == 0 else '#A8C4E0' for i in range(len(all_texts))]
    sizes = [150 if i == 0 else 100 for i in range(len(all_texts))]
    
    ax3.scatter(
        coords2[:, 0], coords2[:, 1],
        c=colors_pca, s=sizes,
        edgecolors='white', linewidths=1.5, zorder=3
    )
    for i, label in enumerate(short_labels):
        prefix = "🔵 " if i == 0 else f"{i}. "
        ax3.annotate(
            f"{prefix}{label}",
            (coords2[i, 0], coords2[i, 1]),
            textcoords="offset points", xytext=(8, 5),
            fontsize=8, color='black'
        )
    ax3.set_title("2D PCA Projection of Embeddings", fontsize=13, fontweight='bold')
    ax3.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
    ax3.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
    ax3.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── Graph 4: 3D PCA ──────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🧊 3D PCA Projection")
    st.caption("Rotate the view using sliders below")
    
    col_el, col_az = st.columns(2)
    elev = col_el.slider("Elevation angle", min_value=0, max_value=90, value=25, step=5, key="elev")
    azim = col_az.slider("Azimuth angle", min_value=0, max_value=360, value=45, step=5, key="azim")
    
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
    colors3[0] = (0.29, 0.56, 0.85)  # #4A90D9 for query
    
    fig4 = plt.figure(figsize=(10, 7))
    ax4 = fig4.add_subplot(111, projection='3d')
    
    for i in range(len(all_texts)):
        ax4.scatter(
            coords3[i, 0], coords3[i, 1], coords3[i, 2],
            color=colors3[i], s=100 if i == 0 else 80,
            edgecolors='white', linewidths=1, zorder=3
        )
        ax4.text(
            coords3[i, 0], coords3[i, 1], coords3[i, 2],
            f" {i+1}. {short_labels[i]}",
            fontsize=7.5, color='black'
        )
    
    for i in range(len(all_texts)):
        for j in range(i+1, len(all_texts)):
            sim = sim_matrix[i][j]
            if sim >= 0.4:
                ax4.plot(
                    [coords3[i, 0], coords3[j, 0]],
                    [coords3[i, 1], coords3[j, 1]],
                    [coords3[i, 2], coords3[j, 2]],
                    color='gray', alpha=sim * 0.6, linewidth=sim * 2
                )
    
    ax4.set_title("3D PCA Projection of Embeddings", fontsize=13, fontweight='bold', pad=15)
    ax4.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=9)
    ax4.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=9)
    ax4.set_zlabel(f"PC3 ({ev[2]*100:.1f}%)", fontsize=9)
    ax4.view_init(elev=elev, azim=azim)
    ax4.grid(True, alpha=0.2)
    
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=colors3[i], label=f"{i+1}. {short_labels[i]}") for i in range(len(all_texts))]
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

elif run_btn and not text_input.strip():
    st.warning("Please enter some text to analyze.")

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.8rem; padding: 1rem 0;">
    <span>🧠 NeuroSim · Built with Streamlit & Sentence-Transformers</span>
    <br>
    <span>Model: all-MiniLM-L6-v2 · Free & Pretrained</span>
</div>
""", unsafe_allow_html=True)
