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
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import time

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Similarity Explorer",
    page_icon="🔍",
    layout="wide"
)

# ── Custom CSS with Interactive Colors ──────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
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
    .info-box {
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #2d3436;
    }
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 0.75rem;
        padding: 1rem 0;
        border-top: 1px solid #dee2e6;
        margin-top: 2rem;
    }
    .paul-score-card {
        background: rgba(255,255,255,0.8);
        padding: 0.8rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e9ecef;
        margin: 0.3rem 0;
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
        <span class="tag">🧠 Paul's Standards</span>
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

# ── Helper Functions ──────────────────────────────────────────
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
    
    clarity = 30 + (20 if n_texts >= 2 else 0) + (20 if n_texts >= 3 else 0) + (30 if primary_score > 0 else 0)
    scores["Clarity"] = min(clarity, 100)
    
    accuracy = 25 + (25 if n_texts >= 2 else 0) + (25 if primary_score > 0 else 0) + (25 if len(all_scores) > 1 else 0)
    scores["Accuracy"] = min(accuracy, 100)
    
    precision = 10
    if primary_score > 0.7: precision += 40
    elif primary_score > 0.5: precision += 25
    if len(all_scores) >= 2: precision += 20
    if len(all_scores) >= 2 and max(all_scores) - min(all_scores) > 0.3:
        precision += 20
    scores["Precision"] = min(precision, 100)
    
    relevance = 25 + (25 if len(all_scores) > 0 else 0) + (25 if primary_score > 0 else 0) + (25 if n_texts >= 2 else 0)
    scores["Relevance"] = min(relevance, 100)
    
    logic = 15 + (30 if primary_score > 0.5 else 0) + (25 if len(all_scores) >= 2 else 0)
    if len(all_scores) >= 2 and max(all_scores) > 0 and min(all_scores) < max(all_scores):
        logic += 30
    scores["Logic"] = min(logic, 100)
    
    significance = 10 + (50 if primary_score > 0.7 else 30 if primary_score > 0.5 else 0)
    significance += 20 if len(all_scores) >= 2 else 0
    if len(all_scores) >= 2 and max(all_scores) > min(all_scores):
        significance += 20
    scores["Significance"] = min(significance, 100)
    
    fairness = 70 + (15 if n_texts >= 3 else 0) + (15 if len(all_scores) >= 2 else 0)
    scores["Fairness"] = min(fairness, 100)
    
    return scores

# ── GRAPH 1: Paul's Bar Chart ────────────────────────────────
def graph_paul_bar_chart(paul_scores):
    categories = list(paul_scores.keys())
    values = list(paul_scores.values())
    
    colors = ['#4ECB71' if v >= 80 else '#F9A826' if v >= 60 else '#FF6B6B' for v in values]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(categories[::-1], values[::-1], color=colors[::-1], edgecolor='white', height=0.6)
    ax.set_xlabel("Score (%)", fontsize=11, color='#2d3436')
    ax.set_title("Paul's Critical Thinking Standards", fontsize=14, fontweight='700', color='#2d3436')
    ax.set_xlim(0, 100)
    ax.set_facecolor('#f8f9fa')
    ax.axvline(80, color='#4ECB71', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (80%)')
    ax.axvline(60, color='#F9A826', linestyle='--', linewidth=1, alpha=0.5, label='Good (60%)')
    
    for bar, score in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{score}%", va='center', fontsize=9, fontweight='600')
    
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    return fig

# ── GRAPH 2: Paul's Heatmap ──────────────────────────────────
def graph_paul_heatmap(paul_scores):
    categories = list(paul_scores.keys())
    values = list(paul_scores.values())
    data = np.array(values).reshape(1, -1)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    
    ax.set_xticks(range(len(categories)))
    ax.set_yticks([0])
    ax.set_xticklabels(categories, fontsize=9, rotation=0)
    ax.set_yticklabels(['Score'], fontsize=9)
    ax.set_title("Paul's Standards Heatmap", fontsize=14, fontweight='700', color='#2d3436')
    
    for i, v in enumerate(values):
        ax.text(i, 0, f"{v}%", ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    plt.colorbar(im, ax=ax, label='Score (%)')
    plt.tight_layout()
    return fig

# ── GRAPH 3: Paul's Radar Chart (Smaller) ────────────────────
def graph_paul_radar(paul_scores):
    categories = list(paul_scores.keys())
    values = list(paul_scores.values())
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, color='#667eea')
    ax.fill(angles, values, alpha=0.25, color='#667eea')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8, color='#2d3436')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=7, color='#6c757d')
    ax.set_title("Paul's Standards Radar", fontsize=12, fontweight='700', color='#2d3436', pad=15)
    ax.grid(True, alpha=0.3)
    
    for angle, value in zip(angles[:-1], values[:-1]):
        ax.text(angle, value + 5, f"{value}%", ha='center', va='center', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    return fig

# ── GRAPH 4: Dynamic Target Plot ─────────────────────────────
def graph_target_plot(paul_scores):
    """Dynamic target/bullseye plot showing overall score"""
    categories = list(paul_scores.keys())
    values = list(paul_scores.values())
    avg_score = np.mean(values)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Create concentric circles
    circles = [0.2, 0.4, 0.6, 0.8, 1.0]
    colors = ['#FF6B6B', '#F9A826', '#F9A826', '#4ECB71', '#4ECB71']
    for i, (circle, color) in enumerate(zip(circles, colors)):
        circ = plt.Circle((0.5, 0.5), circle/2, color=color, alpha=0.15, ec='gray', linewidth=0.5)
        ax.add_patch(circ)
        if i == 0:
            ax.text(0.5, 0.5 - circle/2 - 0.05, f"{int(circle*100)}%", ha='center', va='top', fontsize=8, color='gray')
    
    # Plot the average score as a point
    target_x = 0.5
    target_y = 0.5
    radius = avg_score / 200
    
    circle = plt.Circle((target_x, target_y), radius, color='#667eea', alpha=0.7, ec='white', linewidth=2)
    ax.add_patch(circle)
    
    ax.text(target_x, target_y, f"{avg_score:.1f}%", ha='center', va='center', 
            fontsize=16, fontweight='bold', color='white')
    
    n = len(categories)
    for i, (cat, val) in enumerate(zip(categories, values)):
        angle = (i / n) * 2 * np.pi - np.pi/2
        x = 0.5 + 0.4 * np.cos(angle)
        y = 0.5 + 0.4 * np.sin(angle)
        ax.text(x, y, f"{cat[:4]}", ha='center', va='center', fontsize=7, 
                color='#2d3436', fontweight='600', 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Overall Critical Thinking Score", fontsize=14, fontweight='700', color='#2d3436')
    
    ax.text(0.5, 0.05, f"Based on {n} Paul's Standards", ha='center', va='center', 
            fontsize=9, color='#6c757d')
    
    plt.tight_layout()
    return fig

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
    
    # ── Display Sentence Scores ──────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Sentence Similarity Scores")
    st.caption("Each candidate's similarity score with the query")
    
    score_data = []
    for r in results:
        label, color = similarity_label(r['Similarity Score'])
        score_data.append({
            "Rank": r['Rank'],
            "Candidate": r['Candidate'][:50] + ("..." if len(r['Candidate']) > 50 else ""),
            "Score": f"{r['Similarity Score']:.4f}",
            "Label": label
        })
    
    df = pd.DataFrame(score_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

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

    # ── Compute Paul's Scores ──────────────────────────────────
    paul_scores = compute_paul_scores(similarities[0], similarities, n)

    # ── PAUL'S STANDARDS SECTION ──────────────────────────────
    st.markdown("---")
    st.markdown("## 🧠 Paul's Critical Thinking Standards Analysis")

    # ── Paul's Standards Cards ─────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Standards Scores")
    
    cols = st.columns(4)
    emojis = {
        "Clarity": "🔵",
        "Accuracy": "🟢",
        "Precision": "🟡",
        "Relevance": "🟠",
        "Logic": "🔴",
        "Significance": "🟣",
        "Fairness": "⚪"
    }

    for idx, (standard, score) in enumerate(paul_scores.items()):
        col = cols[idx % 4]
        with col:
            if score >= 80:
                status, color = "✅ Excellent", "#4ECB71"
            elif score >= 60:
                status, color = "👍 Good", "#F9A826"
            else:
                status, color = "⚠️ Needs Work", "#FF6B6B"
            
            st.markdown(f"""
            <div class="paul-score-card">
                <div style="font-size:1.5rem;">{emojis.get(standard, '📌')}</div>
                <div style="font-weight:600;font-size:0.8rem;color:#2d3436;margin-top:0.2rem;">{standard}</div>
                <div style="font-size:1.8rem;font-weight:700;color:{color};">{score}%</div>
                <div style="font-size:0.7rem;color:#6c757d;">{status}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # ── GRAPH 1: Paul's Bar Chart ──────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Graph 1 — Paul's Standards Bar Chart")
    st.caption("Shows the scores for each of Paul's Critical Thinking Standards.")
    
    fig_bar = graph_paul_bar_chart(paul_scores)
    st.pyplot(fig_bar)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── GRAPH 2: Paul's Heatmap ─────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🌡️ Graph 2 — Paul's Standards Heatmap")
    st.caption("Visualizes the scores for each standard in a heatmap format.")
    
    fig_heat = graph_paul_heatmap(paul_scores)
    st.pyplot(fig_heat)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── GRAPH 3: Paul's Radar Chart (Smaller) ──────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗺️ Graph 3 — Paul's Standards Radar Chart")
    st.caption("Shows the overall profile of critical thinking standards.")
    
    fig_radar = graph_paul_radar(paul_scores)
    st.pyplot(fig_radar)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── GRAPH 4: Dynamic Target Plot ────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Graph 4 — Dynamic Target Plot")
    st.caption("Shows the overall critical thinking score with standards distribution.")
    
    fig_target = graph_target_plot(paul_scores)
    st.pyplot(fig_target)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Detailed Explanations ───────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📋 Detailed Standard Analysis")

    top = results[0] if results else None
    bottom = results[-1] if results else None
    t1 = top['Candidate'][:50] if top else ""
    t2 = query[:50]
    b1 = bottom['Candidate'][:50] if bottom else ""
    b2 = query[:50]
    tscore = top['Similarity Score'] if top else 0
    bscore = bottom['Similarity Score'] if bottom else 0

    standards_details = {
        "🔵 Clarity": (
            f"**Score: {paul_scores['Clarity']}%**  \n"
            f"The user entered **{n} sentences**. The **query** was: *'{query}'*.  \n"
            f"{len(candidates)} candidate sentences were compared.  \n"
            f"The model converted each sentence into a **384-dimensional vector**.  \n"
            f"**Cosine similarity** (0=unrelated, 1=identical) was computed for every pair."
        ),
        "🟢 Accuracy": (
            f"**Score: {paul_scores['Accuracy']}%**  \n"
            f"Model used: **all-MiniLM-L6-v2** from sentence-transformers.  \n"
            f"No preprocessing, training, or post-processing was applied.  \n"
            f"Scores are **raw cosine similarities** between embeddings."
        ),
        "🟡 Precision": (
            f"**Score: {paul_scores['Precision']}%**  \n"
            f"**Highest similarity:** **{tscore:.4f}** between *'{t1}'* and *'{t2}'*.  \n"
            f"**Lowest similarity:** **{bscore:.4f}** between *'{b1}'* and *'{b2}'*.  \n"
            f"All scores are shown with **4-decimal precision**."
        ),
        "🟠 Relevance": (
            f"**Score: {paul_scores['Relevance']}%**  \n"
            f"**Graph 1 (Bar Chart):** Shows Paul's standards scores.  \n"
            f"**Graph 2 (Heatmap):** Visualizes standards scores.  \n"
            f"**Graph 3 (Radar):** Shows overall critical thinking profile.  \n"
            f"**Graph 4 (Target):** Shows overall score with standards distribution."
        ),
        "🔴 Logic": (
            f"**Score: {paul_scores['Logic']}%**  \n"
            f"The top pair scored **{tscore:.4f}** because both sentences share similar "
            f"semantic meaning in the embedding space."
        ),
        "🟣 Significance": (
            f"**Score: {paul_scores['Significance']}%**  \n"
            f"The **most important finding** is the highest-scoring pair "
            f"(score = **{tscore:.4f}**). Scores above **0.70** indicate strong semantic similarity."
        ),
        "⚪ Fairness": (
            f"**Score: {paul_scores['Fairness']}%**  \n"
            f"**Limitation:** all-MiniLM-L6-v2 is optimised for **English** and may "
            f"produce lower-quality embeddings for other languages, domain-specific "
            f"jargon, or very short single-word inputs."
        ),
    }

    for standard, explanation in standards_details.items():
        with st.expander(standard, expanded=False):
            st.markdown(explanation)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Model Info ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ℹ️ Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model:** `all-MiniLM-L6-v2`")
        st.markdown("**Dimension:** 384")
        st.markdown("**License:** Apache 2.0")
    with col2:
        st.markdown("**Framework:** Sentence-Transformers")
        st.markdown("**Type:** Free pretrained embedding model")
        st.markdown("**Max Tokens:** 512")

else:
    # ── Placeholder ──────────────────────────────────────────────
    st.info("👆 Enter your texts above and click **Compute Similarity** to see results.")

    st.markdown("### 📊 What You'll See")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="card" style="text-align:center;padding:1.5rem 1rem;">
            <div style="font-size:2.5rem;">📊</div>
            <h4 style="margin:0.3rem 0;">Bar Chart</h4>
            <p style="font-size:0.85rem;margin:0;">Paul's Standards Scores</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
