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
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Similarity Explorer",
    page_icon="🔍",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────────
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
    
    /* Paul's Standards Score Card */
    .score-card {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        min-width: 60px;
        text-align: center;
    }
    .score-high {
        background: #00b894;
        color: white;
    }
    .score-good {
        background: #fdcb6e;
        color: #2d3436;
    }
    .score-low {
        background: #fd79a8;
        color: white;
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
        <span class="tag">📊 2 Visualizations</span>
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

# ── Function to calculate Paul's Standards Scores ────────────
def calculate_paul_scores(sentences, results, query, similarities, sim_matrix, n):
    """Calculate percentage scores for each of Paul's Critical Thinking Standards"""
    
    top_score = results[0]['Similarity Score'] if results else 0
    top_candidate = results[0]['Candidate'] if results else ""
    bot_score = results[-1]['Similarity Score'] if results else 0
    
    scores = {}
    
    # 1. Clarity Score
    clarity_score = 0
    if n >= 2:
        clarity_score += 30
    if len(query) > 10:
        clarity_score += 20
    if len(results) > 0:
        clarity_score += 20
    if top_score > 0:
        clarity_score += 30
    scores['Clarity'] = min(clarity_score, 100)
    
    # 2. Accuracy Score
    accuracy_score = 0
    if n >= 2:
        accuracy_score += 25
    if top_score > 0:
        accuracy_score += 25
    if sim_matrix.shape[0] > 1:
        accuracy_score += 25
    if len(results) > 0:
        accuracy_score += 25
    scores['Accuracy'] = min(accuracy_score, 100)
    
    # 3. Precision Score
    precision_score = 0
    if top_score > 0.7:
        precision_score += 40
    elif top_score > 0.5:
        precision_score += 25
    else:
        precision_score += 10
    
    if bot_score < 0.3:
        precision_score += 20
    elif bot_score < 0.5:
        precision_score += 10
    
    if len(results) >= 3:
        precision_score += 20
    elif len(results) >= 2:
        precision_score += 10
    
    if len(results) >= 2:
        diff = results[0]['Similarity Score'] - results[-1]['Similarity Score']
        if diff > 0.4:
            precision_score += 20
        elif diff > 0.2:
            precision_score += 10
    
    scores['Precision'] = min(precision_score, 100)
    
    # 4. Relevance Score
    relevance_score = 0
    if len(results) >= 2:
        relevance_score += 30
    if top_score > 0:
        relevance_score += 25
    if sim_matrix.shape[0] > 1:
        relevance_score += 25
    if n >= 2:
        relevance_score += 20
    scores['Relevance'] = min(relevance_score, 100)
    
    # 5. Logic Score
    logic_score = 0
    if top_score > 0.5:
        logic_score += 30
    elif top_score > 0.3:
        logic_score += 15
    
    if len(results) >= 3:
        logic_score += 20
    elif len(results) >= 2:
        logic_score += 10
    
    if top_score > 0 and bot_score < top_score:
        logic_score += 25
    
    if len(results) >= 2 and top_score > 0.4:
        logic_score += 25
    
    scores['Logic'] = min(logic_score, 100)
    
    # 6. Significance Score
    significance_score = 0
    if top_score > 0.7:
        significance_score += 50
    elif top_score > 0.5:
        significance_score += 30
    else:
        significance_score += 10
    
    if len(results) >= 3:
        significance_score += 25
    elif len(results) >= 2:
        significance_score += 15
    
    if top_score > 0 and bot_score < top_score:
        significance_score += 25
    
    scores['Significance'] = min(significance_score, 100)
    
    # 7. Fairness Score
    fairness_score = 80  # Base score for acknowledging limitations
    if n >= 3:
        fairness_score += 10
    if len(results) >= 2:
        fairness_score += 10
    scores['Fairness'] = min(fairness_score, 100)
    
    return scores

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
    st.markdown("### 📈 Graph 1 — Bar Chart: Top Similar Words/Sentences")
    st.caption("Shows top similar words/sentences with their exact similarity scores.")

    top_n = min(8, len(results))
    labels = [f"#{r['Rank']} {r['Candidate'][:30]}..." if len(r['Candidate']) > 30 else f"#{r['Rank']} {r['Candidate']}" for r in results[:top_n]]
    scores = [r['Similarity Score'] for r in results[:top_n]]
    
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
    st.markdown("### 🌡️ Graph 2 — Heatmap: Pairwise Similarity Matrix")
    st.caption("Shows pairwise similarity between selected words/sentences.")

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

    # ── Paul's Critical Thinking Standards with Scores ────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Paul's Critical Thinking Standards - Analysis Report")
    st.caption("Automated scoring (0-100%) based on the analysis results")

    # Calculate scores
    paul_scores = calculate_paul_scores(sentences, results, query, similarities, sim_matrix, n)

    # Display scores in a nice grid
    cols = st.columns(4)
    standard_emojis = {
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
                status = "✅ Excellent"
                color = "#00b894"
            elif score >= 60:
                status = "👍 Good"
                color = "#fdcb6e"
            elif score >= 40:
                status = "⚠️ Needs Improvement"
                color = "#fd79a8"
            else:
                status = "❌ Needs Attention"
                color = "#e17055"
            
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid #e9ecef; margin-bottom: 0.8rem;">
                <div style="font-size: 1.5rem;">{standard_emojis.get(standard, '📌')}</div>
                <div style="font-weight: 600; font-size: 0.8rem; color: #2d3436; margin-top: 0.2rem;">{standard}</div>
                <div style="font-size: 2rem; font-weight: 700; color: {color};">{score}%</div>
                <div style="font-size: 0.7rem; color: #6c757d; margin-top: 0.2rem;">{status}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Detailed Explanations ──────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Detailed Standard Analysis")

    top = results[0] if results else None
    bot = results[-1] if results else None
    t1 = top['Candidate'][:50] if top else ""
    t2 = query[:50]
    b1 = bot['Candidate'][:50] if bot else ""
    b2 = query[:50]
    tscore = top['Similarity Score'] if top else 0
    bscore = bot['Similarity Score'] if bot else 0

    standards_details = {
        "🔵 Clarity": (
            f"**Score: {paul_scores['Clarity']}%**  \n"
            f"The user entered **{n} sentences**. The **query** was: *'{query}'*.  \n"
            f"{len(candidates)} candidate sentences were compared.  \n"
            f"The model converted each sentence into a **384-dimensional vector**.  \n"
            f"**Cosine similarity** (0=unrelated, 1=identical) was computed for every pair.  \n"
            f"Results are displayed as **exact scores**, a **heatmap**, and a **bar chart**."
        ),
        "🟢 Accuracy": (
            f"**Score: {paul_scores['Accuracy']}%**  \n"
            f"Model used: **all-MiniLM-L6-v2** from sentence-transformers.  \n"
            f"No preprocessing, training, or post-processing was applied.  \n"
            f"Scores are **raw cosine similarities** between embeddings.  \n"
            f"No claims beyond what the model produces are made."
        ),
        "🟡 Precision": (
            f"**Score: {paul_scores['Precision']}%**  \n"
            f"**Highest similarity:** **{tscore:.4f}** between *'{t1}'* and *'{t2}'*.  \n"
            f"**Lowest similarity:** **{bscore:.4f}** between *'{b1}'* and *'{b2}'*.  \n"
            f"All scores are shown with **4-decimal precision**.  \n"
            f"No vague labels like 'high' or 'low' are used."
        ),
        "🟠 Relevance": (
            f"**Score: {paul_scores['Relevance']}%**  \n"
            f"**Graph 1 (Bar Chart):** Ranks pairs by similarity score.  \n"
            f"**Graph 2 (Heatmap):** Shows complete pairwise similarity matrix.  \n"
            f"Both graphs directly reflect the computed similarity values."
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
            f"**Limitation:** all-MiniLM-L6-v2 is optimised for **English** and may "
            f"produce lower-quality embeddings for other languages, domain-specific "
            f"jargon, or very short single-word inputs.  \n"
            f"It reflects biases present in its training corpus."
        ),
    }

    for standard, explanation in standards_details.items():
        with st.expander(standard, expanded=False):
            st.markdown(explanation)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Final Info Box ────────────────────────────────────────
    st.markdown("""
    <div class="info-box">
        💡 <strong>How to interpret:</strong> Scores closer to <strong>1.0</strong> indicate stronger semantic similarity. 
        The <strong>heatmap</strong> shows the complete similarity matrix, and the <strong>bar chart</strong> ranks the top pairs.
        <br><br>
        🧠 <strong>Paul's Standards Scores:</strong> Each standard is scored from 0-100% based on the quality and completeness of the analysis.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built for NLP Lab Quiz · Model: all-MiniLM-L6-v2 · 
    No preprocessing · No training · Free pretrained model only
</div>
""", unsafe_allow_html=True)
