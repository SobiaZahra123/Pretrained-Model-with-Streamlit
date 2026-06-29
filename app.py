"""
app.py
======
NLP Semantic Similarity Explorer — Main Streamlit Application

Entry point for the entire app. Run with:
    streamlit run app.py

Requirements: see requirements.txt
Model used  : sentence-transformers/all-MiniLM-L6-v2 (free, Apache 2.0)
No preprocessing · No tokenization · No model training
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import time
import base64
from io import BytesIO

# ──────────────────────────────────────────────────────────────
#  Page Configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Similarity Explorer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_DIM = 384
MODEL_INFO = {
    "name": MODEL_NAME,
    "dimension": MODEL_DIM,
    "type": "Sentence Transformer",
    "tokens": "512",
    "language": "English (multilingual support)",
    "license": "Apache 2.0",
    "paper": "https://arxiv.org/abs/1908.10084",
}

# ──────────────────────────────────────────────────────────────
#  Custom CSS
# ──────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,600;14..32,700&family=JetBrains+Mono&display=swap');

        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .glass {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(24px) saturate(180%);
            -webkit-backdrop-filter: blur(24px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 24px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
            transition: all 0.25s ease;
        }

        .glass:hover {
            border-color: rgba(255, 255, 255, 0.13);
            box-shadow: 0 24px 70px rgba(0,0,0,0.5);
        }

        .stApp {
            background: #0b0b14;
            background-image:
                radial-gradient(ellipse at 10% 20%, rgba(88, 70, 255, 0.09) 0%, transparent 60%),
                radial-gradient(ellipse at 90% 80%, rgba(255, 70, 200, 0.07) 0%, transparent 60%);
        }

        .mono { font-family: 'JetBrains Mono', monospace; }
        .text-muted { color: #8e8ea0; font-size: 0.9rem; }
        .divider { border: 0; border-top: 1px solid rgba(255,255,255,0.07); margin: 2rem 0; }

        h1, h2, h3 { color: #f0f0f8; letter-spacing: -0.02em; }
        p, li, .stMarkdown { color: #c8c8d8; }
        
        .stButton > button {
            background: linear-gradient(135deg, #6C63FF, #5A52D5);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 30px rgba(108, 99, 255, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  Model Loader
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

# ──────────────────────────────────────────────────────────────
#  Utility Functions
# ──────────────────────────────────────────────────────────────
def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current" not in st.session_state:
        st.session_state.current = None
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

def save_to_history(texts, all_scores, primary_score, elapsed_ms):
    st.session_state.history.append({
        "texts": texts,
        "all_scores": all_scores,
        "primary_score": primary_score,
        "elapsed_ms": elapsed_ms,
        "timestamp": time.time(),
    })

def get_history_df():
    if not st.session_state.history:
        return pd.DataFrame()
    records = []
    for i, h in enumerate(st.session_state.history):
        records.append({
            "Run #": i + 1,
            "Primary Score": f"{h['primary_score']:.4f}",
            "Pairs": len(h['all_scores']),
            "Texts": " | ".join(h['texts'][:3]) + ("..." if len(h['texts']) > 3 else ""),
            "Time (ms)": f"{h['elapsed_ms']:.1f}",
        })
    return pd.DataFrame(records)

def clear_history():
    st.session_state.history = []

def format_vector(vec, dims=8):
    return "[" + ", ".join([f"{v:.4f}" for v in vec[:dims]]) + (" ..." if len(vec) > dims else "") + "]"

def similarity_label(score):
    if score >= 0.75:
        return "🔵 Very Similar", "#6C63FF"
    elif score >= 0.55:
        return "🟢 Similar", "#4ECB71"
    elif score >= 0.35:
        return "🟡 Moderately Similar", "#F9A826"
    else:
        return "🔴 Dissimilar", "#FF6B6B"

def compute_paul_scores(primary_score, all_scores, texts):
    scores = {}
    n = len(texts)
    
    clarity = 30
    if n >= 2: clarity += 20
    if n >= 3: clarity += 20
    if primary_score > 0: clarity += 30
    scores["Clarity"] = min(clarity, 100)
    
    accuracy = 25
    if n >= 2: accuracy += 25
    if primary_score > 0: accuracy += 25
    if len(all_scores) > 1: accuracy += 25
    scores["Accuracy"] = min(accuracy, 100)
    
    precision = 10
    if primary_score > 0.7: precision += 40
    elif primary_score > 0.5: precision += 25
    if len(all_scores) >= 2: precision += 20
    if max(all_scores) - min(all_scores) > 0.3: precision += 20
    scores["Precision"] = min(precision, 100)
    
    relevance = 25
    if len(all_scores) > 0: relevance += 25
    if primary_score > 0: relevance += 25
    if n >= 2: relevance += 25
    scores["Relevance"] = min(relevance, 100)
    
    logic = 15
    if primary_score > 0.5: logic += 30
    if len(all_scores) >= 2: logic += 25
    if max(all_scores) > 0 and min(all_scores) < max(all_scores): logic += 30
    scores["Logic"] = min(logic, 100)
    
    significance = 10
    if primary_score > 0.7: significance += 50
    elif primary_score > 0.5: significance += 30
    if len(all_scores) >= 2: significance += 20
    if max(all_scores) > min(all_scores): significance += 20
    scores["Significance"] = min(significance, 100)
    
    fairness = 70
    if n >= 3: fairness += 15
    if len(all_scores) >= 2: fairness += 15
    scores["Fairness"] = min(fairness, 100)
    
    return scores

# ──────────────────────────────────────────────────────────────
#  Graph Functions (Matplotlib only - no plotly)
# ──────────────────────────────────────────────────────────────
def bar_chart_top_similar(top_results):
    """Matplotlib bar chart"""
    if not top_results:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha='center', va='center', color='white')
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

def heatmap_similarity(matrix, texts):
    """Matplotlib heatmap"""
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

def pca_scatter(embeddings, texts):
    """Matplotlib PCA scatter plot"""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#6C63FF' if i == 0 else '#4ECB71' for i in range(len(texts))]
    sizes = [150 if i == 0 else 100 for i in range(len(texts))]
    
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes, edgecolors='white', linewidths=1.5)
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

def radar_chart(scores):
    """Matplotlib radar chart"""
    if not scores:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "No data", ha='center', va='center', color='white')
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

def gauge_chart(score):
    """Matplotlib gauge chart"""
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Background bar
    ax.barh(0, 1, height=0.3, color='#2a2a3e', alpha=0.5)
    
    # Colored bar based on score
    color = '#6C63FF' if score >= 0.5 else '#FF6B6B'
    ax.barh(0, score, height=0.3, color=color, edgecolor='white', linewidth=1)
    
    # Add marker
    ax.axvline(score, color='white', linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Similarity Score', fontsize=10, color='white')
    ax.set_title(f'Score: {score:.4f}', fontsize=14, fontweight='700', color='white')
    ax.set_yticks([])
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0b0b14')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, axis='x', color='gray')
    plt.tight_layout()
    return fig

def distribution_chart(scores):
    """Matplotlib distribution histogram"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=10, color='#6C63FF', edgecolor='white', alpha=0.7)
    ax.set_xlabel('Similarity Score', fontsize=10, color='white')
    ax.set_ylabel('Frequency', fontsize=10, color='white')
    ax.set_title('Score Distribution', fontsize=14, fontweight='700', color='white')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0b0b14')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='gray')
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────
#  UI Components
# ──────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 0.5rem 0;">
            <div style="font-size:2rem;">🔮</div>
            <div style="font-size:1.1rem;font-weight:600;color:#f0f0f8;">NLP Explorer</div>
            <div style="font-size:0.7rem;color:#8e8ea0;">v2.0 · Sentence Transformers</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio(
            "Navigate",
            ["Main", "Graphs", "Critical", "History", "About"],
            index=0,
            key="sidebar_nav",
            label_visibility="collapsed",
        )
        
        st.markdown("---")
        st.caption("⚡ `all-MiniLM-L6-v2`")
        st.caption("🔮 Free · No API · No training")
    
    return page

def render_hero_header():
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0;">
        <h1 style="font-size:2.8rem;font-weight:700;background:linear-gradient(135deg,#6C63FF,#A78BFA);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🔮 Semantic Similarity Explorer
        </h1>
        <p style="color:#a8a8c0;font-size:1.1rem;max-width:640px;margin:0 auto;">
            Compare the semantic meaning of any text using free, pretrained
            <strong>sentence-transformers</strong>.
        </p>
        <p style="color:#7a7a90;font-size:0.85rem;">
            Model: <code>all-MiniLM-L6-v2</code> · 384‑dim · 512 tokens
        </p>
    </div>
    """, unsafe_allow_html=True)

def animated_divider():
    st.markdown('<hr style="border:0;border-top:1px solid rgba(255,255,255,0.07);margin:2rem 0;">', unsafe_allow_html=True)

def render_metrics(primary_score, label, color, n_texts, elapsed_ms, dim):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);padding:1rem;border-radius:12px;text-align:center;">
            <div style="font-size:2rem;font-weight:700;color:{color};">{primary_score:.4f}</div>
            <div style="font-size:0.8rem;color:#8e8ea0;">Primary Similarity</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);padding:1rem;border-radius:12px;text-align:center;">
            <div style="font-size:2rem;font-weight:700;color:#f0f0f8;">{n_texts}</div>
            <div style="font-size:0.8rem;color:#8e8ea0;">Texts Compared</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);padding:1rem;border-radius:12px;text-align:center;">
            <div style="font-size:2rem;font-weight:700;color:#f0f0f8;">{elapsed_ms:.0f}ms</div>
            <div style="font-size:0.8rem;color:#8e8ea0;">Inference Time</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);padding:1rem;border-radius:12px;text-align:center;">
            <div style="font-size:2rem;font-weight:700;color:#f0f0f8;">{dim}</div>
            <div style="font-size:0.8rem;color:#8e8ea0;">Embedding Dimension</div>
        </div>
        """, unsafe_allow_html=True)

def render_success_banner(primary_score, label):
    st.markdown(f"""
    <div style="background:rgba(108,99,255,0.10);border:1px solid rgba(108,99,255,0.2);border-radius:12px;padding:1rem;margin:1rem 0;">
        <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
            <span style="font-size:2rem;">✅</span>
            <span style="font-weight:600;color:#f0f0f8;">Analysis Complete</span>
            <span style="color:#8e8ea0;">Primary similarity score:</span>
            <span style="font-weight:700;color:#6C63FF;font-size:1.2rem;">{primary_score:.4f}</span>
            <span style="font-size:0.9rem;color:#a8a8c0;">({label})</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_top_results(top_results):
    if not top_results:
        st.info("No results to display.")
        return
    
    for i, r in enumerate(top_results, 1):
        label, color = similarity_label(r['score'])
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border-radius:12px;padding:0.8rem 1rem;margin:0.5rem 0;display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-weight:600;color:#6C63FF;">#{i}</span>
                <span style="color:#f0f0f8;margin-left:0.8rem;">{r['text']}</span>
            </div>
            <div style="display:flex;align-items:center;gap:0.8rem;">
                <span style="color:#8e8ea0;font-size:0.9rem;">{label}</span>
                <span style="font-weight:700;color:{color};">{r['score']:.4f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_embedding_preview(emb, label="Embedding"):
    preview = format_vector(emb, 8)
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border-radius:12px;padding:0.8rem 1rem;margin:0.3rem 0;font-family:monospace;font-size:0.85rem;color:#c8c8d8;">
        <span style="color:#6C63FF;">{label}</span> → {preview}
    </div>
    """, unsafe_allow_html=True)

def render_critical_thinking_cards(scores):
    if not scores:
        st.info("Run an analysis first to see Paul's standards scores.")
        return
    
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
    
    for idx, (standard, score) in enumerate(scores.items()):
        col = cols[idx % 4]
        with col:
            if score >= 80:
                status = "✅ Excellent"
                color = "#4ECB71"
            elif score >= 60:
                status = "👍 Good"
                color = "#F9A826"
            elif score >= 40:
                status = "⚠️ Needs Work"
                color = "#FF6B6B"
            else:
                status = "❌ Poor"
                color = "#FF6B6B"
            
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);padding:1rem;border-radius:12px;text-align:center;border:1px solid rgba(255,255,255,0.06);margin:0.3rem 0;">
                <div style="font-size:1.5rem;">{emojis.get(standard, '📌')}</div>
                <div style="font-weight:600;font-size:0.8rem;color:#f0f0f8;margin-top:0.2rem;">{standard}</div>
                <div style="font-size:1.6rem;font-weight:700;color:{color};">{score}%</div>
                <div style="font-size:0.7rem;color:#8e8ea0;">{status}</div>
            </div>
            """, unsafe_allow_html=True)

def render_model_info(info):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Model:** `{info['name']}`")
        st.markdown(f"**Dimension:** {info['dimension']}")
        st.markdown(f"**Type:** {info['type']}")
    with col2:
        st.markdown(f"**Max Tokens:** {info['tokens']}")
        st.markdown(f"**Language:** {info['language']}")
        st.markdown(f"**License:** {info['license']}")

def render_footer():
    st.markdown("""
    <hr style="border:0;border-top:1px solid rgba(255,255,255,0.07);margin:2rem 0;">
    <div style="text-align:center;color:#6a6a80;font-size:0.8rem;padding:1rem 0;">
        <span>🔮 NLP Semantic Similarity Explorer</span>
        <span style="margin:0 0.8rem;">·</span>
        <span>Model: <code>all-MiniLM-L6-v2</code></span>
        <span style="margin:0 0.8rem;">·</span>
        <span>Built for NLP Lab Quiz</span>
        <br>
        <span style="font-size:0.7rem;">Free · Pretrained · No training · No API</span>
    </div>
    """, unsafe_allow_html=True)

def render_performance_stats():
    st.caption(f"🔮 Model: `{MODEL_NAME}` · {MODEL_DIM} dims")

def render_score_bar(score, label):
    pct = int(score * 100)
    color = "#6C63FF" if score >= 0.5 else "#FF6B6B"
    st.markdown(f"""
    <div style="margin:0.3rem 0;">
        <div style="display:flex;justify-content:space-between;font-size:0.85rem;">
            <span style="color:#c8c8d8;">{label}</span>
            <span style="color:#f0f0f8;font-weight:600;">{score:.4f}</span>
        </div>
        <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:6px;overflow:hidden;">
            <div style="background:{color};height:100%;width:{pct}%;border-radius:4px;transition:width 0.5s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def results_to_csv(texts, top_results, matrix):
    import csv
    from io import StringIO
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Type", "Text/Score"])
    writer.writerow(["Query", texts[0]])
    writer.writerow(["Top Results"])
    for r in top_results:
        writer.writerow([f"#{r['rank']}", f"{r['text']} ({r['score']:.4f})"])
    writer.writerow([])
    writer.writerow(["Pairwise Similarity Matrix"])
    writer.writerow([""] + [f"T{i+1}" for i in range(len(texts))])
    for i in range(len(texts)):
        row = [f"T{i+1}"] + [f"{matrix[i][j]:.4f}" for j in range(len(texts))]
        writer.writerow(row)
    return output.getvalue().encode('utf-8')

def history_to_csv():
    df = get_history_df()
    return df.to_csv(index=False).encode('utf-8')

def clipboard_button(text, label):
    b64 = base64.b64encode(text.encode()).decode()
    st.markdown(f"""
    <button onclick="navigator.clipboard.writeText(atob('{b64}'))" 
            style="background:rgba(108,99,255,0.2);border:1px solid rgba(108,99,255,0.3);border-radius:8px;padding:0.3rem 1rem;color:#c8c8d8;cursor:pointer;">
        {label}
    </button>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  Main App
# ──────────────────────────────────────────────────────────────
def main():
    inject_css()
    init_session_state()
    model = load_model()
    
    page = render_sidebar()
    
    if page == "Main":
        render_hero_header()
        animated_divider()
        
        st.markdown("## ✍️ Enter Your Texts")
        st.caption("Enter 2 to 10 texts. The model computes cosine similarity between ALL pairs.")
        
        n_inputs = st.slider("Number of texts to compare", min_value=2, max_value=10, value=3)
        
        default_texts = [
            "Artificial intelligence is transforming the world.",
            "Machine learning enables computers to learn from data.",
            "The cat sat on the mat reading a book.",
            "Deep learning models are a subset of machine learning.",
            "Neural networks are inspired by the human brain.",
        ]
        
        texts = []
        cols_per_row = 2
        flat_cols = []
        for row in range((n_inputs + 1) // cols_per_row):
            cols = st.columns(cols_per_row)
            flat_cols.extend(cols)
        
        for i in range(n_inputs):
            with flat_cols[i]:
                default = default_texts[i] if i < len(default_texts) else ""
                txt = st.text_area(
                    f"Text {i+1}",
                    value=default,
                    height=110,
                    placeholder=f"Enter word, sentence, or paragraph #{i+1}…",
                    key=f"text_input_{i}",
                )
                texts.append(txt)
        
        animated_divider()
        
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
        with ctrl_col1:
            top_k = st.slider("Top-K similar results to show", 2, min(9, n_inputs - 1) if n_inputs > 2 else 1, min(5, n_inputs - 1))
        with ctrl_col2:
            run_btn = st.button("🚀 Analyse Similarity", use_container_width=True)
        with ctrl_col3:
            reset_btn = st.button("🔄 Reset", use_container_width=True)
            if reset_btn:
                for i in range(n_inputs):
                    if f"text_input_{i}" in st.session_state:
                        del st.session_state[f"text_input_{i}"]
                st.rerun()
        
        if run_btn:
            clean_texts = [t.strip() for t in texts if t.strip()]
            if len(clean_texts) < 2:
                st.warning("⚠️ Please enter at least 2 non-empty texts.")
                st.stop()
            
            with st.spinner("🔮 Generating embeddings..."):
                start_time = time.time()
                embeddings = model.encode(clean_texts)
                elapsed_ms = (time.time() - start_time) * 1000
            
            primary_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            label, color = similarity_label(primary_score)
            matrix = cosine_similarity(embeddings)
            
            all_scores = []
            for i in range(len(clean_texts)):
                for j in range(i + 1, len(clean_texts)):
                    all_scores.append(float(matrix[i, j]))
            
            top_results = []
            for i in range(1, len(clean_texts)):
                score = float(matrix[0, i])
                top_results.append({"rank": i, "text": clean_texts[i], "score": score})
            top_results.sort(key=lambda x: x['score'], reverse=True)
            top_results = top_results[:top_k]
            
            paul_scores = compute_paul_scores(primary_score, all_scores, clean_texts)
            
            st.session_state.current = {
                "texts": clean_texts,
                "embeddings": embeddings,
                "matrix": matrix,
                "all_scores": all_scores,
                "top_results": top_results,
                "paul_scores": paul_scores,
                "primary_score": primary_score,
                "label": label,
                "elapsed_ms": elapsed_ms,
            }
            
            render_success_banner(primary_score, label)
            render_metrics(primary_score, label, color, len(clean_texts), elapsed_ms, MODEL_DIM)
            animated_divider()
            
            col_g, col_m = st.columns([1.4, 1])
            with col_g:
                st.markdown("#### 🏆 Top Similar Results")
                render_top_results(top_results)
            with col_m:
                st.mark
