def main():
    inject_css()
    init_session_state()
    
    with st.spinner("Loading model..."):
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
            "Deep learning models are a subset of machine learning.",
        ]
        
        texts = []
        cols = st.columns(2)
        for i in range(n_inputs):
            with cols[i % 2]:
                default = default_texts[i] if i < len(default_texts) else ""
                txt = st.text_area(
                    f"Text {i+1}",
                    value=default,
                    height=100,
                    key=f"text_{i}",
                )
                texts.append(txt)
        
        animated_divider()
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            # ✅ FIXED: Proper bounds checking
            max_top_k = min(9, n_inputs - 1)
            min_top_k = min(2, max_top_k)  # If max is 1, min becomes 1
            default_top_k = min(5, max_top_k)
            
            top_k = st.slider(
                "Top-K results",
                min_value=min_top_k,
                max_value=max_top_k,
                value=default_top_k
            )
        with col2:
            run_btn = st.button("🚀 Analyze", use_container_width=True)
        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                for i in range(n_inputs):
                    if f"text_{i}" in st.session_state:
                        del st.session_state[f"text_{i}"]
                st.rerun()
        
        # ... rest of the function remains the same
