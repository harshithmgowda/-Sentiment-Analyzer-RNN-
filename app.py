import sys
import os
from pathlib import Path
import streamlit as st

# Setup sys.path to locate src modules
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
for p in [str(BASE_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
    from src.config import (
        DEVICE,
        MODEL_PATH,
        BEST_MODEL_PATH,
        VOCAB_PATH,
        DATASET_PATH,
        OUTPUTS_DIR,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT,
        RNN_TYPE,
        BIDIRECTIONAL,
        MAX_SEQUENCE_LENGTH,
        get_device_info
    )
    from src.predict import predict_sentiment, load_predictor
except ImportError:
    import torch
    from config import (
        DEVICE,
        MODEL_PATH,
        BEST_MODEL_PATH,
        VOCAB_PATH,
        DATASET_PATH,
        OUTPUTS_DIR,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT,
        RNN_TYPE,
        BIDIRECTIONAL,
        MAX_SEQUENCE_LENGTH,
        get_device_info
    )
    from predict import predict_sentiment, load_predictor

# Page configuration
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# SWISS INTERNATIONAL TYPOGRAPHIC STYLE CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Universal Swiss Reset - Strict 0px radius, no shadows, no gradients */
    *, *::before, *::after {
        border-radius: 0px !important;
        box-shadow: none !important;
        text-shadow: none !important;
    }

    :root {
        --swiss-red: #E53935;
        --pure-ink: #0A0A0A;
        --basel-paper: #F7F7F5;
        --grid-white: #FFFFFF;
        --grid-line: #E0E0DC;
        --muted-swiss: #5C5C58;
    }

    /* Core Document Layout */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--basel-paper) !important;
        color: var(--pure-ink) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background-color: var(--basel-paper) !important;
        border-bottom: 1.5px solid var(--pure-ink) !important;
    }

    /* Top Masthead */
    .swiss-header {
        border-bottom: 2px solid var(--pure-ink);
        padding-bottom: 1.2rem;
        margin-bottom: 1.8rem;
    }

    .swiss-category {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--swiss-red);
        margin-bottom: 0.35rem;
    }

    .swiss-title {
        font-family: 'Inter', sans-serif;
        font-size: clamp(2.2rem, 4vw, 3.4rem);
        font-weight: 900;
        line-height: 1.0;
        letter-spacing: -0.04em;
        text-transform: uppercase;
        color: var(--pure-ink);
        margin: 0 0 0.5rem 0;
    }

    .swiss-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        color: var(--muted-swiss);
    }

    /* Section Subheaders */
    .swiss-section-header {
        border-top: 1.5px solid var(--pure-ink);
        border-bottom: 1px solid var(--grid-line);
        padding: 0.5rem 0;
        margin: 0.8rem 0 1rem 0;
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
    }

    .swiss-section-idx {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        color: var(--swiss-red);
        font-size: 0.82rem;
        letter-spacing: 0.08em;
    }

    .swiss-section-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 0.98rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--pure-ink);
    }

    /* Buttons Styling */
    .stButton > button {
        border: 1.5px solid var(--pure-ink) !important;
        background-color: var(--grid-white) !important;
        color: var(--pure-ink) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 800 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        padding: 0.55rem 0.9rem !important;
        transition: all 0.12s ease-in-out !important;
    }

    .stButton > button:hover {
        background-color: var(--pure-ink) !important;
        color: var(--grid-white) !important;
        border-color: var(--pure-ink) !important;
    }

    /* Primary Action Button (Swiss Vermilion) */
    .stButton > button[kind="primary"] {
        background-color: var(--swiss-red) !important;
        color: var(--grid-white) !important;
        border: 1.5px solid var(--pure-ink) !important;
        font-weight: 900 !important;
        letter-spacing: 0.1em !important;
        font-size: 0.92rem !important;
        padding: 0.75rem 1rem !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: var(--pure-ink) !important;
        color: var(--grid-white) !important;
        border-color: var(--pure-ink) !important;
    }

    /* Textarea & Inputs */
    div[data-baseweb="textarea"] {
        border: 1.5px solid var(--pure-ink) !important;
        background-color: var(--grid-white) !important;
    }

    div[data-baseweb="textarea"] textarea {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        color: var(--pure-ink) !important;
        background-color: var(--grid-white) !important;
        line-height: 1.5 !important;
    }

    /* Custom Swiss Result Cards */
    .swiss-card {
        border: 1.5px solid var(--pure-ink);
        background-color: var(--grid-white);
        padding: 1.3rem;
        margin-bottom: 1rem;
    }

    .swiss-card-pos {
        border: 1.5px solid var(--pure-ink);
        background-color: var(--grid-white);
        border-left: 8px solid var(--pure-ink);
        padding: 1.3rem;
        margin-bottom: 1rem;
    }

    .swiss-card-neg {
        border: 1.5px solid var(--pure-ink);
        background-color: var(--grid-white);
        border-left: 8px solid var(--swiss-red);
        padding: 1.3rem;
        margin-bottom: 1rem;
    }

    .swiss-badge-top {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .sentiment-huge-text {
        font-family: 'Inter', sans-serif;
        font-size: 2.3rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        line-height: 1;
        text-transform: uppercase;
        color: var(--pure-ink);
        margin: 0.2rem 0;
    }

    .sentiment-huge-text.neg {
        color: var(--swiss-red);
    }

    .sentiment-desc {
        font-size: 0.88rem;
        font-weight: 500;
        color: var(--muted-swiss);
        margin-top: 0.3rem;
    }

    /* Metric Grid Boxes */
    .swiss-metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.6rem;
        margin-top: 0.8rem;
    }

    .swiss-metric-box {
        border: 1.5px solid var(--pure-ink);
        background-color: var(--grid-white);
        padding: 0.85rem;
        text-align: left;
    }

    .swiss-metric-lbl {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted-swiss);
    }

    .swiss-metric-val {
        font-family: 'Inter', sans-serif;
        font-size: 1.7rem;
        font-weight: 900;
        color: var(--pure-ink);
        letter-spacing: -0.03em;
        margin-top: 0.2rem;
    }

    /* Progress bar */
    div[data-testid="stProgress"] > div {
        background-color: var(--grid-line) !important;
        border: 1px solid var(--pure-ink) !important;
        height: 12px !important;
    }

    div[data-testid="stProgress"] > div > div {
        background-color: var(--swiss-red) !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 800 !important;
        font-size: 0.84rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: var(--muted-swiss) !important;
        border: 1px solid var(--grid-line) !important;
        background-color: var(--basel-paper) !important;
        padding: 0.65rem 1.4rem !important;
        margin-right: 0.4rem !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--pure-ink) !important;
        background-color: var(--grid-white) !important;
        border: 1.5px solid var(--pure-ink) !important;
        border-bottom: 2px solid var(--swiss-red) !important;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        border: 1.5px solid var(--pure-ink) !important;
        background-color: var(--grid-white) !important;
        margin-top: 0.9rem !important;
    }

    /* Figure frame for Tab 2 */
    .swiss-figure-frame {
        border: 1.5px solid var(--pure-ink);
        background-color: var(--grid-white);
        padding: 0.9rem;
        margin-bottom: 1.2rem;
    }

    .swiss-fig-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-bottom: 1.5px solid var(--pure-ink);
        padding-bottom: 0.35rem;
        margin-bottom: 0.75rem;
        display: flex;
        justify-content: space-between;
    }

    /* Code block styling */
    div[data-testid="stCodeBlock"] {
        border: 1.5px solid var(--pure-ink) !important;
    }

    /* Architecture grid in Tab 3 */
    .spec-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }

    .spec-table td {
        padding: 0.5rem 0.6rem;
        border-bottom: 1px solid var(--grid-line);
    }

    .spec-table tr:last-child td {
        border-bottom: none;
    }

    .spec-key {
        font-weight: 700;
        color: var(--muted-swiss);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.76rem;
        width: 40%;
    }

    .spec-val {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: var(--pure-ink);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# MAIN APPLICATION // MASTHEAD
# ==========================================
st.markdown("""
<div class="swiss-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.35rem;">
        <div class="swiss-category">DEEP LEARNING · RECURRENT NEURAL NETWORK</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; font-weight: 800; background-color: var(--pure-ink); color: #FFFFFF; padding: 0.25rem 0.65rem; letter-spacing: 0.08em; text-transform: uppercase;">⚡ TRAINED BY THE RTX 5050</div>
    </div>
    <h1 class="swiss-title">IMDb SENTIMENT ANALYZER</h1>
    <div class="swiss-subtitle">Real-time Movie Review Sentiment Classification Powered by PyTorch Bi-directional LSTM · <strong>Trained by the RTX 5050</strong></div>
</div>
""", unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs([
    "ANALYZE REVIEW",
    "EVALUATION & PLOTS",
    "MODEL SPECS & PIPELINE GUIDE"
])

# ----------------------------------------------------
# TAB 1: INFERENCE ENGINE
# ----------------------------------------------------
with tab1:
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown("""
        <div class="swiss-section-header">
            <span class="swiss-section-idx">01</span>
            <span class="swiss-section-title">INPUT REVIEW</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<div style=\"font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700; color: #5C5C58; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.4rem;\">QUICK PRESETS:</div>",
            unsafe_allow_html=True
        )

        preset_cols = st.columns(3)
        sample_positive = "An absolute cinematic masterpiece! The acting was transcendent, and the directing was visionary."
        sample_negative = "A terrible waste of time. Painfully boring plot, flat dialogue, and awful acting throughout."
        sample_mixed = "The cinematography and special effects were stunning, but the story was weak and predictable."

        if preset_cols[0].button("★ Masterpiece", use_container_width=True):
            st.session_state["review_text"] = sample_positive
        if preset_cols[1].button("✕ Terrible", use_container_width=True):
            st.session_state["review_text"] = sample_negative
        if preset_cols[2].button("≈ Mixed", use_container_width=True):
            st.session_state["review_text"] = sample_mixed

        current_val = st.session_state.get("review_text", "")
        
        st.markdown("<div style='height: 0.3rem;'></div>", unsafe_allow_html=True)
        user_review = st.text_area(
            "Review Input Text",
            value=current_val,
            height=150,
            placeholder="Type or paste a movie review to analyze its sentiment...",
            label_visibility="collapsed"
        )

        st.markdown("<div style='height: 0.3rem;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("ANALYZE SENTIMENT →", type="primary", use_container_width=True)

    with col_right:
        st.markdown("""
        <div class="swiss-section-header">
            <span class="swiss-section-idx">02</span>
            <span class="swiss-section-title">ANALYSIS RESULT</span>
        </div>
        """, unsafe_allow_html=True)

        if analyze_clicked:
            if not user_review.strip():
                st.markdown("""
                <div class="swiss-card" style="border-left: 8px solid #E53935;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: #E53935; text-transform: uppercase;">
                        INPUT REQUIRED
                    </div>
                    <div style="font-size: 0.88rem; margin-top: 0.3rem; color: #0A0A0A;">
                        Please enter a review or click one of the quick presets on the left.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                try:
                    with st.spinner("Analyzing sentiment..."):
                        result = predict_sentiment(user_review)

                    is_pos = result["prediction"] == "Positive"
                    prob = result["probability"]
                    conf = result["confidence"]

                    # Result Card
                    if is_pos:
                        st.markdown(f"""
                        <div class="swiss-card-pos">
                            <div class="swiss-badge-top">
                                <span style="color: #0A0A0A; font-weight: 800;">CLASSIFICATION</span>
                                <span style="color: #5C5C58;">CONFIDENCE: {conf * 100:.1f}%</span>
                            </div>
                            <div class="sentiment-huge-text">POSITIVE</div>
                            <div class="sentiment-desc">
                                The model detected strong positive sentiment in this review.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="swiss-card-neg">
                            <div class="swiss-badge-top">
                                <span style="color: #E53935; font-weight: 800;">CLASSIFICATION</span>
                                <span style="color: #5C5C58;">CONFIDENCE: {conf * 100:.1f}%</span>
                            </div>
                            <div class="sentiment-huge-text neg">NEGATIVE</div>
                            <div class="sentiment-desc">
                                The model detected negative sentiment in this review.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Metric Grid
                    st.markdown(f"""
                    <div class="swiss-metric-grid">
                        <div class="swiss-metric-box">
                            <div class="swiss-metric-lbl">POSITIVE PROBABILITY</div>
                            <div class="swiss-metric-val">{prob * 100:.1f}%</div>
                        </div>
                        <div class="swiss-metric-box">
                            <div class="swiss-metric-lbl">CONFIDENCE SCORE</div>
                            <div class="swiss-metric-val">{conf * 100:.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style=\"font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; color: #5C5C58; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.2rem;\">PROBABILITY SCALE [0.0 → 1.0]:</div>",
                        unsafe_allow_html=True
                    )
                    st.progress(float(prob))

                    with st.expander("VIEW PREPROCESSED TOKENS"):
                        st.markdown(f"""
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; line-height: 1.6;">
                            <div><strong>TOKEN COUNT:</strong> {result['tokens_count']} tokens</div>
                            <div style="margin-top: 0.4rem;"><strong>CLEANED TEXT:</strong></div>
                            <div style="background: #FFFFFF; border: 1px solid #0A0A0A; padding: 0.5rem; margin-top: 0.2rem; word-break: break-word;">
                                {result['clean_text']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                except FileNotFoundError as err:
                    st.markdown(f"""
                    <div class="swiss-card" style="border-left: 8px solid #E53935;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: #E53935;">
                            MODEL CHECKPOINT NOT FOUND
                        </div>
                        <div style="font-size: 0.88rem; margin-top: 0.3rem;">
                            {str(err)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as ex:
                    st.error(f"Error during prediction: {str(ex)}")
        else:
            st.markdown("""
            <div class="swiss-card">
                <div class="swiss-badge-top">
                    <span>STATUS</span>
                    <span style="color: #E53935;">READY</span>
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 1.2rem; font-weight: 900; letter-spacing: -0.02em; text-transform: uppercase; color: #0A0A0A;">
                    READY FOR REVIEW
                </div>
                <div class="sentiment-desc" style="margin-top: 0.4rem;">
                    Enter a movie review on the left and click <strong>Analyze Sentiment</strong> to see the real-time classification and confidence score.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ----------------------------------------------------
# TAB 2: TRAINING & EVALUATION PLOTS
# ----------------------------------------------------
with tab2:
    st.markdown("""
    <div class="swiss-section-header">
        <span class="swiss-section-idx">01</span>
        <span class="swiss-section-title">TRAINING CURVES & EVALUATION METRICS</span>
    </div>
    """, unsafe_allow_html=True)

    loss_img = OUTPUTS_DIR / "training_loss.png"
    acc_img = OUTPUTS_DIR / "accuracy.png"
    cm_img = OUTPUTS_DIR / "confusion_matrix.png"
    eval_txt = OUTPUTS_DIR / "evaluation_results.txt"

    plot_cols = st.columns(2, gap="medium")

    with plot_cols[0]:
        st.markdown("""
        <div class="swiss-figure-frame">
            <div class="swiss-fig-header">
                <span>TRAINING & VALIDATION LOSS</span>
                <span>LOSS CURVE</span>
            </div>
        """, unsafe_allow_html=True)
        if loss_img.exists():
            st.image(str(loss_img), use_container_width=True)
        else:
            st.warning("training_loss.png not found in outputs/.")
        st.markdown("</div>", unsafe_allow_html=True)

    with plot_cols[1]:
        st.markdown("""
        <div class="swiss-figure-frame">
            <div class="swiss-fig-header">
                <span>TRAINING & VALIDATION ACCURACY</span>
                <span>ACCURACY CURVE</span>
            </div>
        """, unsafe_allow_html=True)
        if acc_img.exists():
            st.image(str(acc_img), use_container_width=True)
        else:
            st.warning("accuracy.png not found in outputs/.")
        st.markdown("</div>", unsafe_allow_html=True)

    cm_cols = st.columns([1, 1], gap="medium")

    with cm_cols[0]:
        st.markdown("""
        <div class="swiss-figure-frame">
            <div class="swiss-fig-header">
                <span>TEST CONFUSION MATRIX</span>
                <span>HELD-OUT TEST SET</span>
            </div>
        """, unsafe_allow_html=True)
        if cm_img.exists():
            st.image(str(cm_img), use_container_width=True)
        else:
            st.warning("confusion_matrix.png not found in outputs/.")
        st.markdown("</div>", unsafe_allow_html=True)

    with cm_cols[1]:
        st.markdown("""
        <div class="swiss-figure-frame">
            <div class="swiss-fig-header">
                <span>EVALUATION REPORT</span>
                <span>TEST METRICS</span>
            </div>
        """, unsafe_allow_html=True)
        if eval_txt.exists():
            with open(eval_txt, "r", encoding="utf-8") as f:
                st.code(f.read(), language="text")
        else:
            st.info("Run python src/evaluate.py to generate test metrics report.")
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------
# TAB 3: MODEL SPECS & PIPELINE GUIDE
# ----------------------------------------------------
with tab3:
    st.markdown("""
    <div class="swiss-section-header">
        <span class="swiss-section-idx">01</span>
        <span class="swiss-section-title">MODEL ARCHITECTURE & HYPERPARAMETERS</span>
    </div>
    """, unsafe_allow_html=True)

    col_spec1, col_spec2 = st.columns(2, gap="medium")

    arch_display = f"Bidirectional {RNN_TYPE.upper()}" if BIDIRECTIONAL else RNN_TYPE.upper()
    device_label = "CUDA GPU" if torch.cuda.is_available() else "CPU"

    with col_spec1:
        st.markdown(f"""
        <div class="swiss-card">
            <div class="swiss-badge-top">
                <span>SPECIFICATIONS</span>
                <span style="color: #E53935;">CONFIG</span>
            </div>
            <table class="spec-table">
                <tr><td class="spec-key">Model Type</td><td class="spec-val">{arch_display}</td></tr>
                <tr><td class="spec-key">Embedding Dim</td><td class="spec-val">{EMBEDDING_DIM}</td></tr>
                <tr><td class="spec-key">Hidden Units</td><td class="spec-val">{HIDDEN_DIM}</td></tr>
                <tr><td class="spec-key">Layers</td><td class="spec-val">{NUM_LAYERS}</td></tr>
                <tr><td class="spec-key">Dropout</td><td class="spec-val">{DROPOUT}</td></tr>
                <tr><td class="spec-key">Max Sequence</td><td class="spec-val">{MAX_SEQUENCE_LENGTH} words</td></tr>
                <tr><td class="spec-key">Execution Device</td><td class="spec-val">{device_label}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_spec2:
        dataset_status = "Available" if Path(DATASET_PATH).exists() else "Pre-trained (Cloud)"
        vocab_status = "Available" if Path(VOCAB_PATH).exists() else "Missing"
        model_status = "Trained & Available" if (Path(BEST_MODEL_PATH).exists() or Path(MODEL_PATH).exists()) else "Not Trained"

        st.markdown(f"""
        <div class="swiss-card">
            <div class="swiss-badge-top">
                <span>CHECKPOINT ASSETS</span>
                <span style="color: #E53935;">STORAGE</span>
            </div>
            <table class="spec-table">
                <tr><td class="spec-key">Dataset</td><td class="spec-val">{dataset_status}</td></tr>
                <tr><td class="spec-key">Vocabulary</td><td class="spec-val">{vocab_status}</td></tr>
                <tr><td class="spec-key">Model Weights</td><td class="spec-val">{model_status}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="swiss-section-header" style="margin-top: 1.5rem;">
        <span class="swiss-section-idx">02</span>
        <span class="swiss-section-title">EXECUTION COMMANDS</span>
    </div>
    """, unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2, gap="medium")

    with col_g1:
        st.markdown("**1. Train Model:**")
        st.code("python src/train.py", language="bash")

        st.markdown("**3. Test CLI Prediction:**")
        st.code('python src/predict.py "A breathtaking cinematic masterpiece!"', language="bash")

    with col_g2:
        st.markdown("**2. Evaluate on Test Set:**")
        st.code("python src/evaluate.py", language="bash")

        st.markdown("**4. Launch Web Application:**")
        st.code("streamlit run app.py", language="bash")
