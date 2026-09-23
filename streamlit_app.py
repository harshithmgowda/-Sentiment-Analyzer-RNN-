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
    page_title="IMDb Sentiment Analyzer | PyTorch Deep RNN",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card-pos {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border-left: 6px solid #10B981;
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .card-neg {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border-left: 6px solid #EF4444;
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .sentiment-badge {
        font-size: 1.6rem;
        font-weight: 700;
    }
    .metric-box {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=64)
    st.title("System & Model")

    # Hardware Info
    st.markdown("### 🖥️ Hardware Device")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"**CUDA GPU Online**\n\n`{gpu_name}`")
    else:
        st.info("**CPU Mode Active**\n\n(No CUDA GPU detected)")

    st.markdown("---")

    # Architecture Overview
    st.markdown("### 🧠 Model Specs")
    arch_display = f"Bidirectional {RNN_TYPE.upper()}" if BIDIRECTIONAL else RNN_TYPE.upper()
    st.write(f"- **Architecture**: `{arch_display}`")
    st.write(f"- **Embedding Dim**: `{EMBEDDING_DIM}`")
    st.write(f"- **Hidden Dim**: `{HIDDEN_DIM}`")
    st.write(f"- **Layers**: `{NUM_LAYERS}`")
    st.write(f"- **Dropout**: `{DROPOUT}`")
    st.write(f"- **Max Seq Length**: `{MAX_SEQUENCE_LENGTH}` words")

    st.markdown("---")

    # Checkpoint File Status
    st.markdown("### 📁 Project Status")
    dataset_status = "✅ Local CSV" if Path(DATASET_PATH).exists() else "☁️ Pre-trained (Cloud)"
    vocab_status = "✅ Found" if Path(VOCAB_PATH).exists() else "❌ Missing"
    model_status = "✅ Found" if (Path(BEST_MODEL_PATH).exists() or Path(MODEL_PATH).exists()) else "❌ Not Trained"

    st.write(f"- **Dataset**: {dataset_status}")
    st.write(f"- **Vocab**: {vocab_status}")
    st.write(f"- **Model**: {model_status}")


# ==========================================
# MAIN APPLICATION
# ==========================================
st.markdown('<div class="main-title">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Natural Language Processing with Deep Recurrent Neural Networks (PyTorch)</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["🔍 Analyze Review", "📊 Training & Evaluation Plots", "📖 Run Pipeline Guide"])

# ----------------------------------------------------
# TAB 1: PREDICTION PLAYGROUND
# ----------------------------------------------------
with tab1:
    col_left, col_right = st.columns([1.2, 0.8], gap="large")

    with col_left:
        st.markdown("#### Enter Movie Review")
        
        # Sample Buttons
        st.markdown("**Quick Preset Examples:**")
        preset_cols = st.columns(3)
        sample_positive = "An absolute cinematic masterpiece! The acting was transcendent, and the directing was visionary."
        sample_negative = "A terrible waste of time. Painfully boring plot, flat dialogue, and awful acting throughout."
        sample_mixed = "The cinematography and special effects were stunning, but the story was weak and predictable."

        review_input = ""
        if preset_cols[0].button("🌟 Masterpiece (Positive)"):
            st.session_state["review_text"] = sample_positive
        if preset_cols[1].button("🗑️ Terrible (Negative)"):
            st.session_state["review_text"] = sample_negative
        if preset_cols[2].button("⚖️ Mixed / Average"):
            st.session_state["review_text"] = sample_mixed

        current_val = st.session_state.get("review_text", "")
        user_review = st.text_area(
            "Type or paste your review here:",
            value=current_val,
            height=140,
            placeholder="e.g. This movie was brilliantly executed with unforgettable performances..."
        )

        analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

    with col_right:
        st.markdown("#### Prediction Output")

        if analyze_clicked:
            if not user_review.strip():
                st.warning("Please type a review or click one of the quick preset examples first.")
            else:
                try:
                    with st.spinner("Analyzing sentiment with model..."):
                        result = predict_sentiment(user_review)

                    is_pos = result["prediction"] == "Positive"
                    prob = result["probability"]
                    conf = result["confidence"]

                    # Display card
                    if is_pos:
                        st.markdown(f"""
                        <div class="card-pos">
                            <span class="sentiment-badge">🟢 POSITIVE</span>
                            <p style="margin-top: 0.4rem; color: #065F46;">
                                The model detected strong positive sentiment in this review.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card-neg">
                            <span class="sentiment-badge">🔴 NEGATIVE</span>
                            <p style="margin-top: 0.4rem; color: #991B1B;">
                                The model detected negative sentiment in this review.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("##### Sentiment Score & Confidence")
                    st.progress(float(prob))
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("Positive Probability", f"{prob * 100:.1f}%")
                    col_m2.metric("Confidence", f"{conf * 100:.1f}%")

                    with st.expander("🛠️ Preprocessed Tokens"):
                        st.write(f"**Cleaned Text:** `{result['clean_text']}`")
                        st.write(f"**Token Count:** `{result['tokens_count']}`")

                except FileNotFoundError as err:
                    st.error(f"⚠️ {str(err)}")
                    st.info("💡 To train the model, run `python src/train.py` in PyCharm.")
                except Exception as ex:
                    st.error(f"An error occurred: {str(ex)}")
        else:
            st.info("👉 Enter a review on the left and click **'Analyze Sentiment'** to see real-time classification.")


# ----------------------------------------------------
# TAB 2: TRAINING & EVALUATION PLOTS
# ----------------------------------------------------
with tab2:
    st.markdown("### 📈 Training Curves & Confusion Matrix")

    loss_img = OUTPUTS_DIR / "training_loss.png"
    acc_img = OUTPUTS_DIR / "accuracy.png"
    cm_img = OUTPUTS_DIR / "confusion_matrix.png"
    eval_txt = OUTPUTS_DIR / "evaluation_results.txt"

    plot_cols = st.columns(2)

    with plot_cols[0]:
        st.markdown("#### Loss Curve")
        if loss_img.exists():
            st.image(str(loss_img), caption="Training & Validation Loss over Epochs", use_container_width=True)
        else:
            st.warning("`training_loss.png` not found in `outputs/`. Run `python src/train.py` to generate it.")

    with plot_cols[1]:
        st.markdown("#### Accuracy Curve")
        if acc_img.exists():
            st.image(str(acc_img), caption="Training & Validation Accuracy over Epochs", use_container_width=True)
        else:
            st.warning("`accuracy.png` not found in `outputs/`. Run `python src/train.py` to generate it.")

    st.markdown("---")

    cm_cols = st.columns([1, 1])
    with cm_cols[0]:
        st.markdown("#### Test Confusion Matrix")
        if cm_img.exists():
            st.image(str(cm_img), caption="Confusion Matrix on Unseen Test Set", use_container_width=True)
        else:
            st.warning("`confusion_matrix.png` not found in `outputs/`. Run `python src/evaluate.py` to generate it.")

    with cm_cols[1]:
        st.markdown("#### Evaluation Summary Report")
        if eval_txt.exists():
            with open(eval_txt, "r", encoding="utf-8") as f:
                st.code(f.read(), language="text")
        else:
            st.info("Run `python src/evaluate.py` to generate the test metrics report.")


# ----------------------------------------------------
# TAB 3: PIPELINE GUIDE
# ----------------------------------------------------
with tab3:
    st.markdown("### 🚀 Step-by-Step PyCharm Execution Guide")
    st.markdown("""
    #### Step 1: Re-train the High-Accuracy Model on CUDA
    ```bash
    python src/train.py
    ```

    #### Step 2: Evaluate on Unseen Test Set
    ```bash
    python src/evaluate.py
    ```

    #### Step 3: Test Real-time Inferences
    ```bash
    python src/predict.py "A breathtaking cinematic masterpiece with brilliant performances!"
    ```

    #### Step 4: Launch Web App
    ```bash
    streamlit run app.py
    ```
    """)
