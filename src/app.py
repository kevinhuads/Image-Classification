import os
import io
from typing import Tuple, List

import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image

from utils import load_model_and_meta, predict_pil, topk_labels
from styles import css

# Default checkpoint: prefer environment override so CI / users can set their own path
DEFAULT_CKPT = os.environ.get(
    "FOOD101_CKPT",
    os.path.join("artifacts", "checkpoints", "headonly_food101.pth")
)


@st.cache_resource
def _cached_loader(ckpt: str):
    """Cached model loader — decorated but not called at import time."""
    return load_model_and_meta(ckpt)


def _build_sidebar() -> Tuple[object, str]:
    """Build sidebar controls and return (uploaded_file, ckpt_path)."""
    st.sidebar.markdown("#### Controls")
    uploaded = st.sidebar.file_uploader("Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])
    st.sidebar.markdown("---")
    st.sidebar.caption("Model")
    st.sidebar.code("Food-101 head-only classifier", language="text")

    # Allow overriding checkpoint via sidebar text input for convenience during dev
    ckpt_input = st.sidebar.text_input("Checkpoint path", value=DEFAULT_CKPT)
    return uploaded, ckpt_input


def _render_left_column(image: Image.Image):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Input image</div>', unsafe_allow_html=True)
    st.image(image, caption=None, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_predictions(labels: List[str], probs: List[float]):
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.metric("Top-1 class", labels[0], delta=f"{probs[0]*100:.1f}%")
    c2.metric("Top-2", labels[1], delta=f"{probs[1]*100:.1f}%")
    c3.metric("Top-3", labels[2], delta=f"{probs[2]*100:.1f}%")

    tab1, tab2 = st.tabs(["Top-5 table", "Confidence chart"])
    with tab1:
        st.markdown('<div class="card soft">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top-5 predictions</div>', unsafe_allow_html=True)
        df = pd.DataFrame({"class": labels, "confidence": probs, "confidence %": [p * 100 for p in probs]})
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "class": st.column_config.TextColumn("class"),
                "confidence": st.column_config.NumberColumn("confidence", format="%.4f"),
                "confidence %": st.column_config.NumberColumn("confidence %", format="%.1f%%"),
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card soft">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Confidence</div>', unsafe_allow_html=True)
        chart_df = pd.DataFrame({"class": labels, "confidence": probs})
        base = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("class:N", sort="-y", title=""),
            y=alt.Y("confidence:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip("class:N"), alt.Tooltip("confidence:Q", format=".2%")],
        ).properties(height=260)
        text = alt.Chart(chart_df).mark_text(dy=-6).encode(
            x=alt.X("class:N", sort="-y"),
            y=alt.Y("confidence:Q", stack=None),
            text=alt.Text("confidence:Q", format=".1%"),
        )
        st.altair_chart(base + text, use_container_width=True)


def build_ui():
    """Compose static parts of the UI (header, layout, style injection). Import-safe."""
    st.set_page_config(
        page_title="Food-101 Demo",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject CSS from the module
    st.markdown(f"<style>{css()}</style>", unsafe_allow_html=True)

    # Header
    st.markdown(
        """
        <div class="app-header">
          <p class="app-title">🍽️ Food-101 Classifier — Demo</p>
          <p class="app-sub">Upload a food photo to obtain Top-5 predictions with confidence scores.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


def run(ckpt: str | None = None):
    """
    Entry point to actually run the Streamlit app.
    - `ckpt` optional override for checkpoint path; if None use DEFAULT_CKPT or env var.
    """
    build_ui()
    uploaded, ckpt_input = _build_sidebar()
    ckpt_to_use = ckpt or ckpt_input or DEFAULT_CKPT

    # Lazy load the model (cached)
    try:
        model, device, preprocess, classes = _cached_loader(ckpt_to_use)
    except Exception as exc:
        st.error(f"Failed to load model from checkpoint {ckpt_to_use}: {exc}")
        # Do not raise here — allow the app to remain visible and show the error.
        model = device = preprocess = classes = None

    left_col, right_col = st.columns([5, 7], gap="large")

    if uploaded is not None:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

        with left_col:
            _render_left_column(image)

        with right_col:
            if model is None:
                st.error("Model not loaded. Check the checkpoint path in the sidebar.")
            else:
                preds = predict_pil(image, model, device, preprocess, topk=5)
                labeled = topk_labels(preds, classes)
                labels = [l for l, _ in labeled]
                probs = [float(p) for _, p in labeled]
                _render_predictions(labels, probs)

    else:
        with left_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Input image</div>', unsafe_allow_html=True)
            st.info("Upload an image in the sidebar to preview it here.")
            st.markdown("</div>", unsafe_allow_html=True)

        with right_col:
            st.markdown('<div class="card soft">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Predictions</div>', unsafe_allow_html=True)
            st.info("Top-5 predictions and confidence chart will appear after an image is uploaded.")
            st.markdown("</div>", unsafe_allow_html=True)
