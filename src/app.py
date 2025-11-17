import os
import io
from dataclasses import dataclass
from typing import Tuple, List, Optional

import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image

from utils import load_model_and_meta, predict_pil, topk_labels
from styles import css

SOCIAL_LINKEDIN = "https://www.linkedin.com/in/huakevin/"
SOCIAL_GITHUB = "https://github.com/kevinhuads"
SOCIAL_EMAIL = "mailto:kevin.hua.ds@gmail.com"

ICON_GITHUB = "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg"
ICON_LINKEDIN = "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg"
ICON_EMAIL = "https://cdn-icons-png.flaticon.com/512/561/561127.png"


# Default checkpoint: prefer environment override so CI / users can set their own path
DEFAULT_CKPT = os.environ.get("FOOD101_CKPT", "model.pth")


def _build_confidence_comment(labels: List[str], probs: List[float]) -> str:
    """
    Build a short qualitative comment about the model confidence
    using the top-1 and top-2 probabilities.
    """
    if not labels or not probs:
        return "No predictions available."

    top1 = probs[0]
    label1 = labels[0]

    # If there is no second prediction, fall back to a simple rule.
    if len(probs) < 2 or len(labels) < 2:
        if top1 >= 0.85:
            return "This is a high confidence prediction. The model strongly prefers this class."
        elif top1 >= 0.60:
            return "This is a medium confidence prediction."
        else:
            return (
                "This is a low confidence prediction. The probability mass is spread across several classes."
            )

    top2 = probs[1]
    label2 = labels[1]
    gap = top1 - top2

    # 1. Very strong preference for the first class.
    if top1 >= 0.85 and gap >= 0.20 and top2 < 0.20:
        return (
            f"This is a high confidence prediction. The model strongly prefers {label1}, "
            "and alternative classes are unlikely."
        )

    # 2. Model is genuinely uncertain between the first two classes.
    if top1 >= 0.50 and top2 >= 0.20 and gap <= 0.10:
        return (
            f"The model is uncertain between {label1} and {label2}. "
            f"If the first choice is not correct, there is a high chance it is {label2}."
        )

    # 3. Requested rule: top-2 is also strong (top2 >= 20 percent).
    if top2 >= 0.20:
        return (
            f"The model favours {label1}, but {label2} also has significant probability. "
            f"If {label1} is not correct, there is a high chance that it is {label2}."
        )

    # 4. Medium confidence in top-1 with weaker alternatives.
    if top1 >= 0.60:
        return (
            f"This is a medium confidence prediction. The model clearly prefers {label1}, "
            "and other classes have much lower probability."
        )

    # 5. Low overall confidence.
    return (
        "The confidence is relatively low and the probability is spread across several classes. "
        "Treat this prediction as indicative rather than definitive."
    )


@dataclass
class ModelBundle:
    """Container for all model-related artefacts."""
    model: object
    device: object
    preprocess: object
    classes: List[str]
    ckpt_path: str


@st.cache_resource
def _cached_loader(ckpt: str) -> ModelBundle:
    """
    Cached model loader.

    Separated from Streamlit UI logic so it can be reused in tests or scripts.
    """
    model, device, preprocess, classes = load_model_and_meta(ckpt, map_location="cpu")
    return ModelBundle(
        model=model,
        device=device,
        preprocess=preprocess,
        classes=classes,
        ckpt_path=ckpt,
    )


def _render_left_column(image: Image.Image):

    st.markdown('<div class="section-title">Input image</div>', unsafe_allow_html=True)
    st.image(image, caption=None, width='stretch')


def _render_model_summary(bundle: Optional[ModelBundle]):
    """
    Model information box with three tabs:
    - Model summary
    - Training configuration
    - How to interpret predictions
    """
    if bundle is None:
        st.warning("Model metadata is not available. Check that the model loaded correctly.")
        return

    tab_summary, tab_training, tab_interpret = st.tabs(
        ["Model summary", "Training configuration", "How to interpret predictions"],
    )

    with tab_summary:
        st.markdown(
            f"""
**Task**

- Single-image, single-label food classification  
- **{len(bundle.classes)}** classes from the Food-101 dataset

**Architecture**

- Swin-T (Swin Transformer Tiny) backbone  
- ImageNet-1k pretrained weights

**Runtime**

- Inference device: `{bundle.device}`
            """,
            unsafe_allow_html=False,
        )

    with tab_training:
        st.markdown(
            """
**Core hyperparameters**

- **Pretrained**: yes  
- **Batch size**: 64  
- **Number of epochs**: 25  
- **Initial learning rate**: 3×10⁻⁴  
- **Weight decay**: 0.01  
- **Backbone frozen**: no  

**Optimisation**

- **Optimiser**: AdamW  
- **Learning-rate schedule**: OneCycleLR  

**Loss and precision**

- **Loss function**: cross-entropy with label smoothing (ε = 0.1)  
- **Mixed precision**: gradient scaling with `torch.amp.GradScaler`
            """,
            unsafe_allow_html=False,
        )

    with tab_interpret:
        st.markdown(
            """
**How to read the predictions**

- Works best on clear images of a single dish.  
- The top-1 class is the model’s best guess; lower confidence indicates higher uncertainty.  
- Visually similar dishes (for example different pasta dishes or grilled meats) are more likely to be confused.  
- The model does not detect ingredients or allergens; predictions are indicative only.  
- The top-k list and the confidence chart show alternative plausible classes.
            """,
            unsafe_allow_html=False,
        )




def _render_predictions(labels: List[str], probs: List[float]):
    """
    Right-column content once predictions have been computed.
    """
    if not labels or not probs:
        st.warning("No predictions available.")
        return

    # Top-3 metrics
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.metric("Top-1", labels[0], delta=f"{probs[0] * 100:.1f}%")
    if len(labels) > 1:
        c2.metric("Top-2", labels[1], delta=f"{probs[1] * 100:.1f}%")
    if len(labels) > 2:
        c3.metric("Top-3", labels[2], delta=f"{probs[2] * 100:.1f}%")

    # Short natural-language summary
    st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
    st.write(
        f"The model predicts **{labels[0]}** "
        f"with **{probs[0] * 100:.1f} %** confidence."
    )

    # Qualitative confidence comment using top-1 and top-2
    comment = _build_confidence_comment(labels, probs)
    st.caption(comment)

    tab1, tab2 = st.tabs(["Top-k table", "Confidence chart"])

    with tab1:

        st.markdown('<div class="section-title">Top-k predictions</div>', unsafe_allow_html=True)

        df = pd.DataFrame(
            {
                "class": labels,
                "confidence %": [p * 100 for p in probs],
            }
        )
        st.dataframe(
            df,
            width='stretch',
            hide_index=True,
            column_config={
                "class": st.column_config.TextColumn("class"),
                "confidence %": st.column_config.NumberColumn("confidence %", format="%.2f%%"),
            },
        )

    with tab2:
        st.markdown('<div class="section-title">Confidence</div>', unsafe_allow_html=True)

        chart_df = pd.DataFrame({"class": labels, "confidence": probs})

        # Expand the y-axis slightly above the tallest bar so text labels are not clipped
        if not chart_df.empty:
            max_conf = float(chart_df["confidence"].max())
        else:
            max_conf = 1.0
        y_domain_max = min(1.0, max_conf * 1.2)

        base = (
            alt.Chart(chart_df)
            .mark_bar(
                cornerRadiusTopLeft=6,
                cornerRadiusTopRight=6,
                stroke="white",        # white outline
                strokeWidth=1,
            )
            .encode(
                x=alt.X("class:N", sort="-y", title=""),
                y=alt.Y(
                    "confidence:Q",
                    title="Probability",
                    axis=alt.Axis(format=".0%"),
                    scale=alt.Scale(domain=[0, y_domain_max]),
                ),
                color=alt.Color("class:N", legend=None),  # one color per bar, no legend
                tooltip=[
                    alt.Tooltip("class:N", title="Class"),
                    alt.Tooltip("confidence:Q", title="Confidence", format=".2%"),
                ],
            )
            .properties(height=420)
        )

        text = (
            alt.Chart(chart_df)
            .mark_text(
                dy=-6,
                baseline="bottom",
                color="white",
                tooltip=False,  # disable tooltip on the text layer
            )
            .encode(
                x=alt.X("class:N", sort="-y"),
                y=alt.Y("confidence:Q"),
                text=alt.Text("confidence:Q", format=".2%"),
            )
        )


        st.altair_chart(base + text, use_container_width=True)



def build_ui():
    """
    Compose static parts of the UI (header, layout, style injection). Import-safe.
    """
    st.set_page_config(
        page_title="Food-101 Demo",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="collapsed", 
    )

    # Inject CSS from the module
    st.markdown(f"<style>{css()}</style>", unsafe_allow_html=True)

    
    # Personal header
    st.title("Kevin Hua · Deep Learning · Computer Vision")
    
    st.markdown(
        f"""
        <div class="social-row">
          <a class="social-pill" href="{SOCIAL_GITHUB}" target="_blank" rel="noopener noreferrer" title="GitHub">
            <img src="{ICON_GITHUB}" alt="GitHub" />
            <span>GitHub</span>
          </a>
          <a class="social-pill" href="{SOCIAL_LINKEDIN}" target="_blank" rel="noopener noreferrer" title="LinkedIn">
            <img src="{ICON_LINKEDIN}" alt="LinkedIn" />
            <span>LinkedIn</span>
          </a>
          <a class="social-pill" href="{SOCIAL_EMAIL}" target="_blank" rel="noopener noreferrer" title="Email">
            <img src="{ICON_EMAIL}" alt="Email" />
            <span>Email</span>
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")  # small spacer

    # Header
    st.markdown(
        """
        <div class="app-header">
          <p class="app-title">🍽️ Food-101 Classifier - Demo</p>
          <p class="app-sub">
            Upload a food photo to obtain Top-k predictions with confidence scores.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


VAL_TOP1_APPROX = 0.912
VAL_TOP2_APPROX = 0.959
VAL_TOP3_APPROX = 0.973


def _render_hero(bundle: Optional["ModelBundle"]) -> None:
    """
    Compact overview at the top of the main demo tab.
    """
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    cols = st.columns(4)

    n_classes = (
        len(bundle.classes)
        if bundle is not None and getattr(bundle, "classes", None) is not None
        else "101"
    )

    cols[0].metric("Number of classes", f"{n_classes}")
    cols[1].metric("Val. top-1 accuracy", f"{VAL_TOP1_APPROX * 100:.1f} %")
    cols[2].metric("Val. top-2 accuracy", f"{VAL_TOP2_APPROX * 100:.1f} %")
    cols[3].metric("Val. top-3 accuracy", f"{VAL_TOP3_APPROX * 100:.1f} %")

    st.caption(
        "Validation metrics are approximate and based on a held-out split of the original Food-101 dataset."
    )


def _render_dataset_summary(bundle: Optional["ModelBundle"]) -> None:
    st.markdown('<div class="section-title">Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        """
- Benchmark: [Food-101](http://www.vision.ee.ethz.ch/datasets_extra/food-101/) image dataset  
- **101** food categories  
- **75 750** training images and **25 250** test images  
- Fixed train / test split from the original publication  
- Images collected from real-world food photography
        """
    )


def _render_project_overview() -> None:
    st.markdown('<div class="section-title">Project context</div>', unsafe_allow_html=True)
    st.markdown(
        """
This demo is one surface of a broader end-to-end computer vision project:

- **Problem**: single-image classification on the public [Food-101](http://www.vision.ee.ethz.ch/datasets_extra/food-101/) benchmark  
- **Goal**: compare modern CNN and transformer architectures and deploy a compact but accurate model for interactive use  
- **Scope**: from exploratory analysis of representations through to training, evaluation, testing and containerised deployment  

**Design choices**

- Chose **Swin-T** as the deployed backbone to balance accuracy and latency  
- Fine-tuned the backbone rather than using a frozen encoder after empirical comparison  
- Used configuration-driven training and experiment tracking to keep runs reproducible  

**My responsibilities**

- Defined the experimentation plan and model comparison  
- Implemented the training / evaluation code and tests  
- Designed this Streamlit interface as a lightweight deployment and portfolio artefact  

For further details, see the full project on
[GitHub](https://github.com/kevinhuads/deepvision-workflow).
        """
    )



def _run_inference(
    image: Image.Image,
    bundle: ModelBundle,
    topk: int,
) -> Tuple[List[str], List[float]]:
    """
    Run the end-to-end inference pipeline and return labels and probabilities.

    This keeps the main `run` function short and easier to follow.
    """
    preds = predict_pil(image, bundle.model, bundle.device, bundle.preprocess, topk=topk)
    labeled = topk_labels(preds, bundle.classes)
    labels = [label for label, _ in labeled]
    probs = [float(p) for _, p in labeled]
    return labels, probs

def build_footer():
    st.markdown(
        """
        <hr style="margin-top: 3rem;">
        <div style="text-align: center; font-size: 0.85rem; opacity: 0.7; padding-bottom: 0.5rem;">
            © 2025 Kevin Hua. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True,
    )


def run(ckpt: Optional[str] = None):
    """
    Entry point to run the Streamlit app.

    Parameters
    ----------
    ckpt:
        Optional override for checkpoint path; if None, uses the default from `FOOD101_CKPT`.
    """
    build_ui()

    ckpt_to_use = ckpt or DEFAULT_CKPT

    # Lazy load the model (cached)
    try:
        bundle = _cached_loader(ckpt_to_use)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load model from checkpoint `{ckpt_to_use}`:\n\n{exc}")
        bundle = None

    tab_demo, tab_model, tab_project = st.tabs(
        ["Interactive demo", "Model & data", "Project details"]
    )

    with tab_demo:
        _render_hero(bundle)

        left_col, right_col = st.columns([5, 7], gap="large")

        with left_col:
            st.markdown('<div class="section-title">Upload an image</div>', unsafe_allow_html=True)
            st.caption("Close-up images of a single dish lead to more reliable predictions.")

            uploaded = st.file_uploader(
                "Upload a JPG or PNG",
                type=["jpg", "jpeg", "png"],
                help="Large, in-focus photos of a single dish tend to work best.",
            )

            with st.expander("Prediction settings", expanded=False):
                topk = st.slider(
                    "Number of classes to display (k)",
                    min_value=1,
                    max_value=10,
                    value=5,
                )
        # fall back value in case the code below is reached without the expander being rendered
        topk = locals().get("topk", 5)

            
        if uploaded is not None:
            try:
                image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
            except Exception:
                st.error("The uploaded file could not be interpreted as an image.")
                image = None

            with right_col:
                st.markdown('<div class="section-title">Input image</div>', unsafe_allow_html=True)
                if image is not None:
                    st.image(image, width='stretch')
                else:
                    st.info("The image preview will appear here once a valid file is uploaded.")

            with left_col:
                st.markdown('<div class="section-title">Predictions</div>', unsafe_allow_html=True)
                if image is not None:
                    if bundle is None:
                        st.error("Model not loaded. Check the checkpoint path in the configuration.")
                    else:
                        labels, probs = _run_inference(image, bundle, topk=topk)
                        _render_predictions(labels, probs)
                else:
                    st.info("Predictions will appear here once the image has been read successfully.")
        else:
            with right_col:
                st.markdown('<div class="section-title">Input image</div>', unsafe_allow_html=True)
                st.info("Upload an image on the left to see the preview.")
            with left_col:
                st.markdown('<div class="section-title">Predictions</div>', unsafe_allow_html=True)
                st.info(
                    "Top-k predictions and the confidence chart will appear once an image is uploaded."
                )


    with tab_model:
        _render_model_summary(bundle)
        st.write("")
        _render_dataset_summary(bundle)

    with tab_project:
        _render_project_overview()

    build_footer()


if __name__ == "__main__":
    run()
