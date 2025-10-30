import streamlit as st
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# ----- Path Setup -----
SRC_PATH = str(Path(__file__).resolve().parents[2] / 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# ----- Import Explainer and CoreProcessor -----
try:
    from src.models.explainer import ModelExplainer
    from src.core_processing import CoreProcessor
except ImportError as e:
    st.error(f"Failed to import CDSS modules: {e}")
    st.stop()

# ----- Streamlit Page Config -----
st.set_page_config(page_title="Model Explanation", page_icon="💡")
st.title("💡 Model Explanation")
st.markdown("""
This page offers insights into why the model made a prediction for any gene-drug pair.
Advanced algorithms (e.g., Integrated Gradients, attention, feature attributions) highlight key atom/residue contributions.
""")

# ----- Load Processor -----
if "core_processor" not in st.session_state:
    st.error("🧬 Core processor not initialized. Please return to the main page before using this tool.")
    st.stop()
core_processor: CoreProcessor = st.session_state["core_processor"]

# ----- Explainer Loader with Caching -----
@st.cache_resource
def load_explainer() -> ModelExplainer:
    """
    Loads ModelExplainer using the processed model from CoreProcessor.
    Raises error and prevents use if unavailable.
    """
    model = getattr(core_processor, "model", None)
    if model is None:
        raise ValueError("No model found in CoreProcessor. Return to main page for full initialization.")
    return ModelExplainer(model)

# ----- Input Data Preprocessing and Explanation -----
@st.cache_data(show_spinner=False)
def get_explanation(gene_id: str, chem_id: str):
    """
    Generates and caches the explanation figure for a gene-drug pair.
    """
    explainer = load_explainer()
    # Data pre-processing (with user-friendly error reporting)
    try:
        data_input = core_processor.get_preprocessed_data_for_pair(gene_id, chem_id)
        if data_input is None:
            raise RuntimeError("Preprocessing failed: invalid gene or chemical ID.")
    except Exception as e:
        raise RuntimeError(f"Preprocessing error: {e}")
    # Generate explanation
    try:
        fig = explainer.explain(data_input)
        if fig is None:
            raise ValueError("Explainer did not return a figure.")
        return fig
    except Exception as e:
        raise RuntimeError(f"Explanation error: {e}")

# ----- Main Functionality UI -----
col1, col2 = st.columns(2)
with col1:
    gene_id = st.text_input("Gene Identifier", placeholder="e.g., CYP2C9", key="exp_gene_id")
with col2:
    chem_id = st.text_input("Chemical Identifier", placeholder="e.g., celecoxib", key="exp_chem_id")

if st.button("Explain Prediction", type="primary", key="exp_button"):
    if not gene_id.strip() or not chem_id.strip():
        st.warning("Please enter both identifiers to generate an explanation.")
    else:
        with st.spinner("⚙️ Generating explanation... (This may take up to a minute)"):
            try:
                fig = get_explanation(gene_id, chem_id)
                st.success("Explanation generated!")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Failed to generate explanation: {e}")
                st.exception(e)

st.markdown("---")
st.info("Note: Explanation quality depends on the underlying model and your selected gene-drug pair.")


