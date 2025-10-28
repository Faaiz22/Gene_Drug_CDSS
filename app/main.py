import streamlit as st
from pathlib import Path
import sys

# Add src to path
SRC_PATH = str(Path(__file__).resolve().parents[1] / 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from utils.config_loader import load_config
from core_processing import CoreProcessor
from utils.exceptions import CDSSException

st.set_page_config(
    page_title="Gene-Drug CDSS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Caching Function ---
@st.cache_resource
def load_core_processor(config_path="config/config.yaml"):
    """
    Loads the config and heavy CoreProcessor object.
    This is cached for the entire session.
    """
    try:
        st.write("Loading configuration...")
        config = load_config(config_path)
        st.write("Initializing Core Processor (this may take a moment)...")
        processor = CoreProcessor(config)
        st.write("Processor loaded.")
        return processor
    except CDSSException as e:
        st.error(f"Initialization Failed: {e.message}\nDetails: {e.details}")
        return None
    except Exception as e:
        st.error(f"A critical error occurred on startup: {e}")
        return None

# --- Main App Logic ---
st.title("🧬 Gene-Drug Clinical Decision Support System (CDSS)")

# Load the processor (will be cached after first run)
processor = load_core_processor()

if processor:
    # Save the processor to session state so all pages can access it
    st.session_state["core_processor"] = processor
    st.session_state["config"] = processor.config
    
    st.markdown("""
    Welcome to the Gene-Drug CDSS. This tool uses a state-of-the-art
    E(n)-Equivariant Graph Neural Network and Protein Language Model
    to predict interactions.
    
    **Please select a tool from the sidebar to begin.**
    
    - **Single Prediction:** Analyze one drug-gene pair.
    - **Batch Analysis:** Upload a file for bulk predictions.
    - **3D Visualization:** Explore 3D molecular structures.
    - **Model Explanation:** Understand *why* the model made a prediction.
    - **Literature Explorer:** Chat with an AI agent about PubMed literature.
    """)
else:
    st.error("Application failed to load. Please check logs.")
    st.stop()
