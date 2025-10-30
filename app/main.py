import streamlit as st
from pathlib import Path
import sys

# --- Path and Import Handling ---
SRC_PATH = str(Path(__file__).resolve().parents[1] / 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

try:
    from utils.config_loader import load_config
    from core_processing import CoreProcessor
    from utils.exceptions import CDSSException
except ImportError as err:
    st.error(f"Failed to import CDSS modules: {err}")
    st.stop()

# --- Page Config ---
st.set_page_config(
    page_title="Gene-Drug CDSS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar with health check and diagnostics
st.sidebar.title("🔧 CDSS Diagnostics")
if "core_processor" in st.session_state:
    st.sidebar.success("Core Processor Loaded ✅")
else:
    st.sidebar.warning("Processor not loaded ❌")

st.sidebar.markdown("---")
config_path = st.sidebar.text_input("Config Path", value="config/config.yaml")
image_path = st.sidebar.text_input("Architecture Image Path", value="app/assets/architecture.png")

# --- Caching Function ---
@st.cache_resource
def load_core_processor(config_path=config_path):
    """
    Loads the config and heavy CoreProcessor object. Cached for the session.
    """
    try:
        st.info("Loading configuration...", icon="🔔")
        config = load_config(config_path)
        st.info("Initializing Core Processor (this may take a moment)...", icon="📊")
        processor = CoreProcessor(config)
        st.success("Processor loaded.", icon="✅")
        return processor
    except CDSSException as e:
        st.error(f"Initialization Failed: {e.message}\nDetails: {e.details}")
        return None
    except FileNotFoundError:
        st.error(f"Config file not found at `{config_path}`. Please check the path in the sidebar.")
        return None
    except Exception as e:
        st.exception(e)
        st.error(f"A critical error occurred during startup: {e}")
        return None

# --- Main App Logic ---
st.title("🧬 Gene-Drug Clinical Decision Support System (CDSS)")

# Load processor (cached)
processor = load_core_processor(config_path)

if processor:
    st.session_state["core_processor"] = processor
    st.session_state["config"] = processor.config

    st.markdown("""
    Welcome to the Gene-Drug CDSS. This tool uses a **state-of-the-art E(n)-Equivariant Graph Neural Network** and **Protein Language Model** to predict gene-drug interactions.

    **Select a tool from the sidebar to begin.**
    """)

    # Architecture diagram (with robust handling)
    img_path = Path(image_path)
    if img_path.is_file():
        st.image(str(img_path), caption="Model Architecture: EGNN and ESM-2 with Cross-Attention")
    else:
        st.warning(f"Architecture diagram not found at `{image_path}`. Update the path in the sidebar to display.")

    st.info(
        "If you encounter any errors or unexpected behavior, please check the sidebar for diagnostics and confirm your config/image file paths.",
        icon="⚠️"
    )
else:
    st.error("Application failed to load. Please check error messages and logs above or in the sidebar.")
    st.stop()
