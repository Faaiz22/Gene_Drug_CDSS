import streamlit as st
from pathlib import Path
import sys
import os # Import os to set environment variables

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

# --- Sidebar ---
st.sidebar.title("🔧 CDSS Diagnostics & Config")

# --- Credential Input ---
st.sidebar.header("Credentials (Required)")
st.sidebar.info("These are required to load the processor and are not stored.")

# Use st.session_state to hold the values
if "pubmed_email" not in st.session_state:
    st.session_state.pubmed_email = ""
if "pubmed_api_key" not in st.session_state: # <-- NEW
    st.session_state.pubmed_api_key = ""
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""

st.session_state.pubmed_email = st.sidebar.text_input(
    "PubMed Email (Required by NCBI)", 
    value=st.session_state.pubmed_email,
    help="NCBI requires an email for API access.",
    type="password"
)

# --- NEW FIELD ---
st.session_state.pubmed_api_key = st.sidebar.text_input(
    "PubMed API Key (Optional, for higher rate)", 
    value=st.session_state.pubmed_api_key,
    help="NCBI API Key for higher request volumes.",
    type="password"
)
# --- END NEW FIELD ---

st.session_state.google_api_key = st.sidebar.text_input(
    "Google API Key (Required for Agent)", 
    value=st.session_state.google_api_key,
    help="Google Generative AI key for the agent.",
    type="password"
)

st.sidebar.markdown("---")
st.sidebar.header("Status")

# --- Caching Function ---
# (This remains in session_state, not cache, which is correct)
def load_core_processor_once(config_path):
    """
    Loads the config and heavy CoreProcessor object.
    This function is called by the button callback.
    """
    try:
        st.info("Loading configuration...", icon="🔔")
        # Config loading will now succeed because the env var is set
        config = load_config(config_path)
        st.info("Initializing Core Processor (this may take a moment)...", icon="📊")
        processor = CoreProcessor(config)
        st.success("Processor loaded.", icon="✅")
        # Store the loaded processor and config in session_state
        st.session_state["core_processor"] = processor
        st.session_state["config"] = processor.config
    except CDSSException as e:
        st.error(f"Initialization Failed: {e.message}\nDetails: {e.details}")
        st.session_state.pop("core_processor", None)
    except FileNotFoundError:
        st.error(f"Config file not found at `{config_path}`. Please check the path.")
        st.session_state.pop("core_processor", None)
    except Exception as e:
        st.exception(e)
        st.error(f"A critical error occurred during startup: {e}")
        st.session_state.pop("core_processor", None)

# --- Callback for the button ---
def initialize_processor():
    """
    Sets environment variables from session_state and loads the processor.
    """
    # Required fields
    if not st.session_state.pubmed_email:
        st.sidebar.error("PubMed Email is required.")
        return
    if not st.session_state.google_api_key:
        st.sidebar.error("Google API Key is required.")
        return
        
    # --- THIS IS THE CRITICAL STEP ---
    # Set environment variables for this session *before* loading config
    os.environ["PUBMED_EMAIL"] = st.session_state.pubmed_email
    os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
    
    # --- NEWLY ADDED ---
    # Set the PubMed API Key if provided.
    # This allows the config_loader to pick it up.
    if st.session_state.pubmed_api_key:
        os.environ["PUBMED_API_KEY"] = st.session_state.pubmed_api_key
    # --- END NEWLY ADDED ---
    
    # Get config path from sidebar
    config_path = st.session_state.get("config_path_input", "config/config.yaml")
    
    # Load the processor
    load_core_processor_once(config_path)


# --- Sidebar UI (continued) ---
if "core_processor" in st.session_state:
    st.sidebar.success("Core Processor Loaded ✅")
else:
    st.sidebar.warning("Processor not loaded ❌")

st.sidebar.button("Load / Reload Processor", on_click=initialize_processor, type="primary")

st.sidebar.markdown("---")
st.sidebar.header("File Paths (Optional)")
st.text_input("Config Path", value="config/config.yaml", key="config_path_input")
image_path = st.sidebar.text_input("Architecture Image Path", value="app/assets/architecture.png")


# --- Main App Logic ---
st.title("🧬 Gene-Drug Clinical Decision Support System (CDSS)")

# The app is now gated by the processor in session_state
if "core_processor" in st.session_state:
    st.markdown("""
    Welcome to the Gene-Drug CDSS. This tool uses a **state-of-the-art E(n)-Equivariant Graph Neural Network** and **Protein Language Model** to predict gene-drug interactions.

    **Select a tool from the sidebar to begin.**
    """)

    # Architecture diagram
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
    st.error("Application failed to load.")
    st.warning("Please enter your credentials in the sidebar and click 'Load Processor'.")
    st.stop()

