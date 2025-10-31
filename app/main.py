"""
Main Streamlit Application Entry Point for Gene-Drug CDSS
"""

import streamlit as st
import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# CRITICAL: Set up Python path FIRST
# ==========================================
# Get the project root (parent of 'app' directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / 'src'

# Add to Python path if not already there
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
    logger.info(f"Added {SRC_PATH} to Python path")

# ==========================================
# Import CDSS modules AFTER path setup
# ==========================================
try:
    from utils.config_loader import load_config
    from core_processing import CoreProcessor
    from utils.exceptions import CDSSException
    logger.info("Successfully imported CDSS modules")
except ImportError as err:
    st.error(f"⚠️ **CRITICAL ERROR**: Failed to import CDSS modules")
    st.error(f"**Error Details**: {err}")
    st.error(
        f"**Troubleshooting**:\n"
        f"1. Ensure all dependencies are installed: `pip install -r requirements.txt`\n"
        f"2. Check that the `src/` directory exists at: {SRC_PATH}\n"
        f"3. Verify Python path: {sys.path}"
    )
    st.stop()

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Gene-Drug CDSS",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# Sidebar: Credentials & Configuration
# ==========================================
st.sidebar.title("🔧 CDSS Configuration")

# Initialize session state for credentials
if "pubmed_email" not in st.session_state:
    st.session_state.pubmed_email = ""
if "pubmed_api_key" not in st.session_state:
    st.session_state.pubmed_api_key = ""
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""

# Credential inputs
st.sidebar.header("🔑 API Credentials")
st.sidebar.info(
    "**Required**: These credentials are needed to:\n"
    "- Fetch molecular data from PubChem/UniProt\n"
    "- Search literature via PubMed\n"
    "- Power the AI agent (Google Gemini)\n\n"
    "**Privacy**: Credentials are NOT stored permanently."
)

st.session_state.pubmed_email = st.sidebar.text_input(
    "PubMed Email (Required by NCBI)",
    value=st.session_state.pubmed_email,
    help="NCBI requires an email for API access (rate limiting purposes)",
    type="password"
)

st.session_state.pubmed_api_key = st.sidebar.text_input(
    "PubMed API Key (Optional)",
    value=st.session_state.pubmed_api_key,
    help="Optional NCBI API key for higher rate limits (get one at: https://www.ncbi.nlm.nih.gov/account/)",
    type="password"
)

st.session_state.google_api_key = st.sidebar.text_input(
    "Google API Key (Required for AI Agent)",
    value=st.session_state.google_api_key,
    help="Required for the agentic analysis features (get one at: https://makersuite.google.com/app/apikey)",
    type="password"
)

st.sidebar.markdown("---")

# ==========================================
# Processor Initialization
# ==========================================
st.sidebar.header("📊 System Status")

def initialize_processor():
    """
    Initialize the CoreProcessor with credentials from session state.
    Sets environment variables and loads the processor.
    """
    # Validation
    if not st.session_state.pubmed_email:
        st.sidebar.error("❌ PubMed Email is required")
        return
    if not st.session_state.google_api_key:
        st.sidebar.error("❌ Google API Key is required")
        return
    
    # Set environment variables (these are read by config_loader)
    os.environ["PUBMED_EMAIL"] = st.session_state.pubmed_email
    os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
    
    if st.session_state.pubmed_api_key:
        os.environ["PUBMED_API_KEY"] = st.session_state.pubmed_api_key
    
    # Load configuration
    config_path = st.session_state.get("config_path_input", "config/config.yaml")
    
    with st.spinner("🔄 Initializing CDSS Processor..."):
        try:
            # Load config (with environment variable substitution)
            config = load_config(config_path)
            st.success("✅ Configuration loaded", icon="📋")
            
            # Initialize CoreProcessor
            st.info("⚙️ Loading machine learning models... (this may take 30-60 seconds)", icon="🤖")
            processor = CoreProcessor(config)
            
            # Store in session state
            st.session_state["core_processor"] = processor
            st.session_state["config"] = config
            
            st.sidebar.success("✅ Processor Initialized Successfully!", icon="🎉")
            logger.info("CoreProcessor initialized successfully")
            
        except FileNotFoundError as e:
            st.sidebar.error(f"❌ Configuration file not found: {e}")
            logger.error(f"Config file not found: {e}")
        except CDSSException as e:
            st.sidebar.error(f"❌ Initialization Failed: {e.message}")
            if e.details:
                st.sidebar.error(f"**Details**: {e.details}")
            logger.error(f"CDSS Exception: {e}")
        except Exception as e:
            st.sidebar.error(f"❌ Unexpected Error: {str(e)}")
            st.sidebar.exception(e)
            logger.exception("Unexpected error during initialization")

# Status indicator
if "core_processor" in st.session_state:
    st.sidebar.success("✅ System Ready", icon="🟢")
else:
    st.sidebar.warning("⚠️ System Not Initialized", icon="🔴")

# Initialize button
st.sidebar.button(
    "🚀 Initialize / Reload System",
    on_click=initialize_processor,
    type="primary",
    use_container_width=True,
    help="Click to load the CDSS processor with your credentials"
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Advanced Settings")
st.text_input(
    "Config File Path",
    value="config/config.yaml",
    key="config_path_input",
    help="Path to the YAML configuration file"
)

# ==========================================
# Main Content Area
# ==========================================
st.title("🧬 Gene-Drug Clinical Decision Support System")

if "core_processor" not in st.session_state:
    # Show welcome message if not initialized
    st.warning(
        "⚠️ **System Not Initialized**\n\n"
        "Please enter your API credentials in the sidebar and click "
        "'Initialize System' to begin using the CDSS.",
        icon="🔐"
    )
    
    st.markdown("---")
    st.markdown("## 🎯 What This System Does")
    st.markdown(
        """
        This Clinical Decision Support System uses **state-of-the-art AI** to predict 
        drug-gene interactions:
        
        - **🔬 Molecular Analysis**: E(n)-Equivariant Graph Neural Networks (EGNN) process 3D drug structures
        - **🧬 Protein Modeling**: ESM-2 Protein Language Model encodes biological sequences
        - **🤖 Agentic AI**: Autonomous reasoning agent orchestrates multi-step analysis
        - **📚 Literature Mining**: Real-time PubMed searches find supporting evidence
        - **💡 Explainability**: Integrated Gradients reveal why predictions are made
        
        ### 🚀 Available Tools (after initialization):
        
        1. **Single Prediction**: Analyze individual drug-gene pairs with AI agent
        2. **Batch Analysis**: Process multiple interactions at scale
        3. **3D Visualization**: Explore molecular structures interactively
        4. **Model Explanation**: Understand prediction mechanisms
        5. **Literature Explorer**: Search and analyze research papers
        """
    )
    
    st.markdown("---")
    st.info(
        "**👈 Get started**: Enter your credentials in the sidebar to unlock all features.",
        icon="🔑"
    )
    
else:
    # System is initialized - show architecture and info
    st.success("✅ **System Active** - All tools are now available in the sidebar", icon="🟢")
    
    st.markdown("---")
    st.markdown("## 🏗️ System Architecture")
    
    # Try to display architecture diagram
    image_path = st.session_state.get("architecture_image_path", "app/assets/architecture.png")
    img_file = Path(image_path)
    
    if img_file.is_file():
        st.image(
            str(img_file),
            caption="Model Architecture: EGNN + ESM-2 with Cross-Attention Fusion",
            use_container_width=True
        )
    else:
        st.warning(
            f"⚠️ Architecture diagram not found at `{image_path}`. "
            f"You can upload one to display the model structure.",
            icon="🖼️"
        )
    
    st.markdown("---")
    st.markdown("## 📖 Quick Start Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            ### 🎯 For Quick Analysis:
            
            1. Go to **Single Prediction** in sidebar
            2. Enter drug name (e.g., "Warfarin")
            3. Enter gene name (e.g., "CYP2C9")
            4. Click "Run Analysis"
            
            The AI agent will autonomously:
            - Fetch molecular data
            - Generate 3D structures
            - Predict interaction probability
            - Explain the prediction
            - Find supporting papers
            """
        )
    
    with col2:
        st.markdown(
            """
            ### 📊 For Batch Processing:
            
            1. Go to **Batch Analysis** in sidebar
            2. Prepare CSV with columns:
               - `gene_id`
               - `chem_id`
            3. Upload and process
            4. Download results
            
            Perfect for:
            - Drug screening campaigns
            - Systematic interaction mapping
            - Large-scale validation studies
            """
        )
    
    st.markdown("---")
    st.info(
        "💡 **Pro Tip**: Use the **Literature Explorer** to discover research trends and "
        "generate comprehensive reports for grant proposals or publications.",
        icon="📚"
    )
    
    st.markdown("---")
    st.error(
        "⚠️ **DISCLAIMER**: This tool is for **research purposes only**. "
        "It is **NOT** intended for clinical decision-making without expert validation. "
        "Always consult healthcare professionals and validate findings experimentally.",
        icon="⚠️"
    )

# ==========================================
# Footer
# ==========================================
st.markdown("---")
st.caption(
    "🔬 **Powered by**: PyTorch Geometric • Hugging Face Transformers • LangChain • PubMed E-utilities\n\n"
    "📧 **Support**: For issues, check logs or contact your system administrator"
)
