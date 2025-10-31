import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Import CDSS modules
try:
    from agent.agent_orchestrator import DTIAgentOrchestrator
    from utils.exceptions import CDSSException
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error("Please ensure all dependencies are installed and the system is properly initialized.")
    st.stop()

st.set_page_config(
    page_title="Single Prediction - Agentic CDSS",
    page_icon="🎯",
    layout="wide"
)
@st.cache_resource
def get_agent():
    # --- Check if processor is loaded ---
    if "core_processor" not in st.session_state:
        st.error("Core processor not loaded. Please go to the 'CDSS Diagnostics' page, enter credentials, and click 'Load Processor'.")
        st.stop()
        
    # --- Check if API key is in session_state ---
    if "google_api_key" not in st.session_state or not st.session_state.google_api_key:
        st.error("Google API Key not provided. Please enter it on the main 'CDSS Diagnostics' page.")
        st.stop()

    # --- Get config and API key from session_state (put there by main.py) ---
    config = st.session_state.config
    api_key = st.session_state.google_api_key
    
    try:
        # Pass the key directly. The orchestrator's __init__ will handle it.
        return DTIAgentOrchestrator(config, api_key=api_key)
    except Exception as e:
        st.error(f"💥 Failed to initialize agent. This may be due to an invalid Google API Key.")
        st.exception(e)
        st.stop()


st.title("🎯 Agentic Drug-Gene Interaction Analysis")
st.markdown("""
This advanced system uses an **autonomous AI agent** to:
- 🔍 Fetch molecular data from public databases
- 🧬 Predict interaction probability using deep learning
- 💡 Explain predictions at the molecular level
- 📚 Search PubMed for supporting research
- 📋 Generate comprehensive clinical reports
""")

# Check for processor again before showing UI
if "core_processor" not in st.session_state:
    st.info("Waiting for Core Processor to be loaded on the main page...")
    st.stop()

# --- (Rest of the file is unchanged) ---

# Sidebar for analysis options
with st.sidebar:
    st.header("⚙️ Analysis Options")
    
    analysis_mode = st.radio(
        "Analysis Depth",
        ["Quick Prediction", "Standard Analysis", "Full Report"],
        index=1
    )
    
    st.markdown("---")
    
    include_explanation = st.checkbox(
        "Include Molecular Explanation",
        value=(analysis_mode != "Quick Prediction"),
        help="Generate detailed explanation using Integrated Gradients"
    )
    
    include_literature = st.checkbox(
        "Search PubMed Literature",
        value=(analysis_mode == "Full Report"),
        help="Find relevant research papers"
    )
    
    generate_report = st.checkbox(
        "Generate Clinical Report",
        value=(analysis_mode == "Full Report"),
        help="Create comprehensive clinical summary"
    )
    
    save_visualizations = st.checkbox(
        "Save Visualizations",
        value=True,
        help="Save molecular structure and attention plots"
    )

# Main content
col1, col2 = st.columns(2)

with col1:
    drug_name = st.text_input(
        "Drug Name or ID",
        placeholder="e.g., Imatinib, CID 5330",
        help="Enter drug name, PubChem CID, or PharmGKB ID"
    )
    drug_id = st.text_input(
        "Drug ID (Optional)",
        placeholder="e.g., CID 5330, PA448515",
        help="Provide specific ID for more accurate results"
    )

with col2:
    gene_name = st.text_input(
        "Gene/Protein Name or ID",
        placeholder="e.g., ABL1, CYP2D6",
        help="Enter gene symbol, protein name, or UniProt ID"
    )
    gene_id = st.text_input(
        "Gene ID (Optional)",
        placeholder="e.g., P00519, PA24356",
        help="Provide specific ID for more accurate results"
    )

# Analysis button
if st.button("🚀 Run Agentic Analysis", type="primary", use_container_width=True):
    if not drug_name or not gene_name:
        st.error("⚠️ Please provide both drug and gene names")
    else:
        try:
            agent = get_agent()
            
            # Build query
            query = f"Analyze the interaction between drug '{drug_name}'"
            if drug_id:
                query += f" (ID: {drug_id})"
            query += f" and gene '{gene_name}'"
            if gene_id:
                query += f" (ID: {gene_id})"
            query += "."
            
            # Run analysis with progress tracking
            with st.spinner("🤖 Agent is thinking and using tools..."):
                result = agent.analyze_interaction(
                    query,
                    include_literature=include_literature,
                    include_explanation=include_explanation,
                    generate_report=generate_report
                )
            
            if result['status'] == 'success':
                st.success("✅ Analysis Complete!")
                
                # Display agent's response
                st.markdown("### 🤖 Agent's Analysis")
                st.markdown(result['response'])
                
                # Show intermediate steps in expander
                if result.get('intermediate_steps'):
                    with st.expander("🔍 View Agent's Reasoning Process"):
                        for i, step in enumerate(result['intermediate_steps'], 1):
                            st.markdown(f"**Step {i}:** {step}")
                
                # Display visualizations if saved
                from src.agent.agentic_tools import agent_state
                
                if save_visualizations and 'explanation' in agent_state:
                    viz_paths = agent_state['explanation'].get('visualization_paths', {})
                    
                    if viz_paths:
                        st.markdown("### 📊 Molecular Visualizations")
                        
                        viz_cols = st.columns(len(viz_paths))
                        
                        for i, (viz_type, viz_path) in enumerate(viz_paths.items()):
                            with viz_cols[i]:
                                st.image(
                                    str(viz_path),
                                    caption=viz_type.replace('_', ' ').title(),
                                    use_container_width=True
                                )
                
                # Display literature if searched
                if include_literature and 'literature' in agent_state:
                    papers = agent_state['literature']
                    
                    if papers:
                        st.markdown("### 📚 Supporting Literature")
                        
                        for paper in papers:
                            with st.expander(f"📄 {paper['title']} ({paper['year']})"):
                                st.markdown(f"**Authors:** {paper['authors']}")
                                st.markdown(f"**Journal:** {paper['journal']}")
                                st.markdown(f"**PMID:** {paper['pmid']}")
                                st.markdown(f"**Abstract:** {paper['abstract_preview']}")
                                if paper.get('url'):
                                    st.markdown(f"[🔗 View on PubMed]({paper['url']})")
                
                # Display clinical report if generated
                if generate_report and 'clinical_report' in agent_state:
                    st.markdown("### 📋 Clinical Report")
                    st.code(agent_state['clinical_report'], language=None)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Report",
                        data=agent_state['clinical_report'],
                        file_name=f"DTI_Report_{drug_name}_{gene_name}.txt",
                        mime="text/plain"
                    )
            
            else:
                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)

# Help section
with st.expander("ℹ️ How to Use This Tool"):
    st.markdown("""
    **Step 1:** Enter drug and gene identifiers
    - Use common names (e.g., "Imatinib", "ABL1") or specific IDs
    - Optional IDs improve accuracy
    
    **Step 2:** Choose analysis options
    - **Quick Prediction**: Fast probability estimate only
    - **Standard Analysis**: Includes molecular explanation
    - **Full Report**: Complete analysis with literature and clinical summary
    
    **Step 3:** Run analysis
    - The AI agent will autonomously:
        1. Fetch molecular data from APIs
        2. Generate 3D structures and embeddings
        3. Predict interaction probability
        4. Explain the prediction (if enabled)
        5. Search relevant papers (if enabled)
        6. Synthesize clinical report (if enabled)
    
    **Understanding Results:**
    - **Probability > 70%**: High confidence of interaction
    - **Probability 50-70%**: Moderate confidence
    - **Probability < 50%**: Low confidence
    - Always consider uncertainty estimates and supporting evidence
    """)

st.markdown("---")
st.caption("⚠️ **Disclaimer:** This tool is for research purposes only. Not for clinical decision-making.")
