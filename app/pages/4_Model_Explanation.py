"""
Model Explanation Page - Understand why predictions are made
"""

import streamlit as st
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Import CDSS modules
try:
    from models.explainer import EnhancedExplainer
    from utils.exceptions import CDSSException
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error("Please ensure all dependencies are installed and the system is properly initialized.")
    st.stop()

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Model Explanation",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Model Explanation & Interpretability")
st.markdown("""
Understand why the model made specific predictions using advanced explainability techniques:
- **Integrated Gradients**: Attribute predictions to specific atoms and residues
- **Attention Weights**: Visualize cross-attention between drug and protein
- **Feature Importance**: Identify key molecular features driving predictions
""")

# Check for core processor
if "core_processor" not in st.session_state:
    st.error(
        "🧬 **Core processor not initialized**\n\n"
        "Please return to the main page, enter your API credentials, "
        "and click 'Initialize System' before using this tool."
    )
    st.info(
        "The explanation system requires:\n"
        "- Loaded DTI prediction model\n"
        "- Initialized feature engineering pipeline\n"
        "- API access for molecular data"
    )
    st.stop()

core_processor = st.session_state.core_processor

# Check if explainer is cached
@st.cache_resource
def load_explainer():
    """Load and cache the explainability module."""
    try:
        config = st.session_state.config
        device = core_processor.model.device if hasattr(core_processor.model, 'device') else 'cpu'
        
        explainer = EnhancedExplainer(
            model=core_processor.model,
            config=config,
            device=device
        )
        return explainer
    except Exception as e:
        logger.error(f"Failed to initialize explainer: {e}")
        raise CDSSException(
            "Failed to initialize explainability module",
            str(e)
        )

# Cached explanation generation
@st.cache_data(show_spinner=False)
def get_explanation(gene_id: str, chem_id: str):
    """
    Generate and cache explanation for a gene-drug pair.
    """
    try:
        # Preprocess data
        data_input = core_processor.get_preprocessed_data_for_pair(gene_id, chem_id)
        
        if data_input is None:
            raise CDSSException(
                "Preprocessing failed",
                "Could not fetch or process molecular data for the given identifiers"
            )
        
        # Load explainer
        explainer = load_explainer()
        
        # Generate explanation
        explanation = explainer.explain_prediction(
            graph=data_input.drug_graph,
            protein_emb=data_input.protein_embedding,
            gene_name=gene_id,
            drug_name=chem_id,
            smiles=data_input.smiles,
            save_dir=None  # Don't save to disk for web app
        )
        
        return explanation
        
    except CDSSException as e:
        raise e
    except Exception as e:
        logger.error(f"Explanation generation error: {e}", exc_info=True)
        raise CDSSException(
            "Explanation generation failed",
            str(e)
        )

# Main UI
st.markdown("---")
st.subheader("🔍 Generate Explanation")

col1, col2 = st.columns(2)

with col1:
    gene_id = st.text_input(
        "Gene Identifier",
        placeholder="e.g., CYP2C9, P12345",
        key="exp_gene_id",
        help="Enter gene symbol, UniProt accession, or PharmGKB ID"
    )

with col2:
    chem_id = st.text_input(
        "Chemical Identifier",
        placeholder="e.g., Warfarin, CID 5330",
        key="exp_chem_id",
        help="Enter drug name, PubChem CID, or PharmGKB ID"
    )

# Explanation options
with st.expander("⚙️ Advanced Options"):
    st.markdown("**Explainability Settings:**")
    
    opt_col1, opt_col2 = st.columns(2)
    
    with opt_col1:
        ig_steps = st.slider(
            "Integrated Gradients Steps",
            min_value=20,
            max_value=100,
            value=50,
            step=10,
            help="More steps = more accurate but slower"
        )
    
    with opt_col2:
        baseline_strategy = st.selectbox(
            "Baseline Strategy",
            ["zero", "gaussian", "uniform"],
            index=0,
            help="Method for computing feature attributions"
        )
    
    show_attention = st.checkbox(
        "Show Attention Weights",
        value=True,
        help="Visualize cross-attention between drug and protein"
    )
    
    show_distribution = st.checkbox(
        "Show Attribution Distribution",
        value=True,
        help="Display distribution of feature importances"
    )

if st.button("🚀 Generate Explanation", type="primary", use_container_width=True):
    if not gene_id.strip() or not chem_id.strip():
        st.warning("⚠️ Please enter both gene and chemical identifiers")
    else:
        # Progress tracking
        progress_container = st.container()
        
        with progress_container:
            with st.spinner("⚙️ Fetching molecular data..."):
                try:
                    # Fetch SMILES and sequence
                    smiles = core_processor.get_smiles_sync(chem_id)
                    sequence = core_processor.get_sequence_sync(gene_id)
                    
                    st.success(f"✅ Data fetched successfully")
                except Exception as e:
                    st.error(f"❌ Data fetching failed: {e}")
                    st.stop()
            
            with st.spinner("🧬 Generating 3D structure and embeddings..."):
                try:
                    # Featurize
                    from molecular_3d.conformer_generator import smiles_to_3d_graph
                    from preprocessing.feature_engineer import FeatureEngineer
                    
                    config = st.session_state.config
                    feature_engineer = FeatureEngineer(config['model'].get('featurization', {}))
                    
                    # Create graph and embeddings
                    drug_graph = smiles_to_3d_graph(smiles, config)
                    protein_emb = feature_engineer.embed_protein(sequence)
                    
                    st.success("✅ Molecular structures generated")
                except Exception as e:
                    st.error(f"❌ Featurization failed: {e}")
                    st.stop()
            
            with st.spinner("🤖 Running model and computing attributions..."):
                try:
                    # Get prediction first
                    prediction = core_processor.run_model(smiles, sequence)
                    prob = prediction['probability']
                    
                    st.success(f"✅ Prediction: {prob:.1%} probability of interaction")
                    
                    # Generate explanation
                    explainer = load_explainer()
                    
                    # Create temporary data object
                    class DataObject:
                        def __init__(self):
                            self.drug_graph = drug_graph
                            self.protein_embedding = protein_emb
                            self.smiles = smiles
                    
                    data_obj = DataObject()
                    
                    explanation = explainer.explain_prediction(
                        graph=drug_graph,
                        protein_emb=protein_emb,
                        gene_name=gene_id,
                        drug_name=chem_id,
                        smiles=smiles,
                        save_dir=None
                    )
                    
                    st.success("✅ Explanation generated successfully")
                    
                except Exception as e:
                    st.error(f"❌ Explanation failed: {e}")
                    logger.exception("Explanation error")
                    st.stop()
        
        # Clear progress container
        progress_container.empty()
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Explanation Results")
        
        # Prediction summary
        st.markdown("### 🎯 Prediction Summary")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                "Interaction Probability",
                f"{explanation['prediction_prob']:.1%}",
                delta="High Confidence" if explanation['prediction_prob'] > 0.7 or explanation['prediction_prob'] < 0.3 else "Moderate"
            )
        
        with metric_col2:
            interpretation = "Likely Interaction" if explanation['prediction_prob'] > 0.5 else "Unlikely Interaction"
            st.metric("Interpretation", interpretation)
        
        with metric_col3:
            confidence_score = 1 - abs(0.5 - explanation['prediction_prob']) * 2
            st.metric("Confidence Score", f"{confidence_score:.1%}")
        
        # Natural language explanation
        st.markdown("### 💬 Natural Language Explanation")
        st.info(explanation['natural_language'])
        
        # Key features
        st.markdown("### 🔑 Key Contributing Features")
        
        feat_col1, feat_col2 = st.columns(2)
        
        with feat_col1:
            st.markdown("**Important Drug Atoms:**")
            if explanation['top_atoms']:
                for i, atom_idx in enumerate(explanation['top_atoms'][:5], 1):
                    importance = explanation['atom_attributions'][atom_idx]
                    st.write(f"{i}. Atom #{atom_idx} (importance: {importance:.3f})")
            else:
                st.write("No significant atoms identified")
        
        with feat_col2:
            st.markdown("**Important Protein Residues:**")
            if explanation['top_residues']:
                for i, res_idx in enumerate(explanation['top_residues'][:5], 1):
                    importance = explanation['protein_attributions'][res_idx]
                    st.write(f"{i}. Residue #{res_idx} (importance: {importance:.3f})")
            else:
                st.write("No significant residues identified")
        
        # Visualizations
        if show_attention and explanation.get('attention_weights') is not None:
            st.markdown("### 🎨 Attention Heatmap")
            
            import numpy as np
            import seaborn as sns
            
            attn = explanation['attention_weights']
            
            if len(attn.shape) > 2:
                attn = attn.mean(axis=0)  # Average across heads
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(attn, cmap='viridis', ax=ax, cbar_kws={'label': 'Attention Weight'})
            ax.set_xlabel('Drug Features')
            ax.set_ylabel('Protein Features')
            ax.set_title('Cross-Attention: Protein → Drug')
            
            st.pyplot(fig)
        
        if show_distribution:
            st.markdown("### 📈 Feature Attribution Distribution")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Drug atoms
            ax1.hist(explanation['atom_attributions'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax1.axvline(explanation['atom_attributions'].mean(), color='red', linestyle='--',
                       label=f"Mean: {explanation['atom_attributions'].mean():.3f}")
            ax1.set_xlabel('Attribution Magnitude')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Drug Atom Attributions')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Protein residues
            ax2.hist(explanation['protein_attributions'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
            ax2.axvline(explanation['protein_attributions'].mean(), color='blue', linestyle='--',
                       label=f"Mean: {explanation['protein_attributions'].mean():.3f}")
            ax2.set_xlabel('Attribution Magnitude')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Protein Residue Attributions')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Technical details
        with st.expander("🔬 Technical Details"):
            st.markdown(f"""
            **Method:** Integrated Gradients with {ig_steps} steps
            
            **Baseline Strategy:** {baseline_strategy}
            
            **Model Architecture:**
            - Drug Encoder: E(n)-Equivariant Graph Neural Network
            - Protein Encoder: ESM-2 Transformer
            - Fusion: Cross-Attention Mechanism
            
            **Attribution Statistics:**
            - Drug Atoms Analyzed: {len(explanation['atom_attributions'])}
            - Protein Residues Analyzed: {len(explanation['protein_attributions'])}
            - Max Drug Attribution: {explanation['atom_attributions'].max():.4f}
            - Max Protein Attribution: {explanation['protein_attributions'].max():.4f}
            
            **Identifiers Used:**
            - Gene: {gene_id}
            - Drug: {chem_id}
            - SMILES: `{smiles[:50]}...`
            - Sequence Length: {len(sequence)} residues
            """)

# Information section
st.markdown("---")
with st.expander("ℹ️ Understanding the Explanations"):
    st.markdown("""
    ### How to Interpret Results
    
    **Prediction Probability:**
    - **>70%**: High confidence of interaction
    - **50-70%**: Moderate confidence
    - **<50%**: Low likelihood of interaction
    
    **Feature Attributions:**
    - Higher values indicate greater importance
    - Positive values contribute to interaction prediction
    - Focus on top 5-10 features for interpretation
    
    **Attention Weights:**
    - Show which parts of the drug the model "pays attention to" relative to the protein
    - Brighter colors indicate stronger attention
    
    **Natural Language Explanation:**
    - Automatically generated summary of key findings
    - Identifies critical molecular features
    - Provides biological context when available
    
    ### Limitations
    - Explanations are model-dependent (not ground truth)
    - Attribution scores are relative, not absolute
    - Complex interactions may involve many weak features
    - Always validate findings with experimental data
    """)

st.caption("💡 Powered by Captum Integrated Gradients and ESM-2 Attention")
