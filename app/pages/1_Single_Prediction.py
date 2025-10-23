import streamlit as st
import torch
from app.caching_utils import (
    load_config, load_enricher, 
    load_sync_resources, featurize_pair
)

# --- Load all resources via cache ---
try:
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config and all resources
    config = load_config()
    enricher = load_enricher(config)
    model, protein_featurizer = load_sync_resources(config, device)
    
    st.set_page_config(page_title="Single Prediction", page_icon="🎯")
except Exception as e:
    st.error(f"Fatal error during resource loading: {e}")
    st.stop()


st.title("🎯 Single Pair Prediction")
st.markdown("Enter a gene identifier and a chemical identifier to predict their interaction probability.")

# --- Input Form ---
col1, col2 = st.columns(2)
with col1:
    gene_id = st.text_input("Gene Identifier", placeholder="e.g., CYP2C9 or PA4450", value="CYP2C9")
with col2:
    chem_id = st.text_input("Chemical Identifier", placeholder="e.g., celecoxib or PA44836", value="celecoxib")

if st.button("Predict Interaction", type="primary"):
    if not gene_id or not chem_id:
        st.warning("Please enter both a gene and a chemical identifier.")
    else:
        try:
            # 1. Featurize the pair
            # Streamlit will automatically 'await' this async cached function
            # and show a spinner.
            graph_batch, protein_emb, smiles, gene_seq = featurize_pair(
                enricher, protein_featurizer, gene_id, chem_id, config, device
            )

            # 2. Run prediction (Sync)
            with torch.no_grad():
                logits, attn_weights = model(graph_batch, protein_emb)
                probability = torch.sigmoid(logits).item()
            
            st.success(f"Prediction for **{gene_id}** and **{chem_id}** complete!")

            # --- Display Real Results ---
            prob_percent = probability * 100
            delta_label = "High Likelihood" if probability > 0.5 else "Low Likelihood"
            
            st.metric(label="Interaction Probability", value=f"{prob_percent:.1f}%", delta=delta_label)
            st.progress(probability)

            with st.expander("View Retrieved Data & Features"):
                st.json({
                    "gene_id": gene_id,
                    "chem_id": chem_id,
                    "retrieved_smiles": smiles,
                    "retrieved_sequence_length": len(gene_seq),
                    "graph_nodes (atoms)": graph_batch.x.shape[0],
                    "protein_embedding_shape": list(protein_emb.shape),
                    "model_prediction": f"{probability:.4f}",
                })
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
