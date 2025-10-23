# app/pages/1_Single_Prediction.py
import streamlit as st
import torch
from app.caching_utils import load_all_resources, featurize_pair # <-- Import our new functions

# --- Load all resources via cache ---
# This runs only ONCE at app startup
try:
    config, enricher, protein_featurizer, model, device = load_all_resources()
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
        with st.spinner("Running prediction..."):
            try:
                # 1. Fetch raw data (uses cache from api_clients.py)
                with st.spinner("Fetching sequence and SMILES..."):
                    gene_seq = enricher.fetch_sequence(gene_id)
                    smiles = enricher.fetch_smiles(chem_id)

                if not gene_seq:
                    st.error(f"Could not retrieve sequence for Gene ID: {gene_id}")
                elif not smiles:
                    st.error(f"Could not retrieve SMILES for Chemical ID: {chem_id}")
                else:
                    # 2. Featurize the pair (uses @st.cache_data)
                    with st.spinner("Featurizing inputs... (may be slow on first run)"):
                        graph_batch, protein_emb = featurize_pair(
                            smiles, gene_seq, protein_featurizer, config, device
                        )

                    # 3. Run prediction
                    with st.spinner("Running model inference..."):
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
