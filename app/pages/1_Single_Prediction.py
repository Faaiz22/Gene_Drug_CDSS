import streamlit as st
import torch
from app.caching_utils import get_cached_prediction # <-- Import our new function

# --- Page Setup ---
# All resources are loaded on demand by get_cached_prediction
try:
    st.set_page_config(page_title="Single Prediction", page_icon="🎯")
except Exception as e:
    # This might fail if streamlit is already set_page_config'd
    pass

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
            # 1. Run the cached prediction
            # Streamlit will 'await' this async function
            # and show a spinner.
            result = get_cached_prediction(gene_id, chem_id)

            st.success(f"Prediction for **{gene_id}** and **{chem_id}** complete!")

            # --- Display Real Results ---
            prob_percent = result['probability'] * 100
            delta_label = "High Likelihood" if result['probability'] > 0.5 else "Low Likelihood"
            
            st.metric(label="Interaction Probability", value=f"{prob_percent:.1f}%", delta=delta_label)
            st.progress(result['probability'])

            with st.expander("View Retrieved Data & Features"):
                st.json({
                    "gene_id": result['gene_id'],
                    "chem_id": result['chem_id'],
                    "retrieved_smiles": result['smiles'],
                    "retrieved_sequence_length": result['sequence_length'],
                    "graph_nodes (atoms)": result['graph_nodes'],
                    "protein_embedding_shape": result['protein_embedding_shape'],
                    "model_prediction": f"{result['probability']:.4f}",
                })
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            # You can add more detailed error logging here
            # st.exception(e)
