import streamlit as st

st.set_page_config(page_title="Single Prediction", page_icon="🎯")

st.title("🎯 Single Pair Prediction")

st.markdown("Enter a gene identifier and a chemical identifier to predict their interaction probability.")

# app/pages/1_Single_Prediction.py
import streamlit as st
import yaml  # Need to load config
from src.utils.api_clients import DataEnricher
from src.models.dti_model import YourDTIModel  # <-- Import your actual model class
from src.preprocessing.feature_engineer import featurize_pair # <-- Import your featurizer

# --- Caching: Load models and clients once ---
@st.cache_resource
def load_resources():
    """Loads the model, config, and enricher at app start."""
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    enricher = DataEnricher(config)
    
    # model = YourDTIModel(...) # <-- Load your trained torch model
    # model.load_state_dict(torch.load(config['paths']['model_weights']))
    # model.eval()
    
    # Placeholder model until yours is ready
    class PlaceholderModel:
        def predict(self, gene_seq, smiles):
            # Your real model logic goes here
            import random
            return random.uniform(0, 1)
            
    model = PlaceholderModel()
    return enricher, model

# Load the resources
enricher, model = load_resources()

st.set_page_config(page_title="Single Prediction", page_icon="🎯")
st.title("🎯 Single Pair Prediction")
st.markdown("Enter a gene identifier and a chemical identifier to predict their interaction probability.")

# --- Input Form ---
col1, col2 = st.columns(2)
with col1:
    gene_id = st.text_input("Gene Identifier", placeholder="e.g., CYP2C9 or PA4450")
with col2:
    chem_id = st.text_input("Chemical Identifier", placeholder="e.g., celecoxib or PA44836")

if st.button("Predict Interaction", type="primary"):
    if not gene_id or not chem_id:
        st.warning("Please enter both a gene and a chemical identifier.")
    else:
        with st.spinner("Running prediction... This may take a moment."):
            try:
                # 1. Fetch data (this will be slow without async or caching)
                gene_seq = enricher.fetch_sequence(gene_id)
                smiles = enricher.fetch_smiles(chem_id)

                if not gene_seq:
                    st.error(f"Could not retrieve sequence for Gene ID: {gene_id}")
                elif not smiles:
                    st.error(f"Could not retrieve SMILES for Chemical ID: {chem_id}")
                else:
                    # 2. Run prediction
                    # This is where your featurization would happen
                    # e.g., graph = smiles_to_3d_graph(smiles, ...)
                    # e.g., protein_emb = protein_to_embedding(gene_seq, ...)
                    probability = model.predict(gene_seq, smiles) # Pass correct features
                    
                    st.success(f"Prediction for **{gene_id}** and **{chem_id}** complete!")

                    # --- Display Real Results ---
                    prob_percent = probability * 100
                    delta_label = "High Likelihood" if probability > 0.5 else "Low Likelihood"
                    
                    st.metric(label="Interaction Probability", value=f"{prob_percent:.1f}%", delta=delta_label)
                    st.progress(probability)

                    with st.expander("View Retrieved Data"):
                        st.json({
                            "gene_id": gene_id,
                            "chem_id": chem_id,
                            "retrieved_sequence_length": len(gene_seq),
                            "retrieved_smiles": smiles,
                            "model_prediction": f"{probability:.4f}",
                        })
            
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# --- Input Form ---
col1, col2 = st.columns(2)
with col1:
    gene_id = st.text_input("Gene Identifier", placeholder="e.g., CYP2C9 or PA4450")
with col2:
    chem_id = st.text_input("Chemical Identifier", placeholder="e.g., celecoxib or PA44836")

if st.button("Predict Interaction", type="primary"):
    if not gene_id or not chem_id:
        st.warning("Please enter both a gene and a chemical identifier.")
    else:
        with st.spinner("Running prediction... This may take a moment."):
            # Placeholder for model prediction logic
            # result = predict_interaction(model, gene_id, chem_id, ...)
            st.success(f"Prediction for **{gene_id}** and **{chem_id}** complete!")

            # --- Display Results (Placeholder) ---
            st.metric(label="Interaction Probability", value="87.5%", delta="High Likelihood")
            st.progress(0.875)

            with st.expander("View Details"):
                st.json({
                    "gene_id": gene_id,
                    "chem_id": chem_id,
                    "retrieved_sequence_length": 1470,
                    "retrieved_smiles": "CC1=CC=C(C=C1)C2=C(C(=NN2)C(F)(F)F)S(=O)(=O)N",
                    "model_confidence_std": 0.05,
                    "interpretation": "The model predicts a high probability of interaction with low uncertainty."
                })
