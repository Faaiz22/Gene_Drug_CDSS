import streamlit as st
import matplotlib.pyplot as plt

# You may want to wrap your imports for robustness on deployment!
try:
    from src.models.explainer import ModelExplainer
    from app.pages.caching_utils import load_resources
except ImportError:
    ModelExplainer = None
    load_resources = None


# --- Caching ---
@st.cache_resource
def load_explainer():
    try:
        # If real data loader is present, use cached model from your infra
        if load_resources is not None and ModelExplainer is not None:
            _, model = load_resources() 
            explainer = ModelExplainer(model)
            return explainer
    except Exception as e:
        st.error(f"Failed to load ModelExplainer: {e}")

    # Fallback: placeholder explainer for dev/UI work
    class PlaceholderExplainer:
        def explain(self, gene_id, chem_id):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            ax1.text(0.5, 0.5, 'Drug Atom Importance\n[Placeholder]', ha='center', fontsize=14)
            ax1.set_title(f"Drug: {chem_id}")
            ax1.axis('off')
            ax2.text(0.5, 0.5, 'Protein Residue Importance\n[Placeholder]', ha='center', fontsize=14)
            ax2.set_title(f"Gene: {gene_id}")
            ax2.axis('off')
            fig.tight_layout()
            return fig, None, None

    return PlaceholderExplainer()

@st.cache_data
def get_explanation(gene_id, chem_id):
    explainer = load_explainer()
    fig, drug_img, protein_img = explainer.explain(gene_id, chem_id)
    return fig, drug_img, protein_img

# --- Page UI ---
st.set_page_config(page_title="Model Explanation", page_icon="💡")
st.title("💡 Model Explanation")
st.markdown("""
This page provides insights into why the model made its prediction for a **gene-drug pair**.
Advanced methods like *Integrated Gradients*, *attention scores*, or *feature attribution* highlight which atoms and residues were most influential.
""")

# --- Input Widgets ---
col1, col2 = st.columns(2)
with col1:
    gene_id = st.text_input("Gene Identifier", placeholder="e.g., CYP2C9")
with col2:
    chem_id = st.text_input("Chemical Identifier", placeholder="e.g., celecoxib")

# UX: Clearly show development status
st.info("This feature is under development. Results may be placeholders if the backend model is not running.", icon="⚙️")

if st.button("Explain Prediction", type="primary"):
    if not gene_id.strip() or not chem_id.strip():
        st.warning("Please enter both identifiers to generate an explanation.")
    else:
        with st.spinner("Generating explanation... This may take 5–30 seconds for large models."):
            try:
                fig, drug_img, protein_img = get_explanation(gene_id, chem_id)
                st.success("Explanation generated!")

                st.subheader("Feature Attributions")
                st.write("Charts below show which drug atoms and protein residues contributed most to the prediction.")
                
                # Column layout for results
                results_1, results_2 = st.columns([2, 3])
                with results_1:
                    st.pyplot(fig)
                with results_2:
                    if drug_img is not None:
                        st.image(drug_img, caption="Drug Atom Importances", use_column_width="always")
                    else:
                        st.image("https://via.placeholder.com/400x300?text=Drug+Atom+Importance", caption="Atom importances on structure (placeholder)")
                    if protein_img is not None:
                        st.image(protein_img, caption="Protein Residue Importances", use_column_width="always")
                    else:
                        st.image("https://via.placeholder.com/400x300?text=Protein+Residue+Importance", caption="Residue importances on sequence (placeholder)")
            except Exception as e:
                st.error(f"Failed to generate explanation: {e}")

st.markdown("---")
st.markdown("**Tips:**\n- If the real model is deployed, results will appear automatically. If placeholders show, deploy backend model and update resource loading logic.")
