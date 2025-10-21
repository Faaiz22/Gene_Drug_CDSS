import streamlit as st

st.set_page_config(
    page_title="Drug-Gene CDSS",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Drug-Gene Interaction Clinical Decision Support System")

st.markdown("""
Welcome to the Drug-Gene Interaction CDSS. This tool uses a deep learning model to predict the likelihood of interaction between a specified drug and a gene.

**👈 Select a page from the sidebar to get started:**

- **Single Prediction**: Predict the interaction for a single drug-gene pair.
- **Batch Analysis**: Upload a file with multiple pairs for analysis.
- **3D Visualization**: View the 3D molecular structures used by the model.
- **Model Explanation**: Understand which features are most important for a prediction.

This system is for research purposes only and should not be used for clinical decision-making.
""")

st.info("Note: The application is currently under development. Model loading and prediction functionality will be integrated soon.")
