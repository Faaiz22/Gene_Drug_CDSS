import streamlit as st
import pandas as pd

st.set_page_config(page_title="Batch Analysis", page_icon="📂")

st.title("📂 Batch Analysis")

st.markdown("Upload a TSV or CSV file with 'gene_id' and 'chem_id' columns to predict interactions for multiple pairs at once.")

# app/pages/2_Batch_Analysis.py
import streamlit as st
import pandas as pd
import time
from app.pages.caching_utils import load_resources # Reuse the cached loader

st.set_page_config(page_title="Batch Analysis", page_icon="📂")
st.title("📂 Batch Analysis")
st.markdown("Upload a TSV or CSV file with 'gene_id' and 'chem_id' columns...")

# Load the cached model and enricher
enricher, model = load_resources()

uploaded_file = st.file_uploader("Choose a file", type=['csv', 'tsv'])

if uploaded_file is not None:
    try:
        dataframe = pd.read_csv(uploaded_file, sep='\t' if 'tsv' in uploaded_file.name else ',')
        st.success(f"Successfully loaded {len(dataframe)} pairs from `{uploaded_file.name}`.")
        
        if 'gene_id' not in dataframe.columns or 'chem_id' not in dataframe.columns:
            st.error("The uploaded file must contain 'gene_id' and 'chem_id' columns.")
        else:
            st.dataframe(dataframe.head())
            
            if st.button("Run Batch Analysis", type="primary"):
                with st.spinner(f"Processing {len(dataframe)} pairs... This may take time."):
                    results = []
                    progress_bar = st.progress(0, text="Starting batch analysis...")
                    
                    total_pairs = len(dataframe)
                    
                    for i, row in dataframe.iterrows():
                        gene_id = str(row['gene_id'])
                        chem_id = str(row['chem_id'])
                        
                        try:
                            # 1. Fetch data
                            gene_seq = enricher.fetch_sequence(gene_id)
                            smiles = enricher.fetch_smiles(chem_id)
                            
                            if not gene_seq or not smiles:
                                probability = None
                                error = "Data not found"
                            else:
                                # 2. Predict
                                probability = model.predict(gene_seq, smiles) # Pass correct features
                                error = None
                                
                        except Exception as e:
                            probability = None
                            error = str(e)

                        # 3. Store result
                        results.append({
                            'gene_id': gene_id,
                            'chem_id': chem_id,
                            'predicted_probability': probability,
                            'error': error
                        })
                        
                        # 4. Update progress
                        percent_complete = (i + 1) / total_pairs
                        progress_bar.progress(percent_complete, text=f"Processed pair {i+1}/{total_pairs} ({gene_id} / {chem_id})")

                    st.success("Batch analysis complete!")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)

                    st.download_button(
                        label="Download Results as CSV",
                        data=results_df.to_csv(index=False).encode('utf-8'),
                        file_name='batch_prediction_results.csv',
                        mime='text/csv',
                    )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

# Note: Create a new file `app/pages/caching_utils.py` to store the
# `load_resources` function so it can be shared by all pages.

uploaded_file = st.file_uploader("Choose a file", type=['csv', 'tsv'])

if uploaded_file is not None:
    try:
        dataframe = pd.read_csv(uploaded_file, sep='\t' if 'tsv' in uploaded_file.name else ',')
        st.success(f"Successfully loaded {len(dataframe)} pairs from `{uploaded_file.name}`.")
        st.dataframe(dataframe.head())

        if 'gene_id' not in dataframe.columns or 'chem_id' not in dataframe.columns:
            st.error("The uploaded file must contain 'gene_id' and 'chem_id' columns.")
        else:
            if st.button("Run Batch Analysis", type="primary"):
                with st.spinner("Processing batch... This may take some time."):
                    # Placeholder for batch prediction logic
                    # results = batch_predict(dataframe)
                    progress_bar = st.progress(0)
                    # for i, row in enumerate(results):
                    #     ... update progress_bar
                    st.success("Batch analysis complete!")

                    # --- Display Results (Placeholder) ---
                    placeholder_results = dataframe.head().copy()
                    placeholder_results['predicted_probability'] = [0.92, 0.13, 0.75, 0.05, 0.88]
                    st.dataframe(placeholder_results)

                    st.download_button(
                        label="Download Results as CSV",
                        data=placeholder_results.to_csv(index=False).encode('utf-8'),
                        file_name='batch_prediction_results.csv',
                        mime='text/csv',
                    )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
