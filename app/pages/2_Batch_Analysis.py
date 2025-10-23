import streamlit as st
import pandas as pd
import asyncio
import io
from app.caching_utils import get_cached_prediction # Import our cached function

st.set_page_config(page_title="Batch Analysis", page_icon="📂")
st.title("📂 Batch Analysis")
st.markdown("Upload a TSV or CSV file with 'gene_id' and 'chem_id' columns...")
st.warning("Note: This feature is for small batches (<100 pairs). Large files may time out.")

uploaded_file = st.file_uploader("Choose a file", type=['csv', 'tsv'])

async def run_batch_processing(dataframe):
    """
    Creates and runs all prediction tasks concurrently.
    """
    tasks = []
    for _, row in dataframe.iterrows():
        # Create a task for each row
        tasks.append(
            get_cached_prediction(
                gene_id=str(row['gene_id']),
                chem_id=str(row['chem_id'])
            )
        )
    
    st.info(f"Submitting {len(tasks)} pairs for processing...")
    progress_bar = st.progress(0, text="Running predictions...")
    
    # Run all tasks concurrently
    # Note: We can't show granular progress with asyncio.gather
    # We will just update the bar as results come in (or at the end)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    progress_bar.progress(1.0, text="Processing complete!")
    
    # Process results
    final_results = []
    for i, res in enumerate(results):
        original_row = dataframe.iloc[i]
        if isinstance(res, Exception):
            final_results.append({
                "gene_id": original_row['gene_id'],
                "chem_id": original_row['chem_id'],
                "probability": None,
                "error": str(res)
            })
        else:
            final_results.append({
                "gene_id": res['gene_id'],
                "chem_id": res['chem_id'],
                "probability": res['probability'],
                "error": None
            })
            
    return pd.DataFrame(final_results)

if uploaded_file is not None:
    try:
        # Read the file content into a dataframe
        content = uploaded_file.getvalue()
        if 'tsv' in uploaded_file.name:
            dataframe = pd.read_csv(io.BytesIO(content), sep='\t')
        else:
            dataframe = pd.read_csv(io.BytesIO(content), sep=',')
            
        st.success(f"Successfully loaded {len(dataframe)} pairs from `{uploaded_file.name}`.")
        st.dataframe(dataframe.head())

        if 'gene_id' not in dataframe.columns or 'chem_id' not in dataframe.columns:
            st.error("The uploaded file must contain 'gene_id' and 'chem_id' columns.")
        else:
            if st.button("Run Batch Analysis", type="primary"):
                with st.spinner(f"Processing {len(dataframe)} pairs... This may take time."):
                    try:
                        # Run the async function
                        # Streamlit automatically handles the event loop
                        results_df = asyncio.run(run_batch_processing(dataframe))
                        
                        st.success("Batch analysis complete!")
                        st.dataframe(results_df)

                        st.download_button(
                            label="Download Results as CSV",
                            data=results_df.to_csv(index=False).encode('utf-8'),
                            file_name='batch_prediction_results.csv',
                            mime='text/csv',
                        )
                    except Exception as e:
                        st.error(f"An error occurred during batch processing: {e}")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
