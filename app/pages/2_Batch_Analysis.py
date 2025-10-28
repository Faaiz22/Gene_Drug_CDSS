# app/pages/2_Batch_Analysis.py (COMPLETE REPLACEMENT)
import streamlit as st
import pandas as pd
import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import time

st.set_page_config(page_title="Batch Analysis", page_icon="📂", layout="wide")

st.title("📂 Batch Analysis")

st.markdown("""
Upload a file with drug-gene pairs for batch prediction analysis.

**File Format Requirements:**
- CSV or TSV format
- Required columns: `gene_id`, `chem_id`
- Maximum 1000 pairs per batch
""")

uploaded_file = st.file_uploader(
    "Choose a file", 
    type=['csv', 'tsv'],
    help="Upload CSV/TSV with gene_id and chem_id columns"
)

def process_single_pair(row: pd.Series) -> Dict:
    """Process a single pair synchronously (runs in thread)"""
    from app.caching_utils import get_cached_prediction
    
    try:
        result = get_cached_prediction(
            gene_id=str(row['gene_id']),
            chem_id=str(row['chem_id'])
        )
        return result
    except Exception as e:
        return {
            "gene_id": row['gene_id'],
            "chem_id": row['chem_id'],
            "probability": None,
            "error": str(e)
        }


def run_batch_processing(dataframe: pd.DataFrame, max_workers: int = 4) -> pd.DataFrame:
    """
    Process batch using ThreadPoolExecutor for true parallelism.
    Streamlit-compatible (no async event loop issues).
    """
    results = []
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_pair, row)
            for _, row in dataframe.iterrows()
        ]
        
        for i, future in enumerate(futures):
            result = future.result()
            results.append(result)
            
            # Update progress
            progress = (i + 1) / len(futures)
            progress_bar.progress(progress)
            
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (len(futures) - i - 1)
            
            status_text.text(
                f"Processing: {i + 1}/{len(futures)} | "
                f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
            )
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ Complete! Total time: {time.time() - start_time:.1f}s")
    
    return pd.DataFrame(results)


if uploaded_file is not None:
    try:
        # Read the file
        content = uploaded_file.getvalue()
        
        if uploaded_file.name.endswith('.tsv'):
            dataframe = pd.read_csv(io.BytesIO(content), sep='\t')
        else:
            dataframe = pd.read_csv(io.BytesIO(content))
        
        # Validate columns
        required_cols = ['gene_id', 'chem_id']
        missing_cols = [col for col in required_cols if col not in dataframe.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info("Your file must contain: `gene_id` and `chem_id` columns")
            st.stop()
        
        # Validate size
        if len(dataframe) > 1000:
            st.error(f"❌ File too large ({len(dataframe)} rows). Maximum 1000 pairs allowed.")
            st.stop()
        
        # Show preview
        st.success(f"✅ Loaded {len(dataframe)} pairs from `{uploaded_file.name}`")
        
        with st.expander("📋 Preview Data", expanded=True):
            st.dataframe(dataframe.head(10), use_container_width=True)
        
        # Configuration
        col1, col2 = st.columns(2)
        with col1:
            max_workers = st.slider(
                "Parallel Workers",
                min_value=1,
                max_value=8,
                value=4,
                help="Number of parallel processing threads"
            )
        
        # Run button
        if st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
            
            st.markdown("---")
            
            with st.spinner("Processing batch..."):
                results_df = run_batch_processing(dataframe, max_workers=max_workers)
            
            # Display results
            st.markdown("### 📊 Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            successful = results_df['probability'].notna().sum()
            failed = results_df['probability'].isna().sum()
            avg_prob = results_df['probability'].mean()
            high_prob = (results_df['probability'] > 0.7).sum()
            
            col1.metric("Total Pairs", len(results_df))
            col2.metric("Successful", successful)
            col3.metric("Failed", failed)
            col4.metric("High Probability (>70%)", high_prob)
            
            # Results table
            st.dataframe(
                results_df.sort_values('probability', ascending=False, na_position='last'),
                use_container_width=True
            )
            
            # Download button
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_data,
                file_name=f'batch_predictions_{time.strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                use_container_width=True
            )
            
            # Show errors if any
            errors = results_df[results_df['error'].notna()]
            if not errors.empty:
                with st.expander(f"⚠️ Errors ({len(errors)} pairs)", expanded=False):
                    st.dataframe(errors[['gene_id', 'chem_id', 'error']], use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)
