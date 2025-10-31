"""
Batch Analysis Page - Process multiple drug-gene pairs simultaneously
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Import CDSS modules using absolute imports
try:
    from utils.validators import validate_batch_dataframe
    from utils.exceptions import CDSSException
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error("Please ensure all dependencies are installed and the system is properly initialized.")
    st.stop()

st.set_page_config(
    page_title="Batch DTI Analysis",
    layout="wide",
    page_icon="🧪"
)

st.title("🧪 Batch DTI Analysis")
st.markdown("Upload a CSV/Excel file with `gene_id` and `chem_id` columns to run predictions in bulk.")

# Check for core processor
if "core_processor" not in st.session_state:
    st.error(
        "⚠️ **Core processor not initialized**\n\n"
        "Please go to the **main page** (Home), enter your API credentials, "
        "and click '**Initialize System**' before using batch analysis.",
        icon="🔐"
    )
    st.info(
        "**Why is this needed?**\n\n"
        "The batch analysis tool requires:\n"
        "- Loaded machine learning models\n"
        "- API credentials for data fetching\n"
        "- Initialized feature engineering pipeline\n\n"
        "All of these are set up on the main page.",
        icon="💡"
    )
    st.stop()

core_processor = st.session_state.core_processor

# Instructions
with st.expander("📋 How to Prepare Your Batch File"):
    st.markdown("""
    **Required Columns:**
    - `gene_id`: Gene symbol, UniProt ID, or PharmGKB ID (e.g., "CYP2D6", "P12345")
    - `chem_id`: Drug name, PubChem CID, or PharmGKB ID (e.g., "Warfarin", "5330")
    
    **Example CSV Format:**
    ```
    gene_id,chem_id
    CYP2D6,Warfarin
    BRCA1,Tamoxifen
    EGFR,Gefitinib
    ```
    
    **Supported Formats:**
    - CSV (`.csv`)
    - Excel (`.xlsx`, `.xls`)
    
    **Limitations:**
    - Maximum 1000 pairs per upload
    - Each identifier will be validated before processing
    """)

# File uploader
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Upload a file containing gene_id and chem_id columns"
)

if uploaded_file:
    try:
        # Read file based on extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.error("Please ensure the file is properly formatted and not corrupted.")
        st.stop()

    # Preview the uploaded data
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    if len(df) > 10:
        st.info(f"Showing first 10 rows of {len(df)} total pairs")

    # Validation
    st.subheader("🔍 Data Validation")
    
    with st.spinner("Validating data format..."):
        validation_errors = validate_batch_dataframe(df)
    
    if validation_errors:
        st.error("❌ **Validation Failed** - Please fix the following errors:")
        for error in validation_errors:
            st.warning(f"• {error}")
        
        st.markdown("---")
        st.info(
            "**Tips:**\n"
            "- Ensure all required columns are present\n"
            "- Remove or fill any empty cells\n"
            "- Check for special characters in identifiers\n"
            "- Verify identifiers are in correct format"
        )
        st.stop()
    else:
        st.success(f"✅ **Validation Passed** - {len(df)} pairs ready for analysis")

    # Analysis configuration
    st.subheader("⚙️ Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_uncertainty = st.checkbox(
            "Include Uncertainty Estimates",
            value=True,
            help="Use Monte Carlo Dropout to estimate prediction uncertainty"
        )
    
    with col2:
        save_results = st.checkbox(
            "Save Detailed Results",
            value=True,
            help="Save full results including raw scores and metadata"
        )

    # Run analysis button
    st.markdown("---")
    
    if st.button(
        f"🚀 Run Batch Analysis on {len(df)} Pairs",
        type="primary",
        use_container_width=True
    ):
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        failed_pairs = []
        
        # Process each pair
        for idx, row in df.iterrows():
            # Update progress
            progress = (idx + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"Processing pair {idx + 1}/{len(df)}: {row['gene_id']} - {row['chem_id']}")
            
            try:
                # Fetch data
                smiles = core_processor.get_smiles_sync(row['chem_id'])
                sequence = core_processor.get_sequence_sync(row['gene_id'])
                
                # Run prediction
                prediction = core_processor.run_model(smiles, sequence)
                
                # Store result
                result = {
                    'gene_id': row['gene_id'],
                    'chem_id': row['chem_id'],
                    'probability': prediction['probability'],
                    'prediction': 'Interaction' if prediction['prediction'] == 1 else 'No Interaction',
                    'confidence': 'High' if prediction['probability'] > 0.7 or prediction['probability'] < 0.3 else 'Moderate',
                }
                
                results.append(result)
                
            except Exception as e:
                failed_pairs.append({
                    'gene_id': row['gene_id'],
                    'chem_id': row['chem_id'],
                    'error': str(e)
                })
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        if results:
            results_df = pd.DataFrame(results)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Successfully Processed",
                    f"{len(results)}/{len(df)}",
                    delta=f"{len(results)/len(df)*100:.1f}%"
                )
            
            with col2:
                interactions = sum(1 for r in results if r['prediction'] == 'Interaction')
                st.metric(
                    "Predicted Interactions",
                    interactions,
                    delta=f"{interactions/len(results)*100:.1f}%"
                )
            
            with col3:
                high_conf = sum(1 for r in results if r['confidence'] == 'High')
                st.metric(
                    "High Confidence",
                    high_conf,
                    delta=f"{high_conf/len(results)*100:.1f}%"
                )
            
            # Display results table
            st.markdown("### 📋 Detailed Results")
            st.dataframe(
                results_df.sort_values('probability', ascending=False),
                use_container_width=True
            )
            
            # Download button
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="batch_dti_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Display failed pairs if any
        if failed_pairs:
            st.markdown("---")
            st.warning(f"⚠️ **{len(failed_pairs)} pairs failed processing**")
            
            with st.expander("View Failed Pairs"):
                failed_df = pd.DataFrame(failed_pairs)
                st.dataframe(failed_df, use_container_width=True)
                
                # Download failed pairs
                failed_csv = failed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Failed Pairs (CSV)",
                    data=failed_csv,
                    file_name="failed_pairs.csv",
                    mime="text/csv"
                )

# Help section
st.markdown("---")
st.caption(
    "💡 **Tip**: For very large datasets (>1000 pairs), consider splitting into multiple batches. "
    "Processing time depends on API response times and may take several minutes."
)
