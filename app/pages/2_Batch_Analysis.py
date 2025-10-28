import streamlit as st
import pandas as pd
from src.utils.validators import validate_batch_dataframe
from src.utils.exceptions import CDSSException

# Set up the Streamlit page
st.set_page_config(page_title="Batch DTI Analysis", layout="wide")
st.title("🧪 Batch DTI Analysis")
st.markdown("Upload a CSV/Excel file with `gene_id` and `chem_id` columns to run predictions in bulk.")

# Check if core_processor is initialized in session state
if "core_processor" not in st.session_state:
    st.error("Core processor not initialized. Please return to the main page.")
    st.stop()

core_processor = st.session_state.core_processor

# File uploader
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Preview the uploaded data
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Validation
    validation_errors = validate_batch_dataframe(df)
    if validation_errors:
        st.error("Errors found in uploaded file:")
        for err in validation_errors:
            st.warning(f"- {err}")
        st.stop()
    else:
        st.success("File format is valid. Ready for analysis.")

    # Run analysis
    if st.button(f"🚀 Run Analysis on {len(df)} Pairs"):
        # Execute your analysis logic here
        try:
            # Assuming there's a method in core_processor to handle the analysis
            results = core_processor.run_analysis(df)
            st.success("Analysis completed successfully!")
            st.dataframe(results)  # Display results
        except Exception as e:
            st.error(f"An error occurred during the batch run: {e}")
