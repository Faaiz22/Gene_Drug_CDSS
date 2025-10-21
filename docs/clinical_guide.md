# Clinical Guide for the Drug-Gene Interaction CDSS

**Disclaimer:** This tool is intended for research and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.

## 1. Introduction

This Clinical Decision Support System (CDSS) is designed to assist clinicians and researchers in exploring potential interactions between drugs and genes. By leveraging a sophisticated deep learning model, it predicts the likelihood that a given chemical compound will interact with a specific protein target.

This can be useful in:
-   **Pharmacogenomics**: Understanding how a patient's genetic makeup might influence their response to a drug.
-   **Drug Repurposing**: Identifying new potential uses for existing drugs.
-   **Hypothesis Generation**: Generating novel research hypotheses for further experimental validation.

## 2. How to Use the System

The system provides several interfaces, accessible via the sidebar:

### Single Prediction

Use this page to quickly check the interaction potential between one drug and one gene.

1.  **Enter Identifiers**: Provide a recognized identifier for the gene (e.g., Gene Symbol like `CYP2C9`, PharmGKB ID like `PA4450`, or UniProt ID like `P11712`) and the drug (e.g., common name like `celecoxib`, PubChem CID, or PharmGKB ID like `PA44836`).
2.  **Run Prediction**: Click the "Predict Interaction" button.
3.  **Interpret Results**: The system will return:
    *   **Interaction Probability**: A score from 0% to 100% indicating the model's predicted likelihood of an interaction.
    *   **Confidence/Uncertainty**: An estimate of the model's confidence in its prediction. High uncertainty may warrant caution.
    *   **Interpretation**: A plain-language summary of the result.

### Batch Analysis

Use this page to analyze a list of drug-gene pairs simultaneously.

1.  **Prepare Your File**: Create a CSV or TSV file with two columns: `gene_id` and `chem_id`.
2.  **Upload and Run**: Upload the file and start the analysis.
3.  **Download Results**: Once complete, you can download a new file containing the original pairs along with their predicted interaction probabilities.

## 3. Understanding the Output

-   **High Probability (> 70%)**: Suggests a strong potential for interaction. This may indicate that the drug is a substrate, inhibitor, or inducer of the gene product. This finding could be a starting point for further literature review or experimental validation.
-   **Moderate Probability (50-70%)**: Indicates a possible but less certain interaction. The relationship may be weak or context-dependent.
-   **Low Probability (< 50%)**: Suggests that an interaction is unlikely according to the model.

Always consider the **uncertainty score**. A high-probability prediction with high uncertainty is less reliable than one with low uncertainty.

## 4. Limitations

-   The model's predictions are based on patterns learned from existing data and do not constitute experimental proof.
-   The model may not perform well on novel drug scaffolds or protein families not well-represented in its training data.
-   The prediction does not specify the *type* of interaction (e.g., inhibition vs. activation) or its clinical significance.

All significant findings should be validated through established experimental methods and clinical studies.