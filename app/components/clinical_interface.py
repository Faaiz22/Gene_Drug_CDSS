# This file is a placeholder for components related to the clinical user interface.
# For example, functions to format predictions into clinically relevant reports,
# or to integrate with external clinical knowledge bases.

def format_prediction_report(result: dict) -> str:
    """Formats a prediction result into a human-readable report."""
    report = f"**Prediction Report for {result['gene_id']} and {result['chem_id']}**\n\n"
    report += f"- **Interaction Probability**: {result['probability']:.2%}\n"
    report += f"- **Model Uncertainty (Std. Dev.)**: {result['mc_std']:.4f}\n\n"
    report += "**Interpretation:**\n"
    if result['probability'] > 0.7:
        report += "The model predicts a **HIGH** likelihood of interaction. "
    elif result['probability'] > 0.5:
        report += "The model predicts a **MODERATE** likelihood of interaction. "
    else:
        report += "The model predicts a **LOW** likelihood of interaction. "

    if result['mc_std'] > 0.1:
        report += "However, the model exhibits high uncertainty, suggesting the prediction may be less reliable."
    else:
        report += "The model is confident in this prediction (low uncertainty)."

    return report
