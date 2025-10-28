"""
LangChain Tools for Agentic DTI Prediction System.
Provides autonomous reasoning capabilities with tool-based interactions.
"""

from langchain.tools import tool
from typing import Dict, Any, Optional, List
import torch
import asyncio
from pathlib import Path

# Global state management for agent
agent_state = {}

"""
Pydantic models for agent tool input validation.
The tool functions themselves are defined in agent_orchestrator.py
to allow them to access the agent's internal state.
"""

from pydantic import BaseModel, Field

class DrugInput(BaseModel):
    drug_name: str = Field(description="The name, PubChem CID, or PharmGKB ID of the drug.")

class GeneInput(BaseModel):
    gene_name: str = Field(description="The gene symbol, UniProt ID, or PharmGKB ID of the gene.")

class LiteratureInput(BaseModel):
    query: str = Field(description="A specific question or topic to search for in the literature, e.g., 'What are the known interactions between Warfarin and CYP2C9?'")

@tool
def fetch_molecular_data_tool(
    drug_name: str,
    gene_name: str,
    drug_id: Optional[str] = None,
    gene_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetches SMILES string for a drug and amino acid sequence for a gene.
    Uses external APIs (PubChem, UniProt) with intelligent fallbacks.
    
    Args:
        drug_name: Common name or identifier for the drug
        gene_name: Gene symbol or protein name
        drug_id: Optional PubChem CID or PharmGKB ID
        gene_id: Optional UniProt ID or PharmGKB ID
    
    Returns:
        Dictionary with 'smiles' and 'sequence' keys, or 'error' on failure
    """
    from src.utils.api_clients import DataEnricher
    from src.core_processing import load_config
    
    try:
        config = load_config()
        enricher = DataEnricher(config)
        
        # Fetch data asynchronously
        loop = asyncio.get_event_loop()
        smiles = loop.run_until_complete(
            enricher.fetch_smiles(drug_id if drug_id else drug_name)
        )
        sequence = loop.run_until_complete(
            enricher.fetch_sequence(gene_id if gene_id else gene_name)
        )
        
        if not smiles:
            return {"error": f"Could not retrieve SMILES for drug: {drug_name}"}
        if not sequence:
            return {"error": f"Could not retrieve sequence for gene: {gene_name}"}
        
        # Store in global state for subsequent tools
        agent_state['drug_name'] = drug_name
        agent_state['gene_name'] = gene_name
        agent_state['smiles'] = smiles
        agent_state['sequence'] = sequence
        
        return {
            "smiles": smiles,
            "sequence": sequence,
            "smiles_length": len(smiles),
            "sequence_length": len(sequence)
        }
    
    except Exception as e:
        return {"error": f"Data fetching failed: {str(e)}"}


@tool
def featurize_drug_protein_pair_tool() -> Dict[str, Any]:
    """
    Converts SMILES and protein sequence into model-ready 3D graph and embeddings.
    Must be called after fetch_molecular_data_tool.
    
    Returns:
        Dictionary with featurization status and metadata
    """
    from src.molecular_3d.conformer_generator import smiles_to_3d_graph
    from src.core_processing import load_config, load_protein_featurizer, get_device
    from torch_geometric.data import Batch
    
    if 'smiles' not in agent_state or 'sequence' not in agent_state:
        return {"error": "No molecular data in state. Run fetch_molecular_data_tool first."}
    
    try:
        config = load_config()
        device = get_device()
        protein_featurizer = load_protein_featurizer(config, device)
        
        smiles = agent_state['smiles']
        sequence = agent_state['sequence']
        
        # Generate 3D drug graph
        graph = smiles_to_3d_graph(smiles, config)
        if graph is None:
            return {"error": "Failed to generate 3D conformer from SMILES"}
        
        # Generate protein embedding
        protein_emb = protein_featurizer.featurize(sequence)
        
        # Prepare batch
        graph_batch = Batch.from_data_list([graph]).to(device)
        protein_batch = protein_emb.unsqueeze(0).to(device)
        
        # Store in state
        agent_state['graph_batch'] = graph_batch
        agent_state['protein_batch'] = protein_batch
        agent_state['graph_metadata'] = {
            'num_atoms': graph.x.shape[0],
            'num_bonds': graph.edge_index.shape[1] // 2,
            'has_3d_coords': True
        }
        
        return {
            "status": "success",
            "num_atoms": graph.x.shape[0],
            "num_bonds": graph.edge_index.shape[1] // 2,
            "protein_embedding_dim": protein_emb.shape[0]
        }
    
    except Exception as e:
        return {"error": f"Featurization failed: {str(e)}"}


@tool
def predict_interaction_tool() -> Dict[str, Any]:
    """
    Runs the trained DTI model to predict interaction probability.
    Must be called after featurize_drug_protein_pair_tool.
    
    Returns:
        Dictionary with prediction probability and confidence metrics
    """
    from src.core_processing import load_dti_model, load_config, get_device
    import torch.nn.functional as F
    
    if 'graph_batch' not in agent_state or 'protein_batch' not in agent_state:
        return {"error": "No featurized data in state. Run featurize_drug_protein_pair_tool first."}
    
    try:
        config = load_config()
        device = get_device()
        model = load_dti_model(config, device)
        
        graph_batch = agent_state['graph_batch']
        protein_batch = agent_state['protein_batch']
        
        model.eval()
        with torch.no_grad():
            logits, attn_weights = model(graph_batch, protein_batch)
            prob = torch.sigmoid(logits).item()
        
        # Monte Carlo Dropout for uncertainty estimation
        mc_probs = []
        model.train()  # Enable dropout
        with torch.no_grad():
            for _ in range(config['processing']['mc_dropout_samples']):
                logits, _ = model(graph_batch, protein_batch)
                mc_probs.append(torch.sigmoid(logits).item())
        
        mc_mean = float(np.mean(mc_probs))
        mc_std = float(np.std(mc_probs))
        
        # Store results
        agent_state['prediction_prob'] = prob
        agent_state['prediction_uncertainty'] = mc_std
        agent_state['attention_weights'] = attn_weights
        
        # Interpret prediction
        if prob > 0.7:
            interpretation = "HIGH likelihood of interaction"
        elif prob > 0.5:
            interpretation = "MODERATE likelihood of interaction"
        else:
            interpretation = "LOW likelihood of interaction"
        
        return {
            "prediction_probability": prob,
            "interpretation": interpretation,
            "uncertainty_std": mc_std,
            "confidence": "High" if mc_std < 0.05 else "Moderate" if mc_std < 0.1 else "Low"
        }
    
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


@tool
def explain_prediction_tool(save_visualizations: bool = True) -> Dict[str, Any]:
    """
    Generates comprehensive explanation for the prediction using Integrated Gradients
    and identifies key molecular features driving the interaction.
    Must be called after predict_interaction_tool.
    
    Args:
        save_visualizations: Whether to save explanation plots
    
    Returns:
        Dictionary with natural language explanation and key features
    """
    from src.models.explainer import EnhancedExplainer
    from src.core_processing import load_dti_model, load_config, get_device
    
    if 'prediction_prob' not in agent_state:
        return {"error": "No prediction in state. Run predict_interaction_tool first."}
    
    try:
        config = load_config()
        device = get_device()
        model = load_dti_model(config, device)
        
        explainer = EnhancedExplainer(model, config, device)
        
        # Get data from state
        graph_batch = agent_state['graph_batch']
        protein_batch = agent_state['protein_batch']
        gene_name = agent_state['gene_name']
        drug_name = agent_state['drug_name']
        smiles = agent_state['smiles']
        
        # Generate explanation
        save_dir = Path(config['paths']['explanation_dir']) / f"{drug_name}_{gene_name}" if save_visualizations else None
        
        explanation = explainer.explain_prediction(
            graph=graph_batch,
            protein_emb=protein_batch.squeeze(0),
            gene_name=gene_name,
            drug_name=drug_name,
            smiles=smiles,
            save_dir=save_dir
        )
        
        # Store in state
        agent_state['explanation'] = explanation
        
        return {
            "status": "success",
            "natural_language_explanation": explanation['natural_language'],
            "top_contributing_atoms": explanation['top_atoms'][:3],
            "top_protein_residues": explanation['top_residues'][:5],
            "visualization_saved": save_visualizations,
            "visualization_paths": {k: str(v) for k, v in explanation['visualization_paths'].items()} if save_visualizations else {}
        }
    
    except Exception as e:
        return {"error": f"Explanation generation failed: {str(e)}"}


@tool
def search_literature_tool(max_papers: int = 5) -> Dict[str, Any]:
    """
    Searches PubMed for relevant research papers on the drug-gene interaction.
    Returns citations and abstracts of the most relevant papers.
    Must be called after fetch_molecular_data_tool.
    
    Args:
        max_papers: Maximum number of papers to retrieve (default: 5)
    
    Returns:
        Dictionary with list of papers and formatted citations
    """
    from src.utils.pubmed_client import PubMedClient
    from src.core_processing import load_config
    import asyncio
    
    if 'drug_name' not in agent_state or 'gene_name' not in agent_state:
        return {"error": "No drug/gene names in state. Run fetch_molecular_data_tool first."}
    
    try:
        config = load_config()
        pubmed_client = PubMedClient(config)
        
        drug_name = agent_state['drug_name']
        gene_name = agent_state['gene_name']
        
        # Determine interaction type from prediction if available
        interaction_type = None
        if 'prediction_prob' in agent_state:
            prob = agent_state['prediction_prob']
            if prob > 0.5:
                interaction_type = "binding"
            else:
                interaction_type = "no interaction"
        
        # Search PubMed
        loop = asyncio.get_event_loop()
        papers = loop.run_until_complete(
            pubmed_client.search_interactions(
                gene_name=gene_name,
                drug_name=drug_name,
                max_results=max_papers,
                interaction_type=interaction_type
            )
        )
        
        if not papers:
            return {
                "status": "no_results",
                "message": f"No papers found for {drug_name} and {gene_name} interaction"
            }
        
        # Format papers
        formatted_papers = []
        for paper in papers:
            formatted_papers.append({
                "pmid": paper['pmid'],
                "title": paper['title'],
                "authors": ", ".join(paper['authors']),
                "journal": paper['journal'],
                "year": paper['year'],
                "citation": pubmed_client.format_citation(paper),
                "abstract_preview": paper['abstract'][:300] + "..." if len(paper['abstract']) > 300 else paper['abstract'],
                "url": paper['url']
            })
        
        # Store in state
        agent_state['literature'] = formatted_papers
        
        return {
            "status": "success",
            "num_papers_found": len(formatted_papers),
            "papers": formatted_papers
        }
    
    except Exception as e:
        return {"error": f"Literature search failed: {str(e)}"}


@tool
def synthesize_clinical_report_tool() -> Dict[str, Any]:
    """
    Synthesizes all gathered information into a comprehensive clinical report.
    Should be called last, after all other analysis tools have been executed.
    
    Returns:
        Dictionary with formatted clinical report and recommendations
    """
    if 'prediction_prob' not in agent_state:
        return {"error": "No prediction available. Complete the analysis pipeline first."}
    
    try:
        # Gather all data from state
        drug_name = agent_state.get('drug_name', 'Unknown Drug')
        gene_name = agent_state.get('gene_name', 'Unknown Gene')
        prob = agent_state.get('prediction_prob', 0.0)
        uncertainty = agent_state.get('prediction_uncertainty', 0.0)
        explanation_text = agent_state.get('explanation', {}).get('natural_language', 'No explanation available')
        papers = agent_state.get('literature', [])
        
        # Build report
        report = f"""
╔═══════════════════════════════════════════════════════════════╗
║        DRUG-GENE INTERACTION CLINICAL REPORT                 ║
╚═══════════════════════════════════════════════════════════════╝

INTERACTION PAIR:
  Drug:    {drug_name}
  Gene:    {gene_name}

PREDICTION SUMMARY:
  Interaction Probability: {prob:.1%}
  Model Confidence: {"High" if uncertainty < 0.05 else "Moderate" if uncertainty < 0.1 else "Low"}
  Uncertainty (σ): {uncertainty:.4f}

INTERPRETATION:
  {"Strong evidence of interaction - clinical relevance likely" if prob > 0.7 else
   "Moderate evidence of interaction - further investigation recommended" if prob > 0.5 else
   "Limited evidence of interaction - unlikely to be clinically significant"}

MOLECULAR EXPLANATION:
  {explanation_text}

SUPPORTING LITERATURE:
"""
        
        if papers:
            report += f"  {len(papers)} relevant papers identified:\n\n"
            for i, paper in enumerate(papers[:3], 1):  # Top 3 papers
                report += f"  [{i}] {paper['citation']}\n"
                report += f"      PMID: {paper['pmid']}\n"
                if paper.get('abstract_preview'):
                    report += f"      {paper['abstract_preview']}\n"
                report += "\n"
        else:
            report += "  No supporting literature found in PubMed.\n"
        
        report += """
CLINICAL RECOMMENDATIONS:
"""
        
        if prob > 0.7:
            report += """  - HIGH priority for clinical consideration
  - Consider pharmacogenomic testing for relevant patient populations
  - Monitor for drug-gene interaction effects in treatment protocols
  - Consult literature for dosing adjustments or contraindications
"""
        elif prob > 0.5:
            report += """  - MODERATE priority for clinical awareness
  - May warrant consideration in complex cases or polypharmacy scenarios
  - Review patient-specific factors before clinical decision-making
  - Additional experimental validation recommended
"""
        else:
            report += """  - LOW clinical priority based on current evidence
  - Unlikely to require specific monitoring or intervention
  - May be safely disregarded in most clinical contexts
  - Consider alternative drug-gene interactions if clinically indicated
"""
        
        report += """
DISCLAIMER:
  This report is generated by an AI-based Clinical Decision Support System
  for research and informational purposes only. It should not replace
  professional medical judgment or established clinical guidelines.
  
═══════════════════════════════════════════════════════════════════
"""
        
        # Store report
        agent_state['clinical_report'] = report
        
        return {
            "status": "success",
            "report": report,
            "summary": {
                "drug": drug_name,
                "gene": gene_name,
                "probability": prob,
                "confidence": "High" if uncertainty < 0.05 else "Moderate" if uncertainty < 0.1 else "Low",
                "num_supporting_papers": len(papers)
            }
        }
    
    except Exception as e:
        return {"error": f"Report synthesis failed: {str(e)}"}


# Export all tools for agent
AGENTIC_TOOLS = [
    fetch_molecular_data_tool,
    featurize_drug_protein_pair_tool,
    predict_interaction_tool,
    explain_prediction_tool,
    search_literature_tool,
    synthesize_clinical_report_tool
]
