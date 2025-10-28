# 🏥 Drug-Gene Interaction Clinical Decision Support System (CDSS)

> **⚠️ RESEARCH USE ONLY**: This system is designed for research and educational purposes. It should not be used as the sole basis for clinical decision-making without validation by healthcare professionals.

## 🎯 Overview

A state-of-the-art AI-powered Clinical Decision Support System that predicts drug-gene interactions using deep learning. The system combines:

- **3D Molecular Modeling**: E(n)-Equivariant Graph Neural Networks (EGNN)
- **Protein Language Models**: ESM-2 transformer architecture
- **Explainable AI**: Integrated Gradients for interpretability
- **Literature Integration**: Real-time PubMed search
- **Agentic AI**: Autonomous reasoning with LangChain

## 🌟 Key Features

### For Clinicians
- 🎯 **Single-Pair Prediction**: Quick interaction assessment with confidence metrics
- 📊 **Batch Analysis**: Process multiple drug-gene pairs simultaneously
- 💡 **Molecular Explanations**: Understand *why* interactions occur
- 📚 **Evidence-Based**: Integrated PubMed literature search
- 📋 **Clinical Reports**: Comprehensive, actionable summaries

### For Researchers
- 🧬 **3D Conformer Generation**: Accurate molecular geometry
- 🔬 **Multi-Modal Learning**: Combines structural and sequence data
- 📈 **Uncertainty Quantification**: Monte Carlo Dropout estimates
- 🔍 **Interpretability**: Attention weights and attribution maps
- 🤖 **Agentic Workflow**: Autonomous tool orchestration

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **CUDA-capable GPU** (recommended for training)
- **8GB+ RAM** (16GB+ recommended)

### Installation

#### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/Drug_Gene_CDSS.git
cd Drug_Gene_CDSS

# Create and activate environment
conda env create -f environment.yml
conda activate drug-gene-cdss

# Install PyTorch Geometric (adjust for your CUDA version)
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/Drug_Gene_CDSS.git
cd Drug_Gene_CDSS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Set up API keys** (required for agentic features):

```bash
# Create .streamlit/secrets.toml
mkdir -p .streamlit
cat > .streamlit/secrets.toml << EOF
GOOGLE_API_KEY = "your-gemini-api-key-here"
EOF
```

2. **Configure model paths** in `config/config.yaml`:

```yaml
paths:
  model_weights: "checkpoints/best_model.pt"  # Download from releases
  data_dir: "data/"
  cache_dir: "cache/"
```

### 🎮 Running the Application

```bash
streamlit run app/main.py
```

The application will be available at `https://genedrugcdss-guhcymfsyudq9mv2lxcgds.streamlit.app/`

## 📚 Usage Guide

### Single Prediction (Clinical Workflow)

1. Navigate to **"Single Prediction"** page
2. Enter drug identifier (e.g., "Imatinib", "CID 5330")
3. Enter gene identifier (e.g., "ABL1", "P00519")
4. Select analysis depth:
   - **Quick**: Fast probability estimate
   - **Standard**: Includes molecular explanation
   - **Full**: Complete analysis with literature
5. Review results:
   - Interaction probability
   - Confidence metrics
   - Molecular explanation
   - Supporting literature
   - Clinical recommendations

### Batch Analysis (Research Workflow)

1. Prepare CSV/TSV file with columns: `gene_id`, `chem_id`
2. Upload via **"Batch Analysis"** page
3. Download results with probabilities

### 3D Visualization

Explore molecular structures used in predictions:
- 3D conformer geometry
- Atom-level features
- Interactive rotation/zoom

### Model Explanation

Understand prediction rationale:
- Feature attributions (Integrated Gradients)
- Attention weight heatmaps
- Top contributing atoms/residues

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Layer                          │
│  Drug (SMILES) ──────────┬────────── Gene (Sequence)   │
└─────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
    ┌────▼────┐                         ┌───▼────┐
    │  EGNN   │                         │  ESM-2  │
    │ (3D GNN)│                         │(Protein)│
    └────┬────┘                         └───┬────┘
         │                                   │
         │         Cross-Attention          │
         └────────────►◄──────────────┘
                      │
                 ┌────▼────┐
                 │   MLP   │
                 │Predictor│
                 └────┬────┘
                      │
              ┌───────▼───────┐
              │  Probability  │
              │  [0.0 - 1.0]  │
              └───────────────┘
```

### Model Components

- **Drug Encoder**: 4-layer EGNN with equivariant message passing
- **Protein Encoder**: ESM-2 (35M parameters, fine-tuned top layers)
- **Interaction Module**: 8-head cross-attention
- **Prediction Head**: 3-layer MLP with dropout regularization


## Project Structure

```
Drug_Gene_CDSS/
├── app/                  # Streamlit web application
├── config/               # Configuration files (e.g., config.yaml)
├── data/                 # Raw data (user-provided relationships.tsv)
├── docs/                 # Project documentation
├── notebooks/            # Jupyter notebooks for experimentation
├── src/                  # Source code for the project
│   ├── models/           # Model definitions (EGNN, DTIModel)
│   ├── molecular_3d/     # 3D conformer generation and visualization
│   ├── preprocessing/    # Data loading, cleaning, and feature engineering
│   └── utils/            # Utility scripts (API clients, etc.)
├── tests/                # Unit and integration tests
├── .gitignore
├── environment.yml       # Conda environment file
├── LICENSE
├── README.md
└── requirements.txt      # Python package requirements
```

## Getting Started

### Prerequisites

- Conda or Miniconda
- Python 3.8+
- An NVIDIA GPU with CUDA support is highly recommended for training.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Drug_Gene_CDSS.git
    cd Drug_Gene_CDSS
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate drug-gene-cdss
    ```

3.  **Install PyTorch Geometric dependencies:**
    The `environment.yml` file handles most dependencies. However, PyTorch Geometric's CUDA-related packages sometimes require manual installation. Please refer to the official [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for instructions tailored to your system's CUDA version.

### Data Preparation

Place your data file in the `data/` directory. The file should be named `relationships.tsv` and contain tab-separated columns, including identifiers for genes and chemicals (e.g., PharmGKB IDs, UniProt IDs, PubChem CIDs).

## Usage

### Training the Model

The primary training pipeline is available as a Jupyter notebook:

1.  Launch Jupyter Lab:
    ```bash
    jupyter lab
    ```
2.  Open `notebooks/DTI_Training_Pipeline.ipynb` and run the cells sequentially.

The notebook will handle data enrichment, preprocessing, model training, and evaluation, saving checkpoints and results to the `checkpoints/` and `results/` directories, respectively.

### Running the Web Application

To launch the Streamlit-based CDSS interface:

```bash
streamlit run app/main.py
```

This will start a local web server where you can perform single predictions, batch analysis, and visualize results.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
