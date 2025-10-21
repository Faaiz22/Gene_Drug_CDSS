# Drug-Gene Interaction Clinical Decision Support System (CDSS)

This repository contains the source code for a deep learning model designed to predict interactions between drugs (chemicals) and genes (proteins). The system serves as a Clinical Decision Support System (CDSS) to help researchers and clinicians identify potential drug-target interactions (DTIs), aiding in drug discovery and personalized medicine.

The model leverages a state-of-the-art E(n)-Equivariant Graph Neural Network (EGNN) to process 3D molecular structures of drugs and a pre-trained ESM-2 transformer model for protein sequence embeddings. A cross-attention mechanism integrates these representations to predict the likelihood of an interaction.

## Key Features

- **3D Molecular Representation**: Uses 3D conformers of molecules, capturing spatial information crucial for molecular interactions.
- **Equivariant GNN**: Employs an EGNN that respects the rotational and translational symmetries of 3D space.
- **Protein Language Model**: Utilizes ESM-2, a powerful transformer model pre-trained on millions of protein sequences, to generate rich embeddings.
- **Cross-Attention Mechanism**: Intelligently combines drug and protein information to focus on relevant interaction features.
- **End-to-End Pipeline**: Includes scripts for data fetching, preprocessing, model training, evaluation, and prediction.
- **Streamlit Interface**: A user-friendly web application for single-pair prediction, batch analysis, and model interpretation (under development).

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
