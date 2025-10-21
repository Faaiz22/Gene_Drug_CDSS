# Technical Documentation

This document provides a technical overview of the Drug-Gene Interaction prediction model and its architecture.

## 1. Model Architecture

The DTI prediction system is a multi-modal deep learning model that processes two distinct inputs: a 3D graph representation of a drug molecule and a learned embedding of a protein sequence.

### 1.1. Drug Encoder: E(n)-Equivariant Graph Neural Network (EGNN)

-   **Input**: A 3D molecular graph where nodes are atoms and edges are bonds. Each atom has initial features (e.g., atomic number, charge) and a 3D coordinate.
-   **Architecture**: We use a stack of EGNN layers. Unlike traditional GNNs, EGNNs are designed to be equivariant to 3D rotations, translations, and reflections. This means that if the input molecule is rotated in space, the resulting node representations will also rotate accordingly, leading to a more robust and physically-grounded representation.
-   **Process**: In each layer, nodes update their features by aggregating messages from their neighbors. The message calculation incorporates both node features and the relative distances between atoms in 3D space.
-   **Output**: After several layers, the node features are pooled (e.g., via global mean pooling) to produce a single, fixed-size vector representation for the entire drug molecule.

### 1.2. Protein Encoder: ESM-2 Transformer

-   **Input**: The amino acid sequence of the target protein.
-   **Model**: We use the `esm2_t12_35M_UR50D` model, a pre-trained protein language model from Meta AI. This model has been trained on millions of protein sequences and has learned rich representations of protein structure and function.
-   **Process**: The input sequence is tokenized and fed into the ESM-2 model. We use the output from the last hidden layer and average the embeddings across all amino acid positions to get a single, fixed-size vector for the protein.
-   **Fine-Tuning**: To adapt the model to our specific task while retaining its powerful pre-trained knowledge, we freeze the initial layers of the ESM-2 model and only fine-tune the top layers during training.

### 1.3. Interaction Module: Cross-Attention

-   **Purpose**: To effectively combine the drug and protein representations.
-   **Mechanism**: A standard multi-head cross-attention mechanism is used. The protein embedding acts as the `query`, while the drug embedding serves as both the `key` and `value`. This allows the model to learn which aspects of the drug are most relevant to the given protein, and vice-versa, creating a context-aware combined representation.

### 1.4. Prediction Head

-   **Input**: The concatenated drug embedding and the output of the cross-attention module.
-   **Architecture**: A multi-layer perceptron (MLP) with ReLU activations and dropout for regularization.
-   **Output**: A single logit, which is passed through a sigmoid function to produce the final interaction probability (a value between 0 and 1).

## 2. Data Preprocessing and Featurization

1.  **Data Enrichment**: Gene and chemical identifiers are used to fetch protein sequences from UniProt and chemical SMILES strings from PubChem via the PharmGKB API as a proxy.
2.  **3D Conformer Generation**: For each SMILES string, a low-energy 3D conformer is generated using the RDKit ETKDGv3 algorithm and optimized with the UFF force field.
3.  **Graph Construction**: The 3D conformer is converted into a graph format suitable for PyTorch Geometric, including node features, edge indices, edge features, and 3D coordinates (`pos`).

## 3. Training

-   **Loss Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`), suitable for binary classification.
-   **Optimizer**: AdamW, which is well-suited for training transformer-based models.
-   **Data Splitting**: We use a scaffold-based split. Molecules are grouped by their core chemical structure (scaffold), and all molecules sharing the same scaffold are placed in the same set (train, validation, or test). This prevents the model from simply memorizing common scaffolds and encourages it to learn more generalizable chemical principles.
-   **Regularization**: Dropout is used in the MLP head. Early stopping is employed to prevent overfitting, saving the model with the best validation AUC score.