# API Reference

This document outlines the core classes and functions available in the `src` directory.

## `src.utils.api_clients`

### `class DataEnricher`

Handles fetching data from external APIs like PharmGKB, UniProt, and PubChem.

-   `__init__(self, config: dict)`: Initializes the enricher with the application configuration.
-   `fetch_smiles(self, chem_id: str) -> Optional[str]`: Fetches the canonical SMILES string for a given chemical ID. Caches results.
-   `fetch_sequence(self, gene_id: str) -> Optional[str]`: Fetches the amino acid sequence for a given gene ID. Caches results.

---

## `src.preprocessing.data_enricher`

### `process_raw_data(config: dict, enricher: DataEnricher) -> pd.DataFrame`

Orchestrates the entire data loading and cleaning pipeline.

---

## `src.preprocessing.feature_engineer`

### `scaffold_split(df, train_size, test_size, val_size, ...)`

Splits a DataFrame based on the Bemis-Murcko scaffolds of the molecules.

### `class ProteinEmbedder`

Wrapper for the ESM-2 protein language model.

-   `__init__(self, config: dict)`: Loads the pre-trained ESM-2 model and tokenizer.
-   `embed(self, sequence: str) -> torch.Tensor`: Generates a fixed-size embedding for a protein sequence.

### `class DTIDataset(Dataset)`

PyTorch Dataset that handles on-the-fly featurization of drug-gene pairs.

---

## `src.molecular_3d.conformer_generator`

### `smiles_to_3d_graph(smiles: str, config: dict) -> Optional[Data]`

Converts a SMILES string into a PyTorch Geometric `Data` object, including 3D coordinates.

---

## `src.models.dti_model`

### `class DTIModel(nn.Module)`

The main neural network model.

-   `__init__(self, config: dict)`: Constructs the model layers based on the configuration.
-   `forward(self, graph, protein) -> Tuple[torch.Tensor, torch.Tensor]`: Defines the forward pass, taking a batch of graphs and protein embeddings and returning logits and attention weights.

### `class Trainer`

Handles the model training and validation loop.

-   `__init__(self, model, train_loader, val_loader, config)`: Initializes the trainer, optimizer, and loss function.
-   `train_loop(self)`: Runs the main training loop, including validation, early stopping, and checkpointing.

### `evaluate_model(model, data_loader, config) -> dict`

Evaluates the model on a given data loader and returns a dictionary of performance metrics.

### `predict_interaction(model, gene_id, chem_id, ...)`

High-level function to predict the interaction for a single new pair, including data fetching and featurization.

---

## `src.models.egnn`

### `class EGNNLayer(MessagePassing)`

An E(n)-Equivariant Graph Neural Network layer used for encoding the 3D drug structure.