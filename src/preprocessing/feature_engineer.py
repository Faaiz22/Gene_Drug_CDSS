import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from transformers import AutoTokenizer, EsmModel
from tqdm.auto import tqdm

from ..molecular_3d.conformer_generator import smiles_to_3d_graph


def generate_scaffold(smiles, include_chirality=False):
    """Generate Bemis-Murcko scaffold from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        scaffold = MurckoScaffold.MurckoDecompose(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality) if scaffold else None
    except: return None


def scaffold_split(df, train_size=0.70, test_size=0.20, val_size=0.10, random_state=42):
    """Split dataset based on molecular scaffolds"""
    print("\n🔬 Generating molecular scaffolds for data splitting...")
    np.random.seed(random_state)
    df['scaffold'] = df['smiles'].apply(generate_scaffold)
    df = df.dropna(subset=['scaffold'])

    scaffolds = df['scaffold'].unique()
    np.random.shuffle(scaffolds)

    train_idx = int(len(scaffolds) * train_size)
    test_idx = int(len(scaffolds) * (train_size + test_size))

    train_scaffolds = scaffolds[:train_idx]
    test_scaffolds = scaffolds[train_idx:test_idx]
    val_scaffolds = scaffolds[test_idx:]

    df_train = df[df['scaffold'].isin(train_scaffolds)].reset_index(drop=True)
    df_test = df[df['scaffold'].isin(test_scaffolds)].reset_index(drop=True)
    df_val = df[df['scaffold'].isin(val_scaffolds)].reset_index(drop=True)

    print(f"\n📊 Data Split Results:")
    print(f"  • Train: {len(df_train)} samples ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  • Test:  {len(df_test)} samples ({len(df_test)/len(df)*100:.1f}%)")
    print(f"  • Val:   {len(df_val)} samples ({len(df_val)/len(df)*100:.1f}%)")

    return df_train, df_test, df_val


class ProteinEmbedder:
    """Protein sequence embedder using ESM model"""
    def __init__(self, config: dict):
        self.config = config
        self.device = config['training']['device']
        model_name = config['model']['protein_model_name']

        print(f"\n⏳ Loading ESM model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(self.device)
        print(f"✓ Model loaded on {self.device}")

        freeze_layers = config['model']['protein_freeze_layers']
        print(f"\n🔒 Freezing embedding layer and first {freeze_layers} encoder layers...")
        for param in self.model.embeddings.parameters(): param.requires_grad = False
        for i in range(freeze_layers):
            for param in self.model.encoder.layer[i].parameters(): param.requires_grad = False
        self.model.eval()

    def embed(self, sequence: str) -> torch.Tensor:
        """Generate embedding for protein sequence"""
        tokens = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config['processing']['max_protein_length']
        )
        with torch.no_grad():
            outputs = self.model(tokens.input_ids.to(self.device))
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)


def featurize_pair(smiles: str, sequence: str, config: dict, embedder: ProteinEmbedder) -> Optional[Dict[str, Any]]:
    """Combine drug graph and protein embedding"""
    graph = smiles_to_3d_graph(smiles, config)
    if graph is None: return None
    protein_emb = embedder.embed(sequence)
    return {"graph": graph, "protein_emb": protein_emb}


class DTIDataset(Dataset):
    """PyTorch Dataset for Drug-Target Interactions"""
    def __init__(self, dataframe: pd.DataFrame, config: dict, embedder: ProteinEmbedder):
        self.config = config
        self.embedder = embedder
        self.data = []

        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Featurizing pairs"):
            features = featurize_pair(row["smiles"], row["sequence"], config, embedder)
            if features is not None:
                features["label"] = row.get("label", 1)
                features["gene_id"] = row.get("gene_id", "")
                features["chem_id"] = row.get("chem_id", "")
                self.data.append(features)

        print(f"✓ Successfully featurized {len(self.data)}/{len(dataframe)} pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Custom collate function for batching"""
    batch = [item for item in batch if item is not None]
    if not batch: return None

    graphs = [item["graph"] for item in batch]
    proteins = torch.stack([item["protein_emb"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float)

    return {
        "graph": Batch.from_data_list(graphs),
        "protein": proteins,
        "labels": labels
    }
