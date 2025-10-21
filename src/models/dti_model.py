import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)
from tqdm.auto import tqdm
from pathlib import Path

from .egnn import EGNNLayer
from ..preprocessing.feature_engineer import featurize_pair, collate_fn


class CrossAttention(nn.Module):
    """Cross-attention between drug and protein representations"""

    def __init__(self, drug_dim, prot_dim, attn_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)
        self.fc_drug = nn.Linear(drug_dim, attn_dim)
        self.fc_prot = nn.Linear(prot_dim, attn_dim)

    def forward(self, drug, protein):
        drug_proj = self.fc_drug(drug).unsqueeze(1)
        prot_proj = self.fc_prot(protein).unsqueeze(1)
        output, attn_weights = self.attn(prot_proj, drug_proj, drug_proj)
        return output.squeeze(1), attn_weights


class DTIModel(nn.Module):
    """Complete Drug-Target Interaction Prediction Model"""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # EGNN layers for drug encoding
        self.layers = nn.ModuleList([
            EGNNLayer(
                config['model']['drug_node_dim'] if i == 0 else config['model']['drug_hidden_dim'],
                config['model']['drug_edge_dim'],
                config['model']['drug_hidden_dim']
            )
            for i in range(config['model']['egnn_num_layers'])
        ])

        # Projection layer
        self.graph_projection = nn.Linear(config['model']['drug_hidden_dim'], config['model']['drug_output_dim'])

        # Cross-attention
        self.cross_attn = CrossAttention(
            config['model']['drug_output_dim'],
            config['model']['protein_embedding_dim'],
            config['model']['attention_dim'],
            config['model']['attention_heads']
        )

        # Prediction head
        mlp_dims = config['model']['mlp_hidden_dims']
        self.mlp = nn.Sequential(
            nn.Linear(config['model']['attention_dim'] * 2, mlp_dims[0]),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(mlp_dims[1], mlp_dims[2]),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(mlp_dims[2], 1)
        )

    def forward(self, graph, protein):
        x, edge_index, edge_attr, pos, batch = graph.x, graph.edge_index, graph.edge_attr, graph.pos, graph.batch

        # EGNN encoding
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, pos)

        # Pool to graph-level representation
        drug_emb = global_mean_pool(x, batch)
        drug_emb = self.graph_projection(drug_emb)

        # Cross-attention
        cross_output, attn_weights = self.cross_attn(drug_emb, protein)

        # Combine and predict
        combined = torch.cat([drug_emb, cross_output], dim=1)
        logits = self.mlp(combined)

        return logits.squeeze(-1), attn_weights


class Trainer:
    """Training engine with AMP, early stopping, and checkpointing"""

    def __init__(self, model, train_loader, val_loader, config: dict):
        self.model = model.to(config['training']['device'])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        self.scaler = torch.cuda.amp.GradScaler()
        self.criterion = nn.BCEWithLogitsLoss()

        self.best_val_auc = 0
        self.early_stop_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'train_auc': []
        }

    def step(self, batch, train=True):
        """Single training/validation step"""
        if train:
            self.model.train()
        else:
            self.model.eval()

        graph = batch["graph"].to(self.config['training']['device'])
        protein = batch["protein"].to(self.config['training']['device'])
        labels = batch["labels"].to(self.config['training']['device'])

        with torch.cuda.amp.autocast(enabled=True):
            logits, _ = self.model(graph, protein)
            loss = self.criterion(logits, labels)

        if train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return logits.detach().cpu(), labels.detach().cpu(), loss.item()

    def train_loop(self):
        """Complete training loop with early stopping"""
        print(f"\n🚀 Starting training for {self.config['training']['num_epochs']} epochs...")
        print(f"="*70)

        for epoch in range(self.config['training']['num_epochs']):
            # Training phase
            train_losses, train_preds, train_labels = [], [], []

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']} [Train]"):
                if batch is None: continue
                preds, labs, loss = self.step(batch, train=True)
                train_losses.append(loss)
                train_preds.extend(torch.sigmoid(preds).numpy())
                train_labels.extend(labs.numpy())

            # Validation phase
            val_losses, val_preds, val_labels = [], [], []

            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']} [Val]"):
                    if batch is None: continue
                    preds, labs, loss = self.step(batch, train=False)
                    val_losses.append(loss)
                    val_preds.extend(torch.sigmoid(preds).numpy())
                    val_labels.extend(labs.numpy())

            # Calculate metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_auc = roc_auc_score(train_labels, train_preds) if len(set(train_labels)) > 1 else 0
            val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)

            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{self.config['training']['num_epochs']} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f}")

            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.early_stop_counter = 0
                self.save_checkpoint(epoch, val_auc)
                print(f"  ✓ Best model saved! (AUC: {val_auc:.4f})")
            else:
                self.early_stop_counter += 1
                print(f"  No improvement ({self.early_stop_counter}/{self.config['training']['early_stopping_patience']})")
                if self.early_stop_counter >= self.config['training']['early_stopping_patience']:
                    print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                    break
            print(f"{'='*70}")

        print(f"\n✓ Training complete! Best validation AUC: {self.best_val_auc:.4f}")

    def save_checkpoint(self, epoch, val_auc):
        """Save model checkpoint"""
        ckpt_dir = Path(self.config['paths']['checkpoint_dir'])
        ckpt = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "val_auc": val_auc,
            "history": self.history,
            "config": self.config,
        }
        path = ckpt_dir / f"best_model_epoch{epoch+1}_auc{val_auc:.4f}.pt"
        torch.save(ckpt, path)


def evaluate_model(model, data_loader, config):
    """Evaluate model on a dataset"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if batch is None: continue
            graph = batch["graph"].to(config['training']['device'])
            protein = batch["protein"].to(config['training']['device'])
            labels = batch["labels"]

            logits, _ = model(graph, protein)
            preds = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {}
    if len(set(all_labels)) > 1:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_preds)
        metrics['pr_auc'] = average_precision_score(all_labels, all_preds)
        metrics['roc_curve'] = roc_curve(all_labels, all_preds)
        metrics['pr_curve'] = precision_recall_curve(all_labels, all_preds)

        pred_labels = (all_preds > 0.5).astype(int)
        metrics['accuracy'] = accuracy_score(all_labels, pred_labels)
        metrics['f1'] = f1_score(all_labels, pred_labels)
        metrics['conf_matrix'] = confusion_matrix(all_labels, pred_labels)

    return metrics, all_preds, all_labels


def predict_interaction(model, gene_id, chem_id, enricher, embedder, config):
    """Predict interaction score for a Gene-Chemical pair"""
    print(f"\n{'='*60}\nPREDICTING INTERACTION\n{'='*60}")
    print(f"Gene ID:     {gene_id}\nChemical ID: {chem_id}\n{'='*60}\n")

    sequence = enricher.fetch_sequence(gene_id)
    if sequence is None:
        print(f"❌ Could not retrieve sequence for gene: {gene_id}")
        return None
    print(f"✓ Sequence retrieved ({len(sequence)} amino acids)")

    smiles = enricher.fetch_smiles(chem_id)
    if smiles is None:
        print(f"❌ Could not retrieve SMILES for chemical: {chem_id}")
        return None
    print(f"✓ SMILES retrieved: {smiles}")

    features = featurize_pair(smiles, sequence, config, embedder)
    if features is None:
        print("❌ Failed to featurize the pair")
        return None
    print("✓ Features generated successfully")

    features["label"] = -1
    batch = collate_fn([features])
    if batch is None:
        print("❌ Failed to create batch")
        return None

    model.eval()
    with torch.no_grad():
        graph = batch["graph"].to(config['training']['device'])
        protein = batch["protein"].to(config['training']['device'])
        logits, attn_weights = model(graph, protein)
        prob = torch.sigmoid(logits).item()

    print("\n⏳ Calculating uncertainty (MC Dropout)...")
    mc_preds = []
    model.train()  # Enable dropout
    with torch.no_grad():
        for _ in range(config['processing']['mc_dropout_samples']):
            logits, _ = model(graph, protein)
            mc_preds.append(torch.sigmoid(logits).item())

    mc_mean = np.mean(mc_preds)
    mc_std = np.std(mc_preds)

    print(f"\n{'='*60}\nPREDICTION RESULTS\n{'='*60}")
    print(f"Interaction Probability: {prob:.4f} ({prob*100:.2f}%)")
    print(f"MC Dropout Mean:         {mc_mean:.4f}")
    print(f"MC Dropout Std:          {mc_std:.4f}")
    print(f"{'='*60}\n")

    return {
        "gene_id": gene_id,
        "chem_id": chem_id,
        "smiles": smiles,
        "sequence_length": len(sequence),
        "probability": prob,
        "mc_mean": mc_mean,
        "mc_std": mc_std,
        "attention_weights": attn_weights.cpu().numpy() if attn_weights is not None else None
    }
