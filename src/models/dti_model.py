import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from models.egnn import EGNN
from utils.exceptions import FeaturizationException, ModelException

log = logging.getLogger(__name__)

# --- 1. Dataset Class ---

class DTIDataset(Dataset):
    """PyTorch Dataset for DTI prediction"""
    
    def __init__(self, df, config, protein_embedder, graph_gen_func):
        self.df = df
        self.config = config
        self.protein_embedder = protein_embedder
        self.graph_gen_func = graph_gen_func

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['SMILES']
        sequence = row['sequence']
        label = torch.tensor(row['label'], dtype=torch.float)

        try:
            # Generate 3D graph
            drug_graph = self.graph_gen_func(smiles, self.config)
            if drug_graph is None:
                raise FeaturizationException("Graph generation failed")
            
            # Generate protein embedding
            protein_embedding = self.protein_embedder.embed(sequence)
            
            return drug_graph, protein_embedding, label
        
        except (FeaturizationException, Exception) as e:
            log.warning(f"Skipping data point {idx} due to error: {e}")
            # Return None to be filtered by collate_fn
            return None

# --- 2. Collate Function ---

def collate_fn(batch):
    """Custom collate function to filter Nones and batch PyG graphs"""
    # Filter out None values from failed featurizations
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None, None, None

    drug_graphs, protein_embeddings, labels = zip(*batch)
    
    # Batch graphs
    drug_batch = Batch.from_data_list(drug_graphs)
    
    # Stack embeddings and labels
    protein_batch = torch.stack(protein_embeddings)
    label_batch = torch.stack(labels)
    
    return drug_batch, protein_batch, label_batch

# --- 3. DTI Model Class ---

class DTIModel(nn.Module):
    """
    Drug-Target Interaction model combining EGNN for drugs
    and ESM embeddings for proteins.
    """
    def __init__(self, config):
        super(DTIModel, self).__init__()
        self.config = config['model']
        
        # Drug Branch (EGNN)
        self.egnn = EGNN(
            in_node_nf=self.config['drug_node_dim'],
            hidden_nf=self.config['egnn_hidden_dim'],
            out_node_nf=self.config['egnn_hidden_dim'], # Output to be pooled
            in_edge_nf=self.config['drug_edge_dim'],
            n_layers=self.config['egnn_layers'],
            attention=True,
            normalize=True,
            tanh=True
        )
        
        # Protein Branch (MLP)
        self.protein_mlp = nn.Sequential(
            nn.Linear(self.config['protein_embedding_dim'], self.config['protein_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(self.config['dropout'])
        )
        
        # Cross-Attention
        # We treat the pooled drug embedding as query, and protein tokens as keys/values
        # Note: ESM2 embeddings are pooled, so we have 1 token.
        self.protein_dim = self.config['protein_hidden_dim']
        self.drug_dim = self.config['egnn_hidden_dim']
        
        # For simplicity, we'll use a simple attention mechanism
        # or just concatenate after pooling.
        
        # --- Using simple concatenation ---
        self.combined_mlp = nn.Sequential(
            nn.Linear(self.drug_dim + self.protein_dim, self.config['combined_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(self.config['combined_hidden_dim'], 1)
        )
        
    def forward(self, drug_graph, protein_embedding):
        # Process drug
        # drug_graph is a PyG Batch object
        # h: [n_nodes_in_batch, hidden_dim], x: [n_nodes_in_batch, 3]
        h, _ = self.egnn(drug_graph.x, drug_graph.pos, drug_graph.edge_index, drug_graph.edge_attr)
        
        # Global Average Pooling for drug graph
        # We need to use the 'batch' attribute to pool
        from torch_geometric.nn import global_mean_pool
        drug_features = global_mean_pool(h, drug_graph.batch) # [batch_size, drug_dim]

        # Process protein
        protein_features = self.protein_mlp(protein_embedding) # [batch_size, protein_dim]
        
        # Combine
        combined_features = torch.cat((drug_features, protein_features), dim=1)
        
        # Final prediction
        logits = self.combined_mlp(combined_features).squeeze(1) # [batch_size]
        
        # We don't return attention weights for this simple model
        # If you implement cross-attention, you would return them here.
        attention_weights = torch.zeros(drug_features.size(0), 1, 1) # Placeholder
        
        return logits, attention_weights

# --- 4. Trainer Class ---

class Trainer:
    """Handles the model training and validation loop"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config['training']
        
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['learning_rate'], 
            weight_decay=self.config['weight_decay']
        )
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.epochs = self.config['epochs']
        self.save_path = config['paths']['model_weights']
        self.best_val_auroc = 0.0
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auroc': [],
            'val_prauc': []
        }
        log.info(f"Trainer initialized. Model moved to {self.device}.")
        log.info(f"Model weights will be saved to {self.save_path}")

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for drug_batch, protein_batch, label_batch in tqdm(self.train_loader, desc="Training", leave=False):
            if drug_batch is None: continue # Skip empty batches
            
            drug_batch = drug_batch.to(self.device)
            protein_batch = protein_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, _ = self.model(drug_batch, protein_batch)
            loss = self.criterion(logits, label_batch)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for drug_batch, protein_batch, label_batch in tqdm(self.val_loader, desc="Validation", leave=False):
                if drug_batch is None: continue
                
                drug_batch = drug_batch.to(self.device)
                protein_batch = protein_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                logits, _ = self.model(drug_batch, protein_batch)
                loss = self.criterion(logits, label_batch)
                
                total_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(label_batch.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        
        try:
            auroc = roc_auc_score(all_labels, all_preds)
            prauc = average_precision_score(all_labels, all_preds)
        except ValueError as e:
            log.warning(f"Could not calculate AUROC/PRAUC (likely only one class present): {e}")
            auroc = 0.5
            prauc = 0.5
            
        return {
            "loss": avg_loss,
            "roc_auc": auroc,
            "pr_auc": prauc
        }

    def train(self):
        log.info(f"Starting training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_metrics = self._validate_epoch()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auroc'].append(val_metrics['roc_auc'])
            self.history['val_prauc'].append(val_metrics['pr_auc'])
            
            log.info(f"Epoch {epoch+1}/{self.epochs} | "
                     f"Train Loss: {train_loss:.4f} | "
                     f"Val Loss: {val_metrics['loss']:.4f} | "
                     f"Val AUROC: {val_metrics['roc_auc']:.4f} | "
                     f"Val PRAUC: {val_metrics['pr_auc']:.4f}")
            
            if val_metrics['roc_auc'] > self.best_val_auroc:
                self.best_val_auroc = val_metrics['roc_auc']
                log.info(f"New best model found! AUROC: {self.best_val_auroc:.4f}")
                
                # --- START OF CORRECTION ---
                # This is the logic that was missing
                try:
                    # Ensure the save directory exists
                    save_dir = Path(self.save_path).parent
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save the model state dictionary
                    # This is the recommended way to save models
                    torch.save(self.model.state_dict(), self.save_path)
                    log.info(f"Best model saved to {self.save_path}")
                except Exception as e:
                    log.error(f"Failed to save model: {e}")
                # --- END OF CORRECTION ---

        log.info("Training complete.")
        log.info(f"Best validation AUROC: {self.best_val_auroc:.4f}")

    def plot_metrics(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_ylabel('BCE Logits Loss')
        ax1.set_title('Model Loss Over Epochs')
        ax1.legend()
        ax1.grid(True)
        
        # AUROC plot
        ax2.plot(self.history['val_auroc'], label='Validation AUROC', color='green')
        ax2.plot(self.history['val_prauc'], label='Validation PR-AUC', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Metric')
        ax2.set_title('Validation Metrics Over Epochs')
        ax2.legend()
        ax2.grid(True)
        
        fig.tight_layout()
        return fig
