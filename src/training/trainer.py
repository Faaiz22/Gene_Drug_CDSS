import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
from tqdm import tqdm

class Trainer:
    """
    Training orchestrator.
    """
    
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device(config['training']['device'])
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            drug_graphs = batch['drug_graph'].to(self.device)
            protein_seqs = batch['protein_seq'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            logits = self.model(drug_graphs, protein_seqs).squeeze()
            loss = self.criterion(logits, labels.float())
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on validation set.
        """
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                drug_graphs = batch['drug_graph'].to(self.device)
                protein_seqs = batch['protein_seq'].to(self.device)
                labels = batch['label']
                
                logits = self.model(drug_graphs, protein_seqs).squeeze()
                probs = torch.sigmoid(logits).cpu()
                
                all_preds.append(probs)
                all_labels.append(labels)
        
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        
        return {
            'roc_auc': roc_auc_score(labels, preds),
            'pr_auc': average_precision_score(labels, preds)
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """
        Full training loop.
        """
        best_auc = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"Val PR-AUC: {metrics['pr_auc']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(metrics['roc_auc'])
            
            # Save best model
            if metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                torch.save(
                    self.model.state_dict(),
                    self.config['paths']['model_checkpoint']
                )
                print(f"✓ Best model saved (AUC: {best_auc:.4f})")
            
            # Log to wandb
            wandb.log({
                'train_loss': train_loss,
                **{f'val_{k}': v for k, v in metrics.items()}
            })
