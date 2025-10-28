# %% [markdown]
# # DTI Model Training Pipeline
# 
# This script loads the training data, initializes the model components from the `src` directory, and runs the training loop. All model and data logic is imported from `src`, not redefined, ensuring reproducibility.

# %%
import sys
import os
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Add 'src' to the Python path ---
# This allows the script (or notebook) to find our custom modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))
# -------------------------------------

# --- Import all components from our 'src' library ---
try:
    from utils.config_loader import load_config
    from utils.logger import setup_logging
    from models.dti_model import (
        DTIDataset, 
        collate_fn, 
        DTIModel, 
        Trainer
    )
    from preprocessing.feature_engineer import ProteinEmbedder
    from molecular_3d.conformer_generator import smiles_to_3d_graph
except ImportError as e:
    print(f"Error: Failed to import modules from 'src'.")
    print(f"Make sure you are running this from the project's root directory or have 'src' in your PYTHONPATH.")
    print(f"Details: {e}")
    sys.exit(1)
# ----------------------------------------------------

# %% [markdown]
# ## 1. Setup and Configuration
# 
# Load config, set up logging, and define the device.

# %%
# Load configuration
try:
    # We use the path-aware config loader
    config = load_config("config/config.yaml") 
except Exception as e:
    print(f"Error loading configuration: {e}")
    sys.exit(1)

# Setup logging
setup_logging(log_path=config['paths'].get('log_file', 'logs/train.log'))
log = logging.getLogger(__name__)
log.info("--- Starting New Training Run ---")

# Set device
device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

# %% [markdown]
# ## 2. Load and Prepare Data
# 
# Load the dataset and create training/validation splits.

# %%
log.info("Loading dataset...")
data_path = config['paths']['dataset'] # Should be resolved by load_config
if not Path(data_path).exists():
    log.error(f"Dataset file not found at: {data_path}")
    sys.exit(1)

df = pd.read_csv(data_path, sep='\t')
df = df[['SMILES', 'sequence', 'label']].dropna().sample(frac=1, random_state=42)
log.info(f"Loaded {len(df)} total data points.")

# Split data
train_df, val_df = train_test_split(
    df, 
    test_size=config['training']['val_split'], 
    random_state=42, 
    stratify=df['label']
)
log.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

# %% [markdown]
# ## 3. Initialize DataLoaders
# 
# We now use the `DTIDataset` class imported from `src.models.dti_model`.

# %%
log.info("Initializing Protein Embedder (ESM-2)...")
try:
    # This heavy object is initialized once and passed to the datasets
    protein_embedder = ProteinEmbedder(config)
except Exception as e:
    log.error(f"Failed to initialize ProteinEmbedder: {e}")
    sys.exit(1)

log.info("Creating datasets and dataloaders...")

# Create dataset instances
train_dataset = DTIDataset(
    df=train_df,
    config=config,
    protein_embedder=protein_embedder,
    graph_gen_func=smiles_to_3d_graph # Pass the imported function
)

val_dataset = DTIDataset(
    df=val_df,
    config=config,
    protein_embedder=protein_embedder,
    graph_gen_func=smiles_to_3d_graph # Pass the imported function
)

# Create dataloader instances
train_loader = DataLoader(
    train_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    num_workers=config['training'].get('num_workers', 0),
    collate_fn=collate_fn, # Use the imported collate_fn
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=False,
    num_workers=config['training'].get('num_workers', 0),
    collate_fn=collate_fn, # Use the imported collate_fn
    pin_memory=True
)

log.info("DataLoaders created successfully.")

# %% [markdown]
# ## 4. Initialize Model
# 
# We now use the `DTIModel` class imported from `src.models.dti_model`.

# %%
log.info("Initializing DTIModel...")
try:
    model = DTIModel(config).to(device)
    log.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
except Exception as e:
    log.error(f"Failed to initialize DTIModel: {e}")
    sys.exit(1)

# %% [markdown]
# ## 5. Run Training
# 
# We now use the `Trainer` class imported from `src.models.dti_model`.

# %%
log.info("Initializing Trainer...")
try:
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
except Exception as e:
    log.error(f"Failed to initialize Trainer: {e}")
    sys.exit(1)

log.info("--- Starting training loop ---")
trainer.train()
log.info("--- Training complete ---")

# %% [markdown]
# ## 6. Plot Metrics and Save
# 
# Visualize the training and validation performance.

# %%
log.info("Plotting metrics...")
try:
    fig = trainer.plot_metrics()
    
    # Save plot
    plot_path = Path(config['paths']['output_dir']) / "training_metrics.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    log.info(f"Training metrics plot saved to {plot_path}")
    
    # Display the plot (this will work in a notebook)
    # In a script, you might just save it.
    try:
        from IPython.display import display
        display(fig)
    except ImportError:
        log.info("Cannot display plot in non-notebook environment. Plot saved.")

except Exception as e:
    log.error(f"Failed to plot metrics: {e}")

# %% [markdown]
# ---
# 
# End of pipeline.
# 
# ---
