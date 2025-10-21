import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_history(history, save_path):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training History - Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # AUC plot
    axes[1].plot(epochs, history['train_auc'], 'b-o', label='Train AUC', linewidth=2, markersize=6)
    axes[1].plot(epochs, history['val_auc'], 'r-s', label='Val AUC', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('AUC', fontsize=12)
    axes[1].set_title('Training History - AUC', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✓ Training history plot saved to: {save_path}")


def plot_evaluation_metrics(metrics, save_path):
    """Plot ROC, PR curve, and confusion matrix"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC Curve
    fpr, tpr, _ = metrics['roc_curve']
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curve
    precision, recall, _ = metrics['pr_curve']
    axes[1].plot(recall, precision, 'r-', linewidth=2, label=f"AP = {metrics['pr_auc']:.3f}")
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Confusion Matrix
    sns.heatmap(metrics['conf_matrix'], annot=True, fmt='d', cmap='Blues',
                ax=axes[2], cbar_kws={'label': 'Count'})
    axes[2].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Predicted Label', fontsize=12)
    axes[2].set_ylabel('True Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✓ Evaluation metrics plot saved to: {save_path}")
