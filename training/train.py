import torch
import torch.nn as nn
import torch.optim as optim
import sys

# Suppress SummaryWriter on Python 3.13+ due to macOS protobuf C++ silent exits
if sys.version_info >= (3, 13):
    class SummaryWriter:
        def __init__(self, log_dir=None):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass
else:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        class SummaryWriter:
            def __init__(self, log_dir=None):
                pass
            def add_scalar(self, *args, **kwargs):
                pass
            def close(self):
                pass
import os
import argparse
import json
from datetime import datetime

from recognition.cnn_model import SignLanguageCNN, save_model_metadata
from training.dataset_loader import create_data_loaders, CLASS_LABELS

def train_model(epochs=30, batch_size=64, lr=0.001, dataset_path="datasets/synthetic_dataset.npz", 
                models_dir="models", logs_dir="logs", patience=7):
    """
    Trains the SignLanguageCNN model and logs metrics to TensorBoard and JSON.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for training: {device}")
    
    # Load Dataloaders
    print("Loading datasets...")
    train_loader, val_loader, _ = create_data_loaders(
        dataset_path=dataset_path, 
        batch_size=batch_size, 
        force_regenerate=True
    )
    
    num_classes = len(CLASS_LABELS)
    print(f"Number of classes: {num_classes}")
    
    # Initialize Model, Loss, Optimizer, and Scheduler
    model = SignLanguageCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Tensorboard writer
    tb_dir = os.path.join(logs_dir, "runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=tb_dir)
    
    # Track statistics
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "best_epoch": 0, "best_val_loss": float('inf')
    }
    
    # Early stopping config
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(models_dir, "best_sign_model.pth")
    last_model_path = os.path.join(models_dir, "last_sign_model.pth")
    
    # Save model metadata first
    metadata_path = save_model_metadata(CLASS_LABELS, models_dir)
    print(f"Saved model metadata to: {metadata_path}")
    
    print("Starting training loop...")
    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # Step LR Scheduler
        scheduler.step(epoch_val_loss)
        
        # Log to TensorBoard
        writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
        writer.add_scalar("Loss/Val", epoch_val_loss, epoch)
        writer.add_scalar("Accuracy/Train", epoch_train_acc, epoch)
        writer.add_scalar("Accuracy/Val", epoch_val_acc, epoch)
        
        # Log to local history dict
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")
        
        # Save Last Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc
        }, last_model_path)
        
        # Save Best Checkpoint (Early Stopping check)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            history["best_val_loss"] = best_val_loss
            history["best_epoch"] = epoch
            epochs_no_improve = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc
            }, best_model_path)
            print(f"  --> Saved new best model checkpont (Val Loss: {epoch_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! Training stopped at epoch {epoch}")
                break
                
    writer.close()
    
    # Save training history to JSON for dashboard visualization
    history_filepath = os.path.join(logs_dir, "training_history.json")
    with open(history_filepath, "w") as f:
        json.dump(history, f, indent=4)
        
    print("Training process finished.")
    print(f"Best Epoch: {history['best_epoch']} | Best Val Loss: {history['best_val_loss']:.4f}")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignVerse AI Model Training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--dataset_path", type=str, default="datasets/synthetic_dataset.npz", help="Dataset NPZ file path")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Directory for Tensorboard and history logs")
    
    args = parser.parse_args()
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dataset_path=args.dataset_path,
        models_dir=args.models_dir,
        logs_dir=args.logs_dir,
        patience=args.patience
    )
