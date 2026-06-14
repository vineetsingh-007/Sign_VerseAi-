import torch
import numpy as np
import os
import json
import argparse
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from recognition.cnn_model import SignLanguageCNN, load_model_metadata
from training.dataset_loader import create_data_loaders, CLASS_LABELS

def evaluate_model(model_path="models/best_sign_model.pth", metadata_path="models/model_metadata.json", 
                   dataset_path="datasets/synthetic_dataset.npz", reports_dir="reports"):
    """
    Evaluates the model on the test dataset and outputs a metrics report and confusion matrix plot.
    """
    os.makedirs(reports_dir, exist_ok=True)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")
    
    # Load metadata
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}. Please train the model first.")
        return
    
    metadata = load_model_metadata(metadata_path)
    class_labels = metadata["class_labels"]
    num_classes = metadata["num_classes"]
    
    # Load Model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
        return
        
    model = SignLanguageCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model successfully loaded.")
    
    # Load Test Loader
    _, _, test_loader = create_data_loaders(dataset_path=dataset_path, force_regenerate=False)
    
    # Collect all predictions and true labels
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate overall metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    
    # Detailed per-class precision, recall, f1
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=range(num_classes)
    )
    
    class_metrics = {}
    for idx, label in enumerate(class_labels):
        class_metrics[label] = {
            "precision": float(precision_per_class[idx]),
            "recall": float(recall_per_class[idx]),
            "f1": float(f1_per_class[idx])
        }
        
    # Generate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    
    # Prepare reports JSON
    metrics_report = {
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall": {
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1)
        },
        "per_class": class_metrics
    }
    
    # Save metrics JSON
    metrics_path = os.path.join(reports_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=4)
    print(f"Saved evaluation metrics report to: {metrics_path}")
    
    # Save Confusion Matrix as Numpy
    cm_path = os.path.join(reports_dir, "confusion_matrix.npy")
    np.save(cm_path, cm)
    
    # Plot Confusion Matrix using Matplotlib
    plt.figure(figsize=(16, 14))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('SignVerse AI - Sign Language Recognition Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45, ha='right')
    plt.yticks(tick_marks, class_labels)
    
    # Add numbers to cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel('True Gesture')
    plt.xlabel('Predicted Gesture')
    plt.tight_layout()
    
    fig_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {fig_path}")
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS SUMMARY")
    print("="*40)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("="*40)
    
    return metrics_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignVerse AI Model Evaluation")
    parser.add_argument("--model_path", type=str, default="models/best_sign_model.pth", help="Trained PyTorch model weight path")
    parser.add_argument("--metadata_path", type=str, default="models/model_metadata.json", help="Model metadata JSON path")
    parser.add_argument("--dataset_path", type=str, default="datasets/synthetic_dataset.npz", help="Dataset NPZ file path")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Directory to save evaluation reports")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        dataset_path=args.dataset_path,
        reports_dir=args.reports_dir
    )
