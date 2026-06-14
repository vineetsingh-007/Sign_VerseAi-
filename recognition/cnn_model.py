import torch
import torch.nn as nn
import json
import os

class SignLanguageCNN(nn.Module):
    """
    MLP architecture for classifying sign language gestures
    from a 126-dimensional hand landmark coordinate vector.
    
    Input: (Batch, 126) or (Batch, 1, 126)
    Output: Class logits of shape (Batch, NumClasses)
    """
    def __init__(self, num_classes):
        super(SignLanguageCNN, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(126, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Flatten input to (batch_size, 126) if it has a channel dimension
        if len(x.shape) == 3:
            x = x.squeeze(1)
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        return self.fc(x)

def save_model_metadata(class_labels, save_dir, filename="model_metadata.json"):
    """Saves model class labels and config to JSON for deployment loading."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, filename)
    metadata = {
        "num_classes": len(class_labels),
        "class_labels": class_labels,
        "input_dim": 126
    }
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=4)
    return filepath

def load_model_metadata(metadata_path):
    """Loads class labels and config."""
    with open(metadata_path, "r") as f:
        return json.load(f)
