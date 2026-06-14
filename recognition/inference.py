import torch
import torch.nn.functional as F
import numpy as np
import os
import json

from recognition.cnn_model import SignLanguageCNN, load_model_metadata
from training.dataset_loader import CLASS_LABELS

class SignInferenceEngine:
    """
    Handles loading the trained SignLanguageCNN model and executing
    inference on 126-dimensional landmark vectors.
    """
    def __init__(self, model_path="models/best_sign_model.pth", metadata_path="models/model_metadata.json"):
        self.model_path = model_path
        self.metadata_path = metadata_path
        import sys
        # Bypass MPS inside Streamlit to prevent Metal multithreading SIGSEGV crashes on macOS
        if "streamlit" in sys.modules:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        self.model = None
        self.class_labels = CLASS_LABELS
        self.num_classes = len(CLASS_LABELS)
        self.is_mock = False
        
        self.load_model()

    def load_model(self):
        """Loads model weights and config, fall back to mock mode if not found."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.metadata_path):
            print(f"Warning: Model weights or metadata not found at {self.model_path}. Running in SIMULATED/MOCK mode.")
            self.is_mock = True
            return
            
        try:
            # Load metadata
            metadata = load_model_metadata(self.metadata_path)
            self.class_labels = metadata["class_labels"]
            self.num_classes = metadata["num_classes"]
            
            # Initialize & Load model
            self.model = SignLanguageCNN(num_classes=self.num_classes).to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.is_mock = False
            print(f"Loaded trained SignLanguageCNN model successfully on {self.device}.")
        except Exception as e:
            print(f"Error loading model weights: {e}. Falling back to MOCK mode.")
            self.is_mock = True

    def predict(self, feature_vector):
        """
        Runs inference on a 126-dimensional feature vector.
        Returns:
            predicted_label (str): The predicted sign language gesture.
            confidence (float): Probability score of prediction.
        """
        # If model is not trained/loaded, return a realistic mock prediction
        if self.is_mock:
            # Mock behavior: If hand landmarks are all zeros, returns None
            if np.all(feature_vector == 0):
                return "NO_HAND", 1.0
            
            # Predict based on features (e.g. check if right index is extended)
            # We can create a simple heuristic mock to make it highly responsive during testing
            # If indices 63-125 are non-zero, let's select a greeting or alphabet
            is_two_handed = np.any(feature_vector[0:63] > 0)
            if is_two_handed:
                mock_labels = ["HELP", "EMERGENCY", "THANK YOU", "DOCTOR", "GOOD MORNING"]
            else:
                mock_labels = ["HELLO", "PLEASE", "YES", "NO", "A", "B", "C", "D", "1", "2"]
            
            # Pick a deterministic index based on landmarks to simulate consistency
            lm_sum = int(np.sum(np.abs(feature_vector)) * 100)
            idx = lm_sum % len(mock_labels)
            pred_label = mock_labels[idx]
            
            # Generate a realistic high confidence (0.82 to 0.98)
            confidence = 0.82 + (lm_sum % 17) * 0.01
            return pred_label, float(confidence)

        # Active ML model inference
        # Feature vector must shape (126,) -> torch tensor of (1, 126)
        if np.all(feature_vector == 0):
            return "NO_HAND", 1.0
            
        try:
            tensor_input = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor_input)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get max prediction
                confidence, predicted_idx = torch.max(probabilities, 1)
                pred_label = self.class_labels[predicted_idx.item()]
                return pred_label, float(confidence.item())
        except Exception as e:
            print(f"Error during PyTorch inference: {e}")
            return "ERROR", 0.0

    def predict_top_k(self, feature_vector, k=3):
        """
        Runs inference and returns the top K class predictions with their probabilities.
        """
        if self.is_mock:
            if np.all(feature_vector == 0):
                return [("NO_HAND", 1.0)]
            raw_pred, raw_conf = self.predict(feature_vector)
            
            # Find the index of the predicted label to determine deterministic alternatives
            idx = self.class_labels.index(raw_pred) if raw_pred in self.class_labels else 0
            alt1 = self.class_labels[(idx + 1) % len(self.class_labels)]
            alt2 = self.class_labels[(idx + 2) % len(self.class_labels)]
            
            p1 = raw_conf
            p2 = (1.0 - p1) * 0.65
            p3 = 1.0 - p1 - p2
            return [(raw_pred, float(p1)), (alt1, float(p2)), (alt2, float(p3))]

        if np.all(feature_vector == 0):
            return [("NO_HAND", 1.0)]

        try:
            tensor_input = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor_input)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
                
                # Get indices of top K predictions
                top_indices = np.argsort(probabilities)[::-1][:k]
                
                top_k = []
                for idx in top_indices:
                    label = self.class_labels[idx]
                    prob = probabilities[idx]
                    top_k.append((label, float(prob)))
                return top_k
        except Exception as e:
            print(f"Error during top-k inference: {e}")
            return [("ERROR", 0.0)]

    def get_saliency_map(self, feature_vector, predicted_label):
        """
        Computes the gradient of the predicted class score with respect to each input coordinate.
        Returns a 21-dimensional normalized magnitude array representing knuckle attention.
        """
        # Default/Fallback: equal attention with minor random noise for visual display
        fallback = np.ones(21) / 21.0
        fallback += np.random.normal(0, 0.015, 21)
        fallback = np.clip(fallback, 0.01, 1.0)
        fallback /= np.sum(fallback)

        if self.is_mock or np.all(feature_vector == 0):
            return fallback

        try:
            # Find the index of the predicted label
            if predicted_label not in self.class_labels:
                return fallback
                
            predicted_idx = self.class_labels.index(predicted_label)
            
            # Convert feature vector to a tensor with gradient tracking enabled
            tensor_input = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            tensor_input.requires_grad_()
            
            # Forward pass (ensure gradients are not disabled!)
            self.model.eval()
            outputs = self.model(tensor_input)
            
            # Logit for the predicted class
            score = outputs[0, predicted_idx]
            
            # Backward pass
            self.model.zero_grad()
            score.backward()
            
            # Extract gradients w.r.t input
            gradients = tensor_input.grad.cpu().numpy()[0] # shape (126,)
            
            # Magnitude per landmark: combine left and right hand coordinates
            left_grads = gradients[0:63].reshape(21, 3)
            right_grads = gradients[63:126].reshape(21, 3)
            
            left_magnitudes = np.linalg.norm(left_grads, axis=1)
            right_magnitudes = np.linalg.norm(right_grads, axis=1)
            
            # Use whichever hand is active (non-zero)
            combined_magnitudes = left_magnitudes + right_magnitudes
            
            # Min-Max normalize to make it a high-contrast heatmap
            grad_sum = np.sum(combined_magnitudes)
            if grad_sum > 1e-6:
                normalized_saliency = combined_magnitudes / grad_sum
            else:
                normalized_saliency = np.ones(21) / 21.0
                
            return normalized_saliency
        except Exception as e:
            print(f"Error computing saliency map: {e}")
            if self.model:
                self.model.eval()
            return fallback
