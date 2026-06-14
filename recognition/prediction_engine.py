from collections import deque, Counter
import numpy as np

class PredictionEngine:
    """
    Applies statistical smoothing and state tracking over raw real-time model predictions.
    Filters jitter, implements debouncing, tracks hand-absence states, and detects
    emergency keywords instantly.
    """
    def __init__(self, window_size=8, confidence_threshold=0.55, mode_ratio_threshold=0.55):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.mode_ratio_threshold = mode_ratio_threshold
        
        self.prediction_window = deque(maxlen=window_size)
        self.confidence_window = deque(maxlen=window_size)
        
        self.last_stable_prediction = None
        self.no_hand_consecutive_frames = 0
        self.no_hand_limit = 12 # frames before resetting stable prediction lock
        
        # Emergency keywords
        self.emergency_keywords = {"HELP", "EMERGENCY", "PAIN", "DOCTOR", "WATER", "MEDICINE", "AMBULANCE"}

    def process_frame(self, pred_label, confidence, hand_detected=True):
        """
        Processes a single frame prediction.
        Args:
            pred_label (str): The predicted class name.
            confidence (float): The probability score from softmax.
            hand_detected (bool): Whether hand landmarks were actively detected in the frame.
        Returns:
            stable_prediction (str or None): The smoothed output symbol/word.
            confidence (float): The average confidence of the stable prediction.
            is_new_prediction (bool): True if this represents a change in the stable output.
            is_emergency (bool): True if the stabilized prediction is an emergency command.
        """
        if not hand_detected:
            self.no_hand_consecutive_frames += 1
            # If hands are away for long enough, clear windows and reset stable lock
            if self.no_hand_consecutive_frames >= self.no_hand_limit:
                self.prediction_window.clear()
                self.confidence_window.clear()
                self.last_stable_prediction = None
            return None, 0.0, False, False

        # Hands are detected, reset counter
        self.no_hand_consecutive_frames = 0

        # Discard low confidence raw predictions immediately
        if confidence < self.confidence_threshold:
            return None, 0.0, False, False

        self.prediction_window.append(pred_label)
        self.confidence_window.append(confidence)

        # Wait until we have enough frames to make a decision
        if len(self.prediction_window) < self.window_size // 2:
            return None, 0.0, False, False

        # Perform mode voting over the window
        counter = Counter(self.prediction_window)
        most_common_label, count = counter.most_common(1)[0]
        mode_ratio = count / len(self.prediction_window)

        # Check if the most common prediction meets the ratio threshold
        if mode_ratio >= self.mode_ratio_threshold:
            # Calculate average confidence for this label in the window
            matching_confidences = [
                c for l, c in zip(self.prediction_window, self.confidence_window) if l == most_common_label
            ]
            avg_confidence = np.mean(matching_confidences) if matching_confidences else 0.0
            
            # Check if this is a new prediction compared to the last stable one
            is_new = (most_common_label != self.last_stable_prediction)
            
            if is_new:
                self.last_stable_prediction = most_common_label
                is_emergency = most_common_label in self.emergency_keywords
                return most_common_label, avg_confidence, True, is_emergency
            else:
                return most_common_label, avg_confidence, False, False

        return None, 0.0, False, False

    def clear(self):
        """Clears all sliding windows and states."""
        self.prediction_window.clear()
        self.confidence_window.clear()
        self.last_stable_prediction = None
        self.no_hand_consecutive_frames = 0
