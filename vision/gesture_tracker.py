from collections import deque
import numpy as np

class GestureTracker:
    """
    Maintains a temporal buffer of recent hand landmark feature vectors
    to track hand movements, velocity, and trajectory. This allows
    distinguishing between static and dynamic (moving) gestures.
    """
    def __init__(self, buffer_size=30, motion_threshold=0.015):
        self.buffer_size = buffer_size
        self.motion_threshold = motion_threshold
        
        # Deque of 126-dimensional landmark vectors
        self.landmark_history = deque(maxlen=buffer_size)
        
        # Track wrist position over time (Left wrist: lm 0 -> index 0..2, Right wrist: lm 0 -> index 63..65)
        self.left_wrist_history = deque(maxlen=buffer_size)
        self.right_wrist_history = deque(maxlen=buffer_size)

    def update(self, combined_landmarks):
        """
        Updates the tracker with the latest 126-dimensional combined landmark vector.
        """
        self.landmark_history.append(combined_landmarks)
        
        # Extract left wrist (indices 0, 1, 2)
        left_wrist = combined_landmarks[0:3]
        # Check if left hand is active (if active, landmarks are non-zero)
        if np.any(left_wrist):
            self.left_wrist_history.append(left_wrist)
        else:
            self.left_wrist_history.append(None)
            
        # Extract right wrist (indices 63, 64, 65)
        right_wrist = combined_landmarks[63:66]
        if np.any(right_wrist):
            self.right_wrist_history.append(right_wrist)
        else:
            self.right_wrist_history.append(None)

    def is_moving(self, hand="Right"):
        """
        Checks if the specified hand is currently moving based on the average
        velocity of the wrist landmark over the history buffer.
        """
        history = self.right_wrist_history if hand == "Right" else self.left_wrist_history
        valid_points = [p for p in history if p is not None]
        
        if len(valid_points) < 5:
            return False
            
        # Calculate consecutive velocities
        velocities = []
        for i in range(1, len(valid_points)):
            vel = np.linalg.norm(np.array(valid_points[i]) - np.array(valid_points[i-1]))
            velocities.append(vel)
            
        mean_velocity = np.mean(velocities) if velocities else 0.0
        return mean_velocity > self.motion_threshold, mean_velocity

    def get_motion_direction(self, hand="Right"):
        """
        Determines the general direction of hand movement (e.g., 'Left', 'Right', 'Up', 'Down', 'Static').
        """
        history = self.right_wrist_history if hand == "Right" else self.left_wrist_history
        valid_points = [p for p in history if p is not None]
        
        if len(valid_points) < 10:
            return "Static"
            
        moving, vel = self.is_moving(hand)
        if not moving:
            return "Static"
            
        # Compare start and end coordinates
        start = np.array(valid_points[0])
        end = np.array(valid_points[-1])
        diff = end - start # [dx, dy, dz]
        
        # In screen space, y is inverted (0 at top, 1 at bottom)
        dx, dy, dz = diff[0], -diff[1], diff[2] # invert dy so upward motion increases y
        
        if abs(dx) > abs(dy):
            return "Right" if dx > 0 else "Left"
        else:
            return "Up" if dy > 0 else "Down"

    def get_recent_sequence(self):
        """
        Returns the history buffer as a numpy array of shape (L, 126),
        padded with zeros if the buffer is not yet full.
        """
        history_len = len(self.landmark_history)
        if history_len == 0:
            return np.zeros((self.buffer_size, 126))
            
        seq = list(self.landmark_history)
        # Pad at start if history is short
        while len(seq) < self.buffer_size:
            seq.insert(0, np.zeros(126))
            
        return np.array(seq)

    def clear(self):
        self.landmark_history.clear()
        self.left_wrist_history.clear()
        self.right_wrist_history.clear()
