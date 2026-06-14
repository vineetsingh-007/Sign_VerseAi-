import numpy as np

class LandmarkExtractor:
    """
    Extracts and normalizes hand landmarks for machine learning inference.
    Supports single-hand and two-hand configurations.
    Returns a unified 126-dimensional vector (2 hands * 21 keypoints * 3 coordinates).
    """
    def __init__(self):
        pass

    def extract_landmarks(self, results):
        """
        Extracts landmarks from MediaPipe Results object.
        Returns a normalized 1D numpy array of shape (126,).
        
        Indices 0 to 62: Left hand landmarks (or zeros if not detected).
        Indices 63 to 125: Right hand landmarks (or zeros if not detected).
        """
        left_hand_features = np.zeros(63)
        right_hand_features = np.zeros(63)

        if results.multi_hand_landmarks and results.multi_handedness:
            detected_hands = {}
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = handedness.classification[0].label # 'Left' or 'Right'
                
                # Extract coordinates
                raw_coords = []
                for lm in hand_landmarks.landmark:
                    raw_coords.append([lm.x, lm.y, lm.z])
                raw_coords = np.array(raw_coords) # shape (21, 3)
                
                # Normalize hand coordinates
                normalized_coords = self.normalize_hand(raw_coords)
                
                # Flatten
                flat_features = normalized_coords.flatten()
                detected_hands[hand_type] = flat_features

            if len(detected_hands) == 2:
                left_hand_features = detected_hands.get("Left", np.zeros(63))
                right_hand_features = detected_hands.get("Right", np.zeros(63))
            elif len(detected_hands) == 1:
                # Do NOT mirror. Keep the inactive hand slot as all zeros.
                if "Left" in detected_hands:
                    left_hand_features = detected_hands["Left"]
                    right_hand_features = np.zeros(63)
                else:
                    right_hand_features = detected_hands["Right"]
                    left_hand_features = np.zeros(63)

        # Concatenate left and right hand features to form a 126-dimensional vector
        combined_features = np.concatenate([left_hand_features, right_hand_features])
        return combined_features

    def normalize_hand(self, coords):
        """
        Normalizes a single hand's 21 landmarks (x, y, z coordinates).
        - Translates all landmarks relative to the wrist (landmark 0) so the wrist is at (0,0,0).
        - Scales all landmarks relative to the maximum distance from the wrist to any other landmark.
        """
        # 1. Translate relative to wrist (landmark 0)
        wrist = coords[0]
        translated_coords = coords - wrist

        # 2. Scale relative to maximum distance from wrist (to avoid division by zero, add small epsilon)
        distances = np.linalg.norm(translated_coords, axis=1)
        max_dist = np.max(distances)
        
        if max_dist > 1e-6:
            normalized_coords = translated_coords / max_dist
        else:
            normalized_coords = translated_coords

        return normalized_coords
