import numpy as np
from training.dataset_loader import get_clean_base_posture, normalize_hand_coords

class LearningEngine:
    """
    Geometric comparison engine for sign language training.
    Compares real-time hand landmarks to gesture templates and provides
    detailed alignment scoring and real-time AI coaching feedback.
    """
    def __init__(self):
        pass

    def get_finger_extensions(self, coords):
        """
        Calculates extension ratios (0.0 to 1.0) for the 5 fingers of a hand (21, 3).
        Ratios: Thumb, Index, Middle, Ring, Pinky.
        """
        ratios = []
        
        # 1. Four fingers (Index, Middle, Ring, Pinky)
        for mcp, tip in [(5, 8), (9, 12), (13, 16), (17, 20)]:
            # Segment lengths
            segments = (
                np.linalg.norm(coords[mcp+1] - coords[mcp]) +
                np.linalg.norm(coords[mcp+2] - coords[mcp+1]) +
                np.linalg.norm(coords[tip] - coords[mcp+2])
            )
            # Tip to MCP distance
            dist_tip_mcp = np.linalg.norm(coords[tip] - coords[mcp])
            ratio = dist_tip_mcp / max(segments, 1e-6)
            ratios.append(ratio)
            
        # 2. Thumb: measure distance from tip (4) to Index MCP (5)
        # Relative to hand scale (Wrist 0 to Middle MCP 9)
        hand_scale = np.linalg.norm(coords[9] - coords[0])
        if hand_scale < 1e-6:
            hand_scale = 1.0
            
        dist_thumb_index = np.linalg.norm(coords[4] - coords[5])
        thumb_ratio = dist_thumb_index / hand_scale
        ratios.insert(0, thumb_ratio)
        
        return np.array(ratios) # [thumb, index, middle, ring, pinky]

    def evaluate_sign(self, user_features, target_label):
        """
        Compares user hand landmarks against target template landmarks.
        Args:
            user_features (126,): Combined left and right hand coordinates.
            target_label (str): The name of the target gesture to practice.
        Returns:
            score (float): Quality score (0 to 100).
            feedback (list of str): Real-time coaching tips.
            finger_alignment (dict): Current extension vs expected extension.
        """
        # 1. Determine which hand is active in user features
        left_user = user_features[0:63]
        right_user = user_features[63:126]
        
        is_left_active = np.any(left_user != 0)
        is_right_active = np.any(right_user != 0)
        
        if not is_left_active and not is_right_active:
            return 0.0, ["Show hand to the camera"], {}

        # 2. Get the active hand landmarks and expected template
        # For simplicity, we compare the user's primary active hand to the right-hand template
        if is_right_active:
            user_coords = right_user.reshape(21, 3)
            is_left_sign = False
        else:
            # If user shows left hand, we mirror it to right hand space for direct template comparison
            user_coords = left_user.reshape(21, 3).copy()
            user_coords[:, 0] *= -1.0 # mirror X
            is_left_sign = True

        # Load expected templates (always compare to right-hand template for consistency)
        target_coords = get_clean_base_posture(target_label, is_left=False)
        target_coords = normalize_hand_coords(target_coords)

        # 3. Calculate Euclidean distance similarity score
        dist = np.linalg.norm(user_coords - target_coords)
        # Scale: dist of 0.0 is perfect (100%), dist of 1.4+ is poor (0%)
        score = 100.0 * (1.0 - (dist / 1.4))
        score = float(np.clip(score, 0.0, 100.0))

        # 4. Finger extension comparisons
        user_ext = self.get_finger_extensions(user_coords)
        target_ext = self.get_finger_extensions(target_coords)
        
        finger_names = ["Thumb", "Index finger", "Middle finger", "Ring finger", "Pinky finger"]
        feedback = []
        finger_alignment = {}
        
        # Thresholds for extensions:
        # >0.70 is extended, <0.45 is folded, in between is partially folded
        for i, name in enumerate(finger_names):
            user_val = user_ext[i]
            target_val = target_ext[i]
            
            # Map values to status
            user_status = "Extended" if user_val > 0.70 else "Folded" if user_val < 0.45 else "Partially Folded"
            target_status = "Extended" if target_val > 0.70 else "Folded" if target_val < 0.45 else "Partially Folded"
            
            finger_alignment[name] = {
                "user": user_status,
                "expected": target_status,
                "score": float(np.clip(100 * (1.0 - abs(user_val - target_val)), 0, 100))
            }
            
            # Generate finger coaching tips
            if target_status == "Extended" and user_status == "Folded":
                feedback.append(f"Extend your {name} fully.")
            elif target_status == "Folded" and user_status == "Extended":
                feedback.append(f"Curl your {name} into your palm.")
            elif target_status == "Extended" and user_status == "Partially Folded":
                feedback.append(f"Straighten your {name} more.")
            elif target_status == "Folded" and user_status == "Partially Folded":
                feedback.append(f"Bend your {name} tighter.")

        # 5. Orientation (Hand Direction Vector) comparisons
        # Vector from wrist (0) to Middle finger MCP (9) represents hand heading
        user_dir = user_coords[9] - user_coords[0]
        user_dir /= max(np.linalg.norm(user_dir), 1e-6)
        
        target_dir = target_coords[9] - target_coords[0]
        target_dir /= max(np.linalg.norm(target_dir), 1e-6)
        
        # Calculate dot product (cosine similarity of heading)
        heading_similarity = np.dot(user_dir, target_dir)
        
        # Generate orientation coaching tips
        if heading_similarity < 0.85:
            # Check horizontal tilt (X axis)
            # note: m = -1 if left hand, which we mirrored, so it's aligned
            if user_dir[0] < target_dir[0] - 0.15:
                feedback.append("Tilt your hand more to the right.")
            elif user_dir[0] > target_dir[0] + 0.15:
                feedback.append("Tilt your hand more to the left.")
                
            # Check vertical tilt (Y axis)
            # note: MediaPipe Y increases downwards, so negative Y is upwards
            if user_dir[1] < target_dir[1] - 0.15:
                feedback.append("Angle your hand slightly downwards.")
            elif user_dir[1] > target_dir[1] + 0.15:
                feedback.append("Angle your hand slightly upwards.")

        # 6. Distance / visibility checks
        # Hand scale (pixel-wise distance wrist to knuckles)
        # If max coordinate distance is extremely small, hand is too far
        hand_span = np.max(np.linalg.norm(user_coords, axis=1))
        if hand_span < 0.2:
            feedback.append("Move your hand closer to the camera.")
        elif hand_span > 0.95:
            feedback.append("Move your hand further back from the camera.")

        if not feedback:
            feedback.append("Excellent form! Hold steady.")
            
        return score, feedback, finger_alignment
