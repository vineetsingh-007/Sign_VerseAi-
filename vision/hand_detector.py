import cv2
import mediapipe as mp
import mediapipe.solutions.hands as mp_hands
import mediapipe.solutions.drawing_utils as mp_drawing
import numpy as np
import time

class HandDetector:
    """
    Wraps MediaPipe Hands processing. Runs detection on image frames,
    calculates bounding boxes, and draws custom glowing futuristic HUD overlays.
    """
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.mp_hands = mp_hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp_drawing

    def find_hands(self, frame, draw=True, draw_skeleton=True, draw_joints=True, draw_bbox=True, draw_telemetry=True):
        """
        Processes a BGR frame and applies a professional, industry-friendly AI landmark overlay.
        Returns:
            results: MediaPipe hands detection results.
            annotated_frame: Frame with solid clean computer vision visualization.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        annotated_frame = frame.copy()
        
        if draw and results.multi_hand_landmarks:
            h, w, _ = frame.shape
            
            for idx, (hand_lms, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                hand_type = handedness.classification[0].label # 'Left' or 'Right'
                score = handedness.classification[0].score
                
                # Professional, clean BGR colors (Steel Blue for Right, Slate Teal for Left)
                primary_color = (220, 110, 30) if hand_type == "Right" else (100, 180, 40)
                
                # Project all landmarks to pixel coordinates
                pts = []
                for lm in hand_lms.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pts.append((cx, cy))
                
                # Calculate Bounding Box
                x_coords = [pt[0] for pt in pts]
                y_coords = [pt[1] for pt in pts]
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                
                # Add padding
                pad = 15
                xmin = max(0, xmin - pad)
                ymin = max(0, ymin - pad)
                xmax = min(w, xmax + pad)
                ymax = min(h, ymax + pad)
                
                # 1. Simple Rectangular Bounding Box
                if draw_bbox:
                    cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), primary_color, 1)
                
                # 2. Telemetry Text Label
                if draw_telemetry:
                    depth_avg = np.mean([lm.z for lm in hand_lms.landmark])
                    hud_label = f"{hand_type} Hand [Conf: {score*100:.0f}%] [Z: {depth_avg:.2f}]"
                    
                    # Compute size for clean background text box
                    (label_width, label_height), baseline = cv2.getTextSize(hud_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    text_y = max(ymin - 5, label_height + 5)
                    
                    cv2.rectangle(annotated_frame, (xmin, text_y - label_height - 3), (xmin + label_width + 4, text_y + 3), primary_color, -1)
                    cv2.putText(annotated_frame, hud_label, (xmin + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                # 3. Clean Connection Lines
                if draw_skeleton:
                    for start_idx, end_idx in self.mp_hands.HAND_CONNECTIONS:
                        pt1 = pts[start_idx]
                        pt2 = pts[end_idx]
                        cv2.line(annotated_frame, pt1, pt2, primary_color, 1, cv2.LINE_AA)
                
                # 4. Minimalist Joint Nodes
                if draw_joints:
                    for pt in pts:
                        cv2.circle(annotated_frame, pt, 4, primary_color, -1)
                        cv2.circle(annotated_frame, pt, 2, (255, 255, 255), -1)
            
        return results, annotated_frame

    def close(self):
        self.hands.close()
