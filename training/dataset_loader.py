import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import json

# Define the complete class labels supported by SIGNVERSE AI
CLASS_LABELS = (
    # Alphabets A-Z
    [chr(i) for i in range(ord('A'), ord('Z') + 1)] +
    # Digits 0-9
    [str(i) for i in range(10)] +
    # Common Words & Greetings
    ["HELLO", "THANK YOU", "PLEASE", "YES", "NO", "GOOD MORNING"] +
    # Emergency Commands
    ["HELP", "DOCTOR", "EMERGENCY", "WATER", "PAIN", "MEDICINE", "AMBULANCE"]
)

class SignLanguageDataset(Dataset):
    """
    A custom PyTorch Dataset for Sign Language landmarks.
    Accepts features of shape (N, 126) and labels of shape (N,).
    """
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def rotate_coords(coords, pitch=0, roll=0, yaw=0):
    """
    Rotates a 3D coordinate array of shape (N, 3) by pitch (X), roll (Y), and yaw (Z) in radians.
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(roll), 0, np.sin(roll)],
                   [0, 1, 0],
                   [-np.sin(roll), 0, np.cos(roll)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return coords @ R.T

def normalize_hand_coords(coords):
    """
    Normalizes a single hand's 21 landmarks (x, y, z coordinates).
    Translates all landmarks relative to the wrist (landmark 0) so the wrist is at (0,0,0).
    Scales all landmarks relative to the maximum distance from the wrist to any other landmark.
    """
    # 1. Translate relative to wrist (landmark 0)
    wrist = coords[0]
    translated = coords - wrist
    
    # 2. Scale relative to maximum distance from wrist
    distances = np.linalg.norm(translated, axis=1)
    max_dist = np.max(distances)
    
    if max_dist > 1e-6:
        normalized_coords = translated / max_dist
    else:
        normalized_coords = translated
        
    return normalized_coords

def get_clean_base_posture(class_label, is_left=False):
    """
    Generates anatomically correct, highly distinctive landmark coordinates (21, 3) 
    for each of the 49 classes of sign language. Wrist is at (0,0,0).
    Uses finger extension states, unique spreads, and rotations to ensure mathematical distinguishability.
    """
    m = -1.0 if is_left else 1.0
    coords = np.zeros((21, 3))
    
    # 1. Initialize knuckles (MCP joints) at realistic relative offsets
    coords[0] = [0.0, 0.0, 0.0]                  # Wrist
    coords[1] = [0.15 * m, 0.06, -0.05]          # Thumb MCP
    coords[5] = [0.10 * m, 0.25, -0.02]          # Index MCP
    coords[9] = [0.00 * m, 0.26, -0.02]          # Middle MCP
    coords[13] = [-0.08 * m, 0.24, -0.02]        # Ring MCP
    coords[17] = [-0.16 * m, 0.20, -0.02]        # Pinky MCP
    
    # Finger segment length ratios
    L_thumb = [0.15, 0.14, 0.11]
    L_index = [0.24, 0.19, 0.14]
    L_middle = [0.26, 0.21, 0.15]
    L_ring = [0.24, 0.19, 0.14]
    L_pinky = [0.19, 0.14, 0.11]
    
    # Configure finger spreads depending on the class groups
    # Fingers tight together (e.g. B, U, M, N) vs spread (5, W, HELLO)
    if class_label in ['B', 'U', 'M', 'N', 'A', 'E', 'S', 'T', 'PLEASE']:
        # Tight parallel fingers
        d_index = np.array([0.02 * m, 0.32, 0.00])
        d_middle = np.array([0.00 * m, 0.34, 0.00])
        d_ring = np.array([-0.02 * m, 0.32, 0.00])
        d_pinky = np.array([-0.04 * m, 0.29, 0.00])
    elif class_label in ['5', 'HELLO', 'EMERGENCY', 'AMBULANCE', 'THANK YOU', 'GOOD MORNING']:
        # Wide spread fingers
        d_index = np.array([0.08 * m, 0.32, 0.00])
        d_middle = np.array([0.00 * m, 0.34, 0.00])
        d_ring = np.array([-0.08 * m, 0.32, 0.00])
        d_pinky = np.array([-0.16 * m, 0.29, 0.00])
    else:
        # Standard default spread
        d_index = np.array([0.05 * m, 0.32, 0.00])
        d_middle = np.array([0.00 * m, 0.34, 0.00])
        d_ring = np.array([-0.05 * m, 0.32, 0.00])
        d_pinky = np.array([-0.10 * m, 0.29, 0.00])
        
    d_thumb = np.array([0.25 * m, 0.15, -0.05])
    
    def set_finger_coords(mcp, tip, direction, lengths, state):
        v = direction / np.linalg.norm(direction)
        if state == 1.0: # Extended straight
            coords[mcp+1] = coords[mcp] + lengths[0] * v
            coords[mcp+2] = coords[mcp+1] + lengths[1] * v
            coords[tip] = coords[mcp+2] + lengths[2] * v
        elif state == 0.0: # Folded tight
            v1 = v + np.array([0.0, 0.0, -0.4])
            v1 /= np.linalg.norm(v1)
            coords[mcp+1] = coords[mcp] + lengths[0] * v1 * 0.75
            v2 = np.array([0.0, -0.45, -0.7])
            coords[mcp+2] = coords[mcp+1] + lengths[1] * v2
            v3 = np.array([0.0, -0.75, -0.3])
            coords[tip] = coords[mcp+2] + lengths[2] * v3
        else: # 0.5 (Partially bent/curved)
            v1 = v + np.array([0.0, 0.0, -0.2])
            v1 /= np.linalg.norm(v1)
            coords[mcp+1] = coords[mcp] + lengths[0] * v1
            v2 = v + np.array([0.0, -0.2, -0.45])
            v2 /= np.linalg.norm(v2)
            coords[mcp+2] = coords[mcp+1] + lengths[1] * v2
            v3 = v + np.array([0.0, -0.45, -0.15])
            v3 /= np.linalg.norm(v3)
            coords[tip] = coords[mcp+2] + lengths[2] * v3

    # Define base finger extensions: [thumb, index, middle, ring, pinky]
    states = [1.0, 1.0, 1.0, 1.0, 1.0] # default open hand
    
    # Map class labels to finger extension states
    if class_label in ['A', 'E', 'M', 'N', 'S', 'T', 'YES', 'PAIN']:
        states = [0.0, 0.0, 0.0, 0.0, 0.0]  # Fist
    elif class_label in ['B', 'HELLO', 'PLEASE', 'THANK YOU', 'GOOD MORNING', 'EMERGENCY', 'AMBULANCE']:
        states = [0.0, 1.0, 1.0, 1.0, 1.0]  # Thumb folded, others extended
    elif class_label in ['C', 'O', '0']:
        states = [0.5, 0.5, 0.5, 0.5, 0.5]  # Curved
    elif class_label in ['D', '1', 'Z', 'HELP', 'DOCTOR', 'MEDICINE']:
        states = [0.0, 1.0, 0.0, 0.0, 0.0]  # Index extended
    elif class_label in ['F', '9']:
        states = [0.0, 0.0, 1.0, 1.0, 1.0]  # Index folded/touching, others extended
    elif class_label in ['G', 'L', 'Q']:
        states = [1.0, 1.0, 0.0, 0.0, 0.0]  # Thumb + Index extended
    elif class_label in ['H', 'U', 'V', 'K', '2', 'NO']:
        states = [0.0, 1.0, 1.0, 0.0, 0.0]  # Index + Middle extended
    elif class_label in ['I', 'J']:
        states = [0.0, 0.0, 0.0, 0.0, 1.0]  # Pinky extended
    elif class_label in ['W', 'WATER', '6']:
        states = [0.0, 1.0, 1.0, 1.0, 0.0]  # Index, Middle, Ring extended
    elif class_label in ['7']:
        states = [0.0, 1.0, 1.0, 0.0, 1.0]  # Thumb + Ring touch/folded
    elif class_label in ['8']:
        states = [0.0, 1.0, 0.0, 1.0, 1.0]  # Thumb + Middle touch/folded
    elif class_label in ['3']:
        states = [1.0, 1.0, 1.0, 0.0, 0.0]  # Thumb, Index, Middle extended
    elif class_label in ['4']:
        states = [0.0, 1.0, 1.0, 1.0, 1.0]  # Four fingers extended
    elif class_label in ['5']:
        states = [1.0, 1.0, 1.0, 1.0, 1.0]  # Five extended
    elif class_label in ['X']:
        states = [0.0, 0.5, 0.0, 0.0, 0.0]  # Index hooked (0.5), others folded
    elif class_label in ['Y']:
        states = [1.0, 0.0, 0.0, 0.0, 1.0]  # Thumb + Pinky extended
    elif class_label in ['R']:
        states = [0.0, 1.0, 1.0, 0.0, 0.0]  # Crossed fingers (Index + Middle extended)
        
    # Apply baseline coordinates
    set_finger_coords(1, 4, d_thumb, L_thumb, states[0])
    set_finger_coords(5, 8, d_index, L_index, states[1])
    set_finger_coords(9, 12, d_middle, L_middle, states[2])
    set_finger_coords(13, 16, d_ring, L_ring, states[3])
    set_finger_coords(17, 20, d_pinky, L_pinky, states[4])
    
    # 2. Refined anatomical coordinate adjustments to make classes unique & realistic
    
    # Fist subclasses (A, E, M, N, S, T) - distinguished by thumb tip location
    if class_label == 'A':
        # Thumb resting on side of Index MCP (landmark 5)
        coords[4] = coords[5] + np.array([0.05 * m, 0.02, 0.02])
    elif class_label == 'E':
        # Thumb folded tightly in front of curled fingers
        coords[4] = coords[9] + np.array([0.00 * m, -0.09, -0.06])
    elif class_label == 'M':
        # Thumb tucked under Ring finger knuckle (13)
        coords[4] = coords[13] + np.array([0.00 * m, -0.05, -0.06])
    elif class_label == 'N':
        # Thumb tucked under Middle finger knuckle (9)
        coords[4] = coords[9] + np.array([0.00 * m, -0.05, -0.06])
    elif class_label == 'S':
        # Thumb wrapped in front of index/middle fingers
        coords[4] = coords[9] + np.array([0.04 * m, -0.02, -0.02])
    elif class_label == 'T':
        # Thumb tucked under Index finger knuckle (5)
        coords[4] = coords[5] + np.array([0.00 * m, -0.06, -0.06])
        
    # Closed curved shapes (O, 0) vs open curved (C)
    elif class_label in ['O', '0']:
        # Tips touching to form a closed circle
        avg_tips = (coords[8] + coords[12]) / 2.0
        coords[4] = avg_tips + np.array([0.01 * m, -0.02, -0.01])
        
    # F, 9: Thumb tip touching Index tip
    elif class_label in ['F', '9']:
        coords[4] = coords[9] + np.array([0.07 * m, 0.02, -0.08])
        coords[8] = coords[4].copy()
        
    # 6: Thumb tip touching Pinky tip
    elif class_label == '6':
        coords[4] = coords[13] + np.array([-0.05 * m, 0.02, -0.08])
        coords[20] = coords[4].copy()
        
    # 7: Thumb tip touching Ring tip
    elif class_label == '7':
        coords[4] = coords[9] + np.array([-0.02 * m, 0.02, -0.08])
        coords[16] = coords[4].copy()
        
    # 8: Thumb tip touching Middle tip
    elif class_label == '8':
        coords[4] = coords[9] + np.array([0.02 * m, 0.02, -0.08])
        coords[12] = coords[4].copy()
        
    # U (touching) vs V/2 (spread)
    elif class_label == 'U':
        # Make index and middle tips touch
        coords[12] = coords[8] + np.array([-0.04 * m, 0.02, 0.00])
    elif class_label in ['V', '2']:
        # Spread index and middle
        coords[8] += np.array([0.05 * m, 0.00, 0.00])
        coords[12] += np.array([-0.05 * m, 0.00, 0.00])
        
    # R: Index and Middle crossed
    elif class_label == 'R':
        # Swap X coordinates of tips slightly to represent crossing
        x8 = coords[8][0]
        x12 = coords[12][0]
        coords[8][0] = x12
        coords[12][0] = x8
        coords[8][2] += 0.03 # Bring Index slightly in front
        
    # K: Thumb tip touching Middle finger PIP/DIP
    elif class_label == 'K':
        coords[4] = coords[10].copy()
        
    # G/Q/L: Hand orientations
    elif class_label == 'G':
        # Pointing sideways: Swap X and Y coordinates to rotate 90 deg
        coords[:, [0, 1]] = coords[:, [1, 0]]
        coords[:, 0] *= m
    elif class_label == 'Q':
        # Pointing downwards: Flip Y axis
        coords[:, 1] *= -0.8
        
    # 3. Apply orientation rotations (pitch, roll, yaw) to resolve identical overlapping hand shapes
    pitch_val = 0
    roll_val = 0
    yaw_val = 0
    
    # Distinguish HELLO, PLEASE, B, 5, THANK YOU, GOOD MORNING, EMERGENCY, AMBULANCE (all flat open hands)
    if class_label == 'HELLO':
        # Wave orientation (side tilt)
        roll_val = 0.3 * m
    elif class_label == 'PLEASE':
        # Chest rub orientation (forward tilt)
        pitch_val = 0.35
    elif class_label == 'THANK YOU':
        # Lip move orientation (downward tilt)
        pitch_val = -0.4
    elif class_label == 'GOOD MORNING':
        # Morning sun orientation (upward tilt)
        pitch_val = 0.45
    elif class_label == 'EMERGENCY':
        # Shaking emergency (yaw tilt)
        yaw_val = 0.35 * m
    elif class_label == 'AMBULANCE':
        # Flashing lights (yaw opposite tilt)
        yaw_val = -0.35 * m
        
    # Distinguish D, 1, Z, HELP, DOCTOR, MEDICINE (all index pointing)
    elif class_label == '1':
        # Number 1: Palm facing backward
        roll_val = 1.0 * m
    elif class_label == 'Z':
        # Alphabet Z: slightly angled tracing tilt
        yaw_val = 0.4 * m
    elif class_label == 'HELP':
        # Hand flat support tilt
        pitch_val = -0.3
    elif class_label == 'DOCTOR':
        # Wrist pulse touch tilt
        pitch_val = 0.3
    elif class_label == 'MEDICINE':
        # Rubbing tilt
        roll_val = -0.4 * m
        
    # Distinguish H, U, V, K, 2, NO (all 2-finger extensions)
    elif class_label == 'H':
        # Index/Middle pointing horizontally sideways
        yaw_val = -1.3 * m
    elif class_label == 'NO':
        # snapping tilt
        pitch_val = 0.5
    elif class_label == '2':
        # Number 2: tilted backward slightly
        roll_val = -0.5 * m
        
    # Distinguish YES, PAIN, A, E, M, N, S, T (all fists)
    elif class_label == 'YES':
        # Fist nodding forward
        pitch_val = 0.5
    elif class_label == 'PAIN':
        # Twisting fists
        yaw_val = 0.6 * m
        
    # Distinguish W, WATER, 6 (all 3-finger extensions)
    elif class_label == 'WATER':
        # Chin touch tilt
        pitch_val = 0.3
        
    # Rotate the coordinates
    if pitch_val != 0 or roll_val != 0 or yaw_val != 0:
        coords = rotate_coords(coords, pitch=pitch_val, roll=roll_val, yaw=yaw_val)
        
    # Invert Y axis to match MediaPipe's top-to-bottom coordinate system
    coords[:, 1] *= -1.0
    
    return coords

def generate_synthetic_features_for_class(class_label, class_idx, samples_per_class=150):
    """
    Generates synthetic 126-dimensional landmarks for a class by calling get_clean_base_posture,
    adding scaling, rotation, Gaussian noise, and random translation.
    Includes left/right hand augmentation for single-handed classes.
    """
    features = []
    labels = []
    
    two_handed_classes = ["HELP", "EMERGENCY", "AMBULANCE", "THANK YOU", "DOCTOR", "GOOD MORNING"]
    is_two_handed = class_label in two_handed_classes

    for _ in range(samples_per_class):
        # 1. Two-handed gesture: Both hands are generated
        if is_two_handed:
            # Generate Left Hand
            left_base = get_clean_base_posture(class_label, is_left=True)
            left_scale = np.random.uniform(0.85, 1.15)
            left_hand = left_base * left_scale
            left_hand += np.random.normal(0, 0.012, left_hand.shape)
            left_hand = normalize_hand_coords(left_hand)
            left_flat = left_hand.flatten()
            
            # Generate Right Hand
            right_base = get_clean_base_posture(class_label, is_left=False)
            right_scale = np.random.uniform(0.85, 1.15)
            right_hand = right_base * right_scale
            right_hand += np.random.normal(0, 0.012, right_hand.shape)
            right_hand = normalize_hand_coords(right_hand)
            right_flat = right_hand.flatten()
            
        # 2. Single-handed gesture: Generate Left hand only or Right hand only (50% split)
        else:
            is_left = (np.random.rand() > 0.5)
            base = get_clean_base_posture(class_label, is_left=is_left)
            scale = np.random.uniform(0.85, 1.15)
            hand = base * scale
            hand += np.random.normal(0, 0.012, hand.shape)
            hand = normalize_hand_coords(hand)
            flat = hand.flatten()
            
            if is_left:
                left_flat = flat
                right_flat = np.zeros(63)
            else:
                left_flat = np.zeros(63)
                right_flat = flat
                
        # Combine Left & Right to form the 126-dimensional features
        combined = np.concatenate([left_flat, right_flat])
        features.append(combined)
        labels.append(class_idx)
        
    return np.array(features), np.array(labels)

def generate_synthetic_dataset(save_path="datasets/synthetic_dataset.npz", samples_per_class=1200):
    """
    Generates a complete synthetic sign language landmark dataset and saves it as .npz.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    all_features = []
    all_labels = []
    
    for idx, label in enumerate(CLASS_LABELS):
        feats, lbls = generate_synthetic_features_for_class(label, idx, samples_per_class)
        all_features.append(feats)
        all_labels.append(lbls)
        
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    np.savez(save_path, features=all_features, labels=all_labels)
    print(f"Synthetic dataset saved at {save_path}. Total samples: {len(all_labels)}")
    
    # Save the class labels file
    labels_file = os.path.join(os.path.dirname(save_path), "class_labels.json")
    with open(labels_file, "w") as f:
        json.dump(CLASS_LABELS, f, indent=4)
        
    return all_features, all_labels, CLASS_LABELS

def create_data_loaders(dataset_path="datasets/synthetic_dataset.npz", batch_size=64, test_size=0.2, val_size=0.1, force_regenerate=False):
    """
    Loads features and labels from .npz and splits them into Train, Val, and Test loaders.
    Generates synthetic dataset if it doesn't exist.
    """
    if force_regenerate or not os.path.exists(dataset_path):
        features, labels, _ = generate_synthetic_dataset(dataset_path)
    else:
        data = np.load(dataset_path)
        features = data['features']
        labels = data['labels']
        
    # Split: Train (70%), Val (10%), Test (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(features, labels, test_size=test_size, random_state=42, stratify=labels)
    
    # Calculate percentage of val relative to train_val to get overall 10%
    val_rel_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_rel_size, random_state=42, stratify=y_train_val)
    
    # Create datasets
    train_dataset = SignLanguageDataset(X_train, y_train)
    val_dataset = SignLanguageDataset(X_val, y_val)
    test_dataset = SignLanguageDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
