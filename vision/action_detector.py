import cv2
import numpy as np
from collections import deque, Counter
import time
import os
import json
import config

class ActionDetector:
    def __init__(self, history_size=30):
        self.history_size = history_size
        
        # Buffers for face tracking (cx, cy, w, h)
        self.face_history = deque(maxlen=history_size)
        
        # Buffers for motion detection
        self.prev_mouth_crop = None
        self.prev_chin_crop = None
        
        self.mouth_motion_history = deque(maxlen=history_size)
        self.chin_motion_history = deque(maxlen=history_size)
        
        # Buffers for mouth opening (singing detection)
        self.mouth_open_history = deque(maxlen=history_size)
        
        # Buffer for final action decisions to apply temporal smoothing
        self.action_history = deque(maxlen=15)
        
        # Cooldown state for eating detection
        self.last_eating_trigger = 0.0

        # KNN Dataset variables
        self.dataset_path = os.path.join(config.DATA_DIR, "action_dataset.json")
        self.dataset = []
        self.means = []
        self.stds = []
        self.load_dataset()

    def load_dataset(self):
        """Loads action dataset from disk and computes feature standardization parameters."""
        dataset_loaded = False
        if os.path.exists(self.dataset_path):
            try:
                with open(self.dataset_path, "r") as f:
                    self.dataset = json.load(f)
                dataset_loaded = True
            except Exception as e:
                print(f"[ActionDetector] Error loading action dataset: {e}")
                self.dataset = []
        else:
            self.dataset = []

        # If no dataset was loaded or it is empty, generate baseline samples
        if not dataset_loaded or len(self.dataset) == 0:
            self.generate_default_dataset()

        # Compute feature normalization parameters (means/stds) across dataset
        if len(self.dataset) >= 5:
            try:
                feats = np.array([sample["features"] for sample in self.dataset])
                self.means = np.mean(feats, axis=0).tolist()
                # Add tiny epsilon to prevent division by zero for static features
                self.stds = (np.std(feats, axis=0) + 1e-6).tolist()
                print(f"[ActionDetector] Loaded action dataset with {len(self.dataset)} samples. Standardizations computed.")
            except Exception as e:
                print(f"[ActionDetector] Error computing dataset normalization: {e}")
                self.means = []
                self.stds = []
        else:
            self.means = []
            self.stds = []

    def generate_default_dataset(self):
        """Generates realistic default baseline samples for physical actions if dataset is empty/small."""
        print("[ActionDetector] Generating realistic default baseline samples for physical actions...")
        default_samples = []
        
        # We generate 30 samples per class using uniform random distributions based on calibrated ranges
        # [std_cx, std_cy, std_w, range_cx, range_cy, range_w, avg_mouth, avg_chin, open_mean, open_std]
        
        # 1. Coding/Typing (Very quiet head, minimal mouth/chin, minimal mouth opening)
        for _ in range(30):
            std_cx = np.random.uniform(0.2, 1.5)
            std_cy = np.random.uniform(0.2, 1.2)
            std_w = np.random.uniform(0.1, 1.0)
            range_cx = np.random.uniform(0.5, 3.5)
            range_cy = np.random.uniform(0.5, 3.0)
            range_w = np.random.uniform(0.3, 2.5)
            avg_mouth = np.random.uniform(0.005, 0.025)
            avg_chin = np.random.uniform(0.005, 0.025)
            open_mean = np.random.uniform(0.005, 0.02)
            open_std = np.random.uniform(0.001, 0.005)
            
            feat = [std_cx, std_cy, std_w, range_cx, range_cy, range_w, avg_mouth, avg_chin, open_mean, open_std]
            default_samples.append({"action": "Coding/Typing", "features": feat, "timestamp": time.time()})
            
        # 2. Eating (Quiet head, high chin motion, moderate mouth motion, low-moderate open)
        for _ in range(30):
            std_cx = np.random.uniform(0.5, 2.5)
            std_cy = np.random.uniform(0.5, 2.5)
            std_w = np.random.uniform(0.3, 2.0)
            range_cx = np.random.uniform(1.5, 6.0)
            range_cy = np.random.uniform(1.5, 6.0)
            range_w = np.random.uniform(1.0, 5.0)
            avg_mouth = np.random.uniform(0.07, 0.14)
            avg_chin = np.random.uniform(0.15, 0.28) # high chin motion signature
            open_mean = np.random.uniform(0.02, 0.06)
            open_std = np.random.uniform(0.015, 0.035)
            
            feat = [std_cx, std_cy, std_w, range_cx, range_cy, range_w, avg_mouth, avg_chin, open_mean, open_std]
            default_samples.append({"action": "Eating", "features": feat, "timestamp": time.time()})

        # 3. Singing (Quiet/medium head bobbing, very high mouth motion, high mouth opening ratio and variance)
        for _ in range(30):
            std_cx = np.random.uniform(0.8, 4.0)
            std_cy = np.random.uniform(0.8, 4.0)
            std_w = np.random.uniform(0.5, 3.0)
            range_cx = np.random.uniform(2.0, 9.0)
            range_cy = np.random.uniform(2.0, 9.0)
            range_w = np.random.uniform(1.5, 7.0)
            avg_mouth = np.random.uniform(0.09, 0.18)
            avg_chin = np.random.uniform(0.03, 0.08)
            open_mean = np.random.uniform(0.06, 0.15)
            open_std = np.random.uniform(0.04, 0.09)
            
            feat = [std_cx, std_cy, std_w, range_cx, range_cy, range_w, avg_mouth, avg_chin, open_mean, open_std]
            default_samples.append({"action": "Singing", "features": feat, "timestamp": time.time()})

        # 4. Walking (High displacement range/variance on head, high crop motion due to jitter)
        for _ in range(30):
            std_cx = np.random.uniform(22.0, 40.0)
            std_cy = np.random.uniform(18.0, 35.0)
            std_w = np.random.uniform(12.0, 25.0)
            range_cx = np.random.uniform(65.0, 120.0)
            range_cy = np.random.uniform(50.0, 100.0)
            range_w = np.random.uniform(28.0, 60.0)
            avg_mouth = np.random.uniform(0.03, 0.15)
            avg_chin = np.random.uniform(0.03, 0.15)
            open_mean = np.random.uniform(0.01, 0.03)
            open_std = np.random.uniform(0.005, 0.02)
            
            feat = [std_cx, std_cy, std_w, range_cx, range_cy, range_w, avg_mouth, avg_chin, open_mean, open_std]
            default_samples.append({"action": "Walking", "features": feat, "timestamp": time.time()})
            
        self.dataset.extend(default_samples)
        try:
            os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
            with open(self.dataset_path, "w") as f:
                json.dump(self.dataset, f, indent=2)
            print(f"[ActionDetector] Successfully generated and saved {len(default_samples)} baseline samples to {self.dataset_path}")
        except Exception as e:
            print(f"[ActionDetector] Error saving default action dataset: {e}")

    def save_sample(self, action_name, feature_vector):
        """Appends a new sample to the dataset on disk and recomputes statistics."""
        if not feature_vector:
            return
        
        sample = {
            "action": action_name,
            "features": feature_vector,
            "timestamp": time.time()
        }
        self.dataset.append(sample)
        
        try:
            with open(self.dataset_path, "w") as f:
                json.dump(self.dataset, f, indent=2)
            # Recompute normalization stats
            self.load_dataset()
        except Exception as e:
            print(f"[ActionDetector] Error saving action sample: {e}")

    def get_feature_vector(self):
        """
        Extracts a 10-dimensional visual feature vector representing the current state.
        Returns: [std_cx, std_cy, std_w, range_cx, range_cy, range_w, avg_mouth, avg_chin, open_mean, open_std]
        """
        if len(self.face_history) < 15:
            return None
            
        cxs = [f[0] for f in self.face_history]
        cys = [f[1] for f in self.face_history]
        ws = [f[2] for f in self.face_history]
        
        std_cx = float(np.std(cxs))
        std_cy = float(np.std(cys))
        std_w = float(np.std(ws))
        
        range_cx = float(max(cxs) - min(cxs))
        range_cy = float(max(cys) - min(cys))
        range_w = float(max(ws) - min(ws))
        
        avg_mouth = float(np.mean(self.mouth_motion_history)) if self.mouth_motion_history else 0.0
        avg_chin = float(np.mean(self.chin_motion_history)) if self.chin_motion_history else 0.0
        
        open_mean = float(np.mean(self.mouth_open_history)) if self.mouth_open_history else 0.0
        open_std = float(np.std(self.mouth_open_history)) if self.mouth_open_history else 0.0
        
        return [
            std_cx, std_cy, std_w,
            range_cx, range_cy, range_w,
            avg_mouth, avg_chin,
            open_mean, open_std
        ]

    def normalize(self, feat):
        """Standardizes a feature vector using dataset statistics."""
        if not self.means or not self.stds:
            return feat
        return [(feat[i] - self.means[i]) / self.stds[i] for i in range(len(feat))]

    def classify_knn(self, norm_feat, k=5):
        """Classifies the normalized feature vector using Weighted KNN and Inverse Distance Weighting."""
        if not self.dataset:
            return "Neutral"
            
        # Define feature weights to emphasize characteristic motion signatures
        # features order: [std_cx, std_cy, std_w, range_cx, range_cy, range_w, avg_mouth, avg_chin, open_mean, open_std]
        weights = np.array([1.0, 1.0, 1.0, 1.5, 1.5, 1.2, 2.2, 2.5, 2.2, 2.2])
        
        dists = []
        for sample in self.dataset:
            # Normalize sample features
            norm_sample = self.normalize(sample["features"])
            # Compute Weighted Euclidean distance
            diff = np.array(norm_feat) - np.array(norm_sample)
            weighted_diff = diff * weights
            dist = float(np.sqrt(np.sum(weighted_diff ** 2)))
            dists.append((dist, sample["action"]))
            
        dists.sort(key=lambda x: x[0])
        
        # Accumulate weights via Inverse Distance Weighting
        class_weights = {}
        for dist, action in dists[:min(k, len(dists))]:
            weight = 1.0 / (dist + 1e-4)
            class_weights[action] = class_weights.get(action, 0.0) + weight
            
        if not class_weights:
            return "Neutral"
            
        # Select action with the maximum accumulated distance weight
        best_action = max(class_weights, key=class_weights.get)
        return best_action

    def detect(self, frame, gray_frame, faces):
        """
        Analyzes the current frame and detected face bounding boxes to recognize actions.
        Returns the smoothed action classification.
        """
        now = time.time()
        
        # If no face is detected
        if len(faces) == 0:
            self.face_history.clear()
            self.mouth_motion_history.clear()
            self.chin_motion_history.clear()
            self.mouth_open_history.clear()
            self.prev_mouth_crop = None
            self.prev_chin_crop = None
            
            self.action_history.append("Neutral")
            return self.get_smoothed_action()

        # Target the main/closest face (largest face area)
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        
        cx = x + w / 2.0
        cy = y + h / 2.0
        self.face_history.append((cx, cy, w, h))

        # --- Stabilize coordinates to prevent crop boundary jitter from generating false motion ---
        recent_faces = list(self.face_history)[-5:]
        avg_cx = np.mean([f[0] for f in recent_faces])
        avg_cy = np.mean([f[1] for f in recent_faces])
        avg_w = np.mean([f[2] for f in recent_faces])
        avg_h = np.mean([f[3] for f in recent_faces])
        
        x_stab = int(avg_cx - avg_w / 2.0)
        y_stab = int(avg_cy - avg_h / 2.0)
        w_stab = int(avg_w)
        h_stab = int(avg_h)

        # Limit crops to image boundaries
        frame_h, frame_w = gray_frame.shape
        
        # Mouth region: lower third, middle of face
        my_start = max(0, y_stab + int(0.65 * h_stab))
        my_end = min(frame_h, y_stab + int(0.95 * h_stab))
        mx_start = max(0, x_stab + int(0.22 * w_stab))
        mx_end = min(frame_w, x_stab + int(0.78 * w_stab))
        
        # Chin/neck region below the face
        cy_start = min(frame_h - 1, y_stab + h_stab)
        cy_end = min(frame_h, y_stab + int(1.35 * h_stab))
        cx_start = max(0, x_stab + int(0.20 * w_stab))
        cx_end = min(frame_w, x_stab + int(0.80 * w_stab))

        # --- Extract Crops ---
        mouth_crop = gray_frame[my_start:my_end, mx_start:mx_end]
        chin_crop = gray_frame[cy_start:cy_end, cx_start:cx_end]

        # Draw dynamic diagnostics on color frame to visually showcase active tracking
        cv2.rectangle(frame, (mx_start, my_start), (mx_end, my_end), (255, 120, 0), 1)  # Light blue for mouth
        cv2.rectangle(frame, (cx_start, cy_start), (cx_end, cy_end), (0, 255, 255), 1)  # Yellow for chin/neck

        # Initialize crops if empty
        if mouth_crop.size == 0 or chin_crop.size == 0:
            self.action_history.append("Coding/Typing")
            return self.get_smoothed_action()

        # --- Motion Analysis (Frame Differencing) ---
        motion_mouth = 0.0
        motion_chin = 0.0

        if self.prev_mouth_crop is not None and self.prev_mouth_crop.shape == mouth_crop.shape:
            diff_m = cv2.absdiff(mouth_crop, self.prev_mouth_crop)
            _, thresh_m = cv2.threshold(diff_m, 15, 255, cv2.THRESH_BINARY)
            motion_mouth = np.sum(thresh_m == 255) / float(thresh_m.size)
            
        if self.prev_chin_crop is not None and self.prev_chin_crop.shape == chin_crop.shape:
            diff_c = cv2.absdiff(chin_crop, self.prev_chin_crop)
            _, thresh_c = cv2.threshold(diff_c, 15, 255, cv2.THRESH_BINARY)
            motion_chin = np.sum(thresh_c == 255) / float(thresh_c.size)

        self.prev_mouth_crop = mouth_crop.copy()
        self.prev_chin_crop = chin_crop.copy()

        self.mouth_motion_history.append(motion_mouth)
        self.chin_motion_history.append(motion_chin)

        # --- Mouth Opening (Open Cavity Ratio for Singing) ---
        # Dark regions inside the mouth cavity represent opening mouth
        mean_b = np.mean(mouth_crop)
        thresh_val = int(mean_b * 0.70)
        _, mouth_thresh = cv2.threshold(mouth_crop, thresh_val, 255, cv2.THRESH_BINARY_INV)
        mouth_open_ratio = np.sum(mouth_thresh == 255) / float(mouth_thresh.size)
        self.mouth_open_history.append(mouth_open_ratio)

        # --- CLASSIFICATION ROUTINE ---
        feat = self.get_feature_vector()
        
        # If dataset has enough samples (e.g. at least 15 samples), run KNN ML classification
        if feat and len(self.dataset) >= 15:
            norm_feat = self.normalize(feat)
            pred_action = self.classify_knn(norm_feat)
            self.action_history.append(pred_action)
            return self.get_smoothed_action()

        # FALLBACK: Heuristic-based action classification (no dataset trained yet)
        
        # Compute head motion metrics for walking and suppression
        std_cx, std_cy, std_w = 0.0, 0.0, 0.0
        range_cx, range_cy, range_w = 0.0, 0.0, 0.0
        if len(self.face_history) >= 15:
            cxs = [f[0] for f in self.face_history]
            cys = [f[1] for f in self.face_history]
            ws = [f[2] for f in self.face_history]
            
            std_cx = np.std(cxs)
            std_cy = np.std(cys)
            std_w = np.std(ws)
            
            range_cx = max(cxs) - min(cxs)
            range_cy = max(cys) - min(cys)
            range_w = max(ws) - min(ws)
            
        # --- HEURISTIC 1: Walking / Moving Detection ---
        # Walking is characterized by larger displacement ranges
        is_walking = (std_cx > 20.0 or std_cy > 18.0 or std_w > 12.0) and (range_cx > 60.0 or range_cy > 45.0 or range_w > 25.0)
        if is_walking:
            self.action_history.append("Walking")
            return self.get_smoothed_action()

        # Suppress Eating & Singing checks if the head is shifting/moving to prevent crop motion artifacts
        is_head_moving = (std_cx > 8.0 or std_cy > 8.0 or std_w > 4.0)
        
        if not is_head_moving:
            # --- HEURISTIC 2: Eating / Drinking ---
            is_eating = False
            if len(self.chin_motion_history) >= 10:
                max_recent_chin = max(list(self.chin_motion_history)[-10:])
                avg_recent_mouth = np.mean(list(self.mouth_motion_history)[-5:])
                
                if max_recent_chin > 0.12 and avg_recent_mouth > 0.06:
                    is_eating = True
                    self.last_eating_trigger = now

            if is_eating or (now - self.last_eating_trigger < 2.0):
                self.action_history.append("Eating")
                return self.get_smoothed_action()

            # --- HEURISTIC 3: Singing / Speaking ---
            if len(self.mouth_open_history) >= 15:
                open_std = np.std(list(self.mouth_open_history)[-15:])
                open_mean = np.mean(list(self.mouth_open_history)[-15:])
                
                if open_std > 0.035 and open_mean > 0.02:
                    self.action_history.append("Singing")
                    return self.get_smoothed_action()

        # --- DEFAULT: Coding / Typing ---
        self.action_history.append("Coding/Typing")
        return self.get_smoothed_action()

    def get_smoothed_action(self):
        """Returns the most common action in the recent decision buffer to prevent flickering."""
        if not self.action_history:
            return "Neutral"
        counts = Counter(self.action_history)
        return counts.most_common(1)[0][0]
