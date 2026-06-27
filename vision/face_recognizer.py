import cv2
import os
import numpy as np
import json
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceRecognizer")

class FaceRecognizer:
    def __init__(self):
        # Create the SFace recognizer using the ONNX model
        try:
            self.recognizer = cv2.FaceRecognizerSF.create(config.SFACE_MODEL_PATH, "")
            logger.info("SFace deep learning recognizer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SFace recognizer: {e}")
            raise ImportError(f"SFace initialization failed. Check model path: {config.SFACE_MODEL_PATH}")

        self.label_map_path = os.path.join(config.DATA_DIR, "labels.json")
        self.label_to_name = {}
        self.name_to_label = {}
        
        # In-memory dictionary for user mean embeddings: {username: float_list}
        self.user_embeddings = {}
        self.is_trained = False
        
        # Proactively load the model if it exists
        self.load_model()

    def load_model(self):
        """Loads the trained user embeddings and label mapping if they exist."""
        if os.path.exists(config.EMBEDDINGS_PATH):
            try:
                with open(config.EMBEDDINGS_PATH, 'r') as f:
                    self.user_embeddings = json.load(f)
                
                # Reconstruct label_to_name for compatibility with CLI list options
                self.label_to_name = {idx: name for idx, name in enumerate(sorted(self.user_embeddings.keys()))}
                self.name_to_label = {v: k for k, v in self.label_to_name.items()}
                
                self.is_trained = len(self.user_embeddings) > 0
                logger.info(f"Loaded SFace deep learning model. Registered users: {list(self.user_embeddings.keys())}")
            except Exception as e:
                logger.error(f"Error loading face embeddings model: {e}")
                self.is_trained = False
        else:
            logger.info("No trained face embeddings database found. Ready for training.")
            self.is_trained = False

    def train_model(self):
        """
        Scans config.FACES_DIR for subdirectories of faces, converts images to BGR,
        resizes to 112x112, extracts features, averages them, and saves the embeddings map.
        """
        logger.info("Starting training/extraction of Face Recognition model...")
        
        # Scan directories
        subdirs = [d for d in os.listdir(config.FACES_DIR) if os.path.isdir(os.path.join(config.FACES_DIR, d))]
        
        if not subdirs:
            logger.warning("No face training directories found in config.FACES_DIR.")
            return False

        new_embeddings = {}
        total_samples = 0
        
        for name in subdirs:
            name_path = os.path.join(config.FACES_DIR, name)
            user_features = []
            
            for file_name in os.listdir(name_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(name_path, file_name)
                    
                    # Read image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    # Handle grayscale conversion to BGR if necessary
                    if len(img.shape) == 2 or img.shape[2] == 1:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        
                    # Resize directly to 112x112 for SFace feature extraction
                    aligned_face = cv2.resize(img, (112, 112))
                    
                    try:
                        feat = self.recognizer.feature(aligned_face)
                        user_features.append(feat)
                        total_samples += 1
                    except Exception as e:
                        logger.error(f"Failed to extract features for {img_path}: {e}")
                        
            if user_features:
                # Average all feature vectors for this user
                stacked_feats = np.vstack(user_features)
                mean_feat = np.mean(stacked_feats, axis=0, keepdims=True)
                
                # Normalize the mean feature vector to ensure unit length for cosine match
                norm = np.linalg.norm(mean_feat)
                if norm > 0:
                    mean_feat = mean_feat / norm
                    
                new_embeddings[name] = mean_feat.tolist()[0]
                
        if not new_embeddings:
            logger.warning("No valid face features extracted for training.")
            return False

        # Save embeddings
        try:
            with open(config.EMBEDDINGS_PATH, 'w') as f:
                json.dump(new_embeddings, f, indent=2)
                
            # Save label.json map for legacy / CLI compatibility
            new_label_to_name = {idx: name for idx, name in enumerate(sorted(new_embeddings.keys()))}
            with open(self.label_map_path, 'w') as f:
                json.dump(new_label_to_name, f, indent=2)
                
            self.user_embeddings = new_embeddings
            self.label_to_name = new_label_to_name
            self.name_to_label = {v: k for k, v in self.label_to_name.items()}
            self.is_trained = True
            
            logger.info(f"Model trained successfully. Loaded embeddings for {len(self.user_embeddings)} users with {total_samples} samples total.")
            return True
        except Exception as e:
            logger.error(f"Failed to save trained embeddings: {e}")
            return False

    def predict(self, frame, face=None):
        """
        Predicts the identity of the face.
        Supports:
          - predict(frame, face): Takes full BGR frame and DetectedFace object (Recommended)
          - predict(gray_face): Falls back to BGR-converting the grayscale crop if face is None
        Returns:
            - name: The matched person's name, or 'Unknown'
            - confidence: The matching score (Cosine similarity, higher is better)
        """
        if not self.is_trained or not self.user_embeddings:
            return "Unknown", 0.0
            
        try:
            # 1. Feature extraction
            if face is not None and hasattr(face, "raw_face"):
                # Use standard deep learning alignCrop and feature extraction
                aligned_face = self.recognizer.alignCrop(frame, face.raw_face)
                feat = self.recognizer.feature(aligned_face)
            else:
                # Fallback for legacy calls passing a pre-cropped grayscale face image
                cropped_img = frame
                if len(cropped_img.shape) == 2:
                    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
                aligned_face = cv2.resize(cropped_img, (112, 112))
                feat = self.recognizer.feature(aligned_face)

            # Normalize the input feature vector
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm

            # 2. Compare against all known user embeddings
            best_name = "Unknown"
            best_score = -1.0
            
            for name, embedding_list in self.user_embeddings.items():
                target_feat = np.array(embedding_list, dtype=np.float32).reshape(1, 128)
                
                # Match score using Cosine distance/similarity
                score = self.recognizer.match(feat, target_feat, cv2.FaceRecognizerSF_FR_COSINE)
                
                if score > best_score:
                    best_score = score
                    best_name = name

            # 3. Apply confidence threshold (Cosine Similarity >= threshold is a match)
            if best_score >= config.CONFIDENCE_THRESHOLD:
                return best_name, float(best_score)
            else:
                return "Unknown", float(best_score)
                
        except Exception as e:
            logger.error(f"Prediction error in FaceRecognizer: {e}")
            return "Unknown", 0.0
