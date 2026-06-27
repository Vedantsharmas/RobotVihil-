import cv2
import os
import numpy as np
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceDetector")

class DetectedFace:
    """
    A wrapper around YuNet's raw face detection row.
    Implements tuple unpacking and index operations (x, y, w, h)
    for 100% backward compatibility with Haar Cascade bounding box outputs.
    """
    def __init__(self, raw_face):
        self.raw_face = raw_face
        # bbox is at indices 0 to 3: [x, y, width, height]
        self.x = int(raw_face[0])
        self.y = int(raw_face[1])
        self.w = int(raw_face[2])
        self.h = int(raw_face[3])
        self.score = float(raw_face[14]) if len(raw_face) > 14 else 1.0
        
    def __getitem__(self, idx):
        if idx == 0: return self.x
        if idx == 1: return self.y
        if idx == 2: return self.w
        if idx == 3: return self.h
        raise IndexError("Index out of range for face bounding box (must be 0-3).")
        
    def __iter__(self):
        return iter([self.x, self.y, self.w, self.h])
        
    def __len__(self):
        return 4
        
    def __repr__(self):
        return f"DetectedFace(x={self.x}, y={self.y}, w={self.w}, h={self.h}, score={self.score:.2f})"

class FaceDetector:
    def __init__(self):
        # Initialize YuNet deep learning face detector
        if not os.path.exists(config.YUNET_MODEL_PATH):
            logger.error(f"YuNet ONNX model not found at {config.YUNET_MODEL_PATH}!")
            raise FileNotFoundError(f"YuNet model file not found at {config.YUNET_MODEL_PATH}")
            
        self.detector = cv2.FaceDetectorYN.create(
            config.YUNET_MODEL_PATH,
            "",
            (config.FRAME_WIDTH, config.FRAME_HEIGHT)
        )
        
        # Load OpenCV's built-in Haar Cascade for smile detection (legacy fallback)
        smile_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
        self.smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
        if self.smile_cascade.empty():
            logger.warning("Failed to load Haar Cascade smile classifier.")
            
        logger.info("Face detector (YuNet) and smile detector initialized successfully.")

        # Load pre-trained ONNX expression recognition model (FER+)
        import urllib.request
        self.model_path = os.path.join(config.DATA_DIR, "emotion-ferplus-8.onnx")
        self.model_url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
        self.net = None

        if not os.path.exists(self.model_path):
            logger.info(f"ONNX expression model not found at {self.model_path}. Downloading from ONNX Model Zoo...")
            try:
                req = urllib.request.Request(
                    self.model_url,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    data = response.read()
                    with open(self.model_path, 'wb') as f:
                        f.write(data)
                logger.info("ONNX expression model downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download ONNX expression model: {e}. Falling back to legacy heuristics.")

        if os.path.exists(self.model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                # Optimize OpenCV DNN target for CPU execution
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logger.info("ONNX expression model loaded successfully using OpenCV DNN (CPU Optimized).")
                # Determine scaling factor dynamically based on model file size (custom CPU model is smaller, < 10MB)
                self.use_normalization = False
                file_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
                if file_size_mb < 10.0:
                    self.use_normalization = True
                    logger.info(f"Custom model detected ({file_size_mb:.2f}MB). Using 1/255 scaling.")
                else:
                    logger.info(f"Original model detected ({file_size_mb:.2f}MB). Using 1.0 scaling.")
            except Exception as e:
                logger.error(f"Failed to load ONNX model using OpenCV DNN: {e}. Falling back to legacy heuristics.")

    def detect_faces(self, frame):
        """
        Detects faces in the given frame using deep learning YuNet detector.
        Returns:
            - gray_frame: Grayscale version of the frame (for compatibility)
            - faces: A list of DetectedFace objects (behaves like (x, y, w, h) tuples)
        """
        if frame is None:
            return None, []

        # Convert to grayscale for legacy modules that expect it
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update input size based on the current frame size dynamically
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        
        # Run detection
        retval, detections = self.detector.detect(frame)
        
        faces = []
        if detections is not None:
            for detection in detections:
                faces.append(DetectedFace(detection))
                
        return gray_frame, faces

    def detect_expression(self, gray_face):
        """
        Detects facial expressions from a grayscale face crop.
        Returns:
            - mood: String representing the detected mood/expression ("Happy", "Surprised", "Focused/Angry", "Sad/Tired", "Neutral").
        """
        if gray_face is None or gray_face.size == 0:
            return "Neutral"

        # Try DNN-based prediction first if loaded
        if self.net is not None:
            try:
                # Apply global histogram equalization to normalize contrast and lighting
                equalized_face = cv2.equalizeHist(gray_face)
                # Preprocess: resize directly to 64x64.
                face_64 = cv2.resize(equalized_face, (64, 64))
                # Set scale factor depending on whether model expects normalized inputs
                scalefactor = 1.0 / 255.0 if self.use_normalization else 1.0
                blob = cv2.dnn.blobFromImage(face_64, scalefactor=scalefactor, size=(64, 64), mean=0, swapRB=False, crop=False)
                
                self.net.setInput(blob)
                preds = self.net.forward()
                
                # FERPlus mapping:
                # 0: neutral -> Neutral
                # 1: happiness -> Happy
                # 2: surprise -> Surprised
                # 3: sadness -> Sad/Tired
                # 4: anger -> Focused/Angry
                # 5: disgust -> Focused/Angry
                # 6: fear -> Sad/Tired
                # 7: contempt -> Focused/Angry
                idx = int(np.argmax(preds[0]))
                mapping = {
                    0: "Neutral",
                    1: "Happy",
                    2: "Surprised",
                    3: "Sad/Tired",
                    4: "Focused/Angry",
                    5: "Focused/Angry",
                    6: "Sad/Tired",
                    7: "Focused/Angry"
                }
                return mapping.get(idx, "Neutral")
            except Exception as e:
                logger.warning(f"Error running ONNX expression model: {e}. Falling back to legacy heuristics.")

        # Legacy Haar-cascade and Sobel heuristic-based detection fallback
        try:
            if gray_face.shape != (200, 200):
                gray_face = cv2.resize(gray_face, (200, 200))
            # 1. Eyebrow Gradient Computation (Used for Focused/Angry and Sad/Tired)
            # Use a tight eyebrow crop: y[35:65], x[45:155] (completely excludes eyes and forehead)
            eyebrow_crop = gray_face[35:65, 45:155]
            # Use CLAHE instead of global histogram equalization for stable adaptive contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            eyebrow_clahe = clahe.apply(eyebrow_crop)
            sobel_x = cv2.Sobel(eyebrow_clahe, cv2.CV_64F, 1, 0, ksize=3)
            mean_grad = np.mean(np.abs(sobel_x))

            # --- Check 1: Focused/Angry (Active Eyebrow furrowing - strong priority) ---
            # CLAHE-based threshold: Focused/Angry > 38.0
            if mean_grad > 38.0:
                return "Focused/Angry"

            # --- Check 2: Mouth Open / Surprised ---
            # Use a tight mouth crop to completely ignore nostrils, nose lines, and chin shadows: y[125:185], x[45:155]
            mouth_tight = gray_face[125:185, 45:155]
            mean_brightness = np.mean(mouth_tight)
            # Highly responsive multiplier set to 0.55, size 12, aspect ratio 0.5, and tight horizontal center checks
            thresh_val = int(mean_brightness * 0.55)
            _, thresh = cv2.threshold(mouth_tight, thresh_val, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            surprised = False
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                # Centered horizontal checks adjusted for the 110px crop width
                if w > 12 and h > 12 and 15 < (x + w/2) < 95:
                    aspect_ratio = h / w
                    if aspect_ratio > 0.5:
                        surprised = True
                        break
            if surprised:
                return "Surprised"

            # --- Check 3: Smile Detection (Happy) ---
            # Tighter crop starting at y=130 to exclude nostrils and nasolabial folds: y[130:185], x[45:155]
            mouth_crop = gray_face[130:185, 45:155]
            # Higher minNeighbors (12) and scaleFactor (1.35) for zero false positives on neutral faces
            smiles = self.smile_cascade.detectMultiScale(mouth_crop, scaleFactor=1.35, minNeighbors=12, minSize=(15, 15))
            if len(smiles) > 0:
                return "Happy"

            # --- Check 4: Sad/Tired (Relaxed/drooping eyebrows) ---
            # CLAHE-based threshold: Sad/Tired < 22.0
            if mean_grad < 22.0:
                return "Sad/Tired"

        except Exception as e:
            logger.error(f"Error in legacy expression detection: {e}")

        return "Neutral"

    def draw_faces(self, frame, faces, color=(0, 255, 0), thickness=2):
        """
        Draws rectangles around detected faces on the source frame.
        """
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        return frame


from collections import deque, Counter
import time

class TemporalFilter:
    def __init__(self, size=10):
        self.size = size
        self.names = deque(maxlen=size)
        self.moods = deque(maxlen=size)
        self.last_seen = 0.0

    def update(self, name, mood):
        self.names.append(name)
        self.moods.append(mood)
        self.last_seen = time.time()

    def get_smoothed(self):
        if not self.names:
            return "Unknown", "Neutral"
        
        name_counts = Counter(self.names)
        smoothed_name = name_counts.most_common(1)[0][0]
        
        mood_counts = Counter(self.moods)
        smoothed_mood = mood_counts.most_common(1)[0][0]
        
        return smoothed_name, smoothed_mood

    def clear(self):
        self.names.clear()
        self.moods.clear()


