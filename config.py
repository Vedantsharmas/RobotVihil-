import os

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")
MODEL_PATH = os.path.join(DATA_DIR, "face_recognizer_model.yml")

# Create directories if they do not exist
os.makedirs(FACES_DIR, exist_ok=True)

# Vision configuration
CAMERA_INDEX = 0  # 0 is usually the default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Deep learning model configurations
YUNET_MODEL_PATH = os.path.join(DATA_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL_PATH = os.path.join(DATA_DIR, "face_recognition_sface_2021dec.onnx")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "face_embeddings.json")

# Face recognition parameters
# For SFace, confidence represents Cosine Similarity (range -1 to 1, higher is better).
# Standard threshold for matching is 0.363.
CONFIDENCE_THRESHOLD = 0.363
GREETING_COOLDOWN = 60.0  # seconds between greetings for the same person

# Single person mode and greeting timing settings
SINGLE_PERSON_ONLY = True  # If True, the robot only detects and greets the single closest person (largest face)
GREETING_STABILIZATION_TIME = 1.0  # Seconds to wait/stabilize before speaking a greeting

# Face detection parameters (Haar Cascades)
FACE_SCALE_FACTOR = 1.1   # Standard scale factor for robust detection
FACE_MIN_NEIGHBORS = 5    # 5 or 6 reduces false positive boxes on backgrounds


# Voice configuration
VOICE_RATE = 150  # Speech speed (words per minute)
VOICE_VOLUME = 1.0  # Speech volume (0.0 to 1.0)
USE_ONLINE_TTS = True  # Set to True to use gTTS (needs internet), False for pyttsx3 (offline)

# Wake words and command configuration
WAKE_WORDS = ["robot", "assistant", "hey robot", "hello", "hi", "hey"]
COMMAND_TIMEOUT = 5  # Seconds to wait for command after wake word

# Conversational AI Configuration (Gemini API)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Target user configuration (only greet this specific user, set to None to greet any recognized user)
TARGET_USER = "janvi shah"
LOCK_TIMEOUT = 30.0  # seconds to automatically lock if target user is not seen

# Greeting template for recognized users (uses {name} and {mood} placeholders)
GREETING_TEMPLATE = "Hello {name}, you look {mood} today! Nice to see you."

# Profound expression-based greetings for the target user (Janvi Shah)
EXPRESSION_GREETINGS = {
    "happy": "A smile is the reflection of a peaceful mind, Janvi Shah. Your happiness creates a ripple of positive energy. What beautiful creation or thought are we celebrating today?",
    "angry": "I notice a shadow of tension in your expression, Janvi Shah. In frustration, there is often a quiet search for resolution. Take a slow breath. What has disrupted your peace, and how can we solve it?",
    "surprised": "Wonder is the spark of discovery, Janvi Shah. The unexpected is what keeps our universe alive and learning. What sudden novelty has caught your attention today?",
    "sad": "Even the quietest stars need the darkness to be seen, Janvi Shah. It is okay to feel weary or quiet. Give yourself grace, rest, and know that I am here to hold space or assist whenever you are ready.",
    "neutral": "Hello, Janvi Shah. In this calm and focused silence, we find the canvas for new ideas. I am online and ready. What shall we bring to life today?"
}

# General expression-based greetings for any other recognized user (uses {name} placeholder)
GENERAL_EXPRESSION_GREETINGS = {
    "happy": "A smile is the reflection of a peaceful mind, {name}. Your happiness creates a ripple of positive energy. What beautiful creation or thought are we celebrating today?",
    "angry": "I notice a shadow of tension in your expression, {name}. In frustration, there is often a quiet search for resolution. Take a slow breath. What has disrupted your peace, and how can we solve it?",
    "surprised": "Wonder is the spark of discovery, {name}. The unexpected is what keeps our universe alive and learning. What sudden novelty has caught your attention today?",
    "sad": "Even the quietest stars need the darkness to be seen, {name}. It is okay to feel weary or quiet. Give yourself grace, rest, and know that I am here to hold space or assist whenever you are ready.",
    "neutral": "Hello, {name}. In this calm and focused silence, we find the canvas for new ideas. I am online and ready. What shall we bring to life today?"
}

# Expression-based greetings for unknown guests
GUEST_EXPRESSION_GREETINGS = {
    "happy": "Hello guest! I see a bright smile on your face. Happiness is contagious. How can I help make your day even better?",
    "angry": "Hello guest. I notice some tension or frustration in your expression. Take a deep breath. Is there anything I can help you resolve?",
    "surprised": "Hello there! You look quite surprised. Did something unexpected happen? I am here to help you explore or find answers.",
    "sad": "Hello guest. You seem a bit sad or tired today. Remember to take it easy and give yourself some rest. I am here if you need any assistance.",
    "neutral": "Hello! I detect a visitor. I am online and ready to assist. Please let me know if you have any commands or questions."
}

# Groq API Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")




