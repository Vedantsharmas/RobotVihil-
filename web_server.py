import os
import sys

# Suppress OpenCV videoio warnings and verbose logging
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_LOG_LEVEL"] = "0"

import cv2
import time
import threading
import queue
import json
import logging
from flask import Flask, Response, jsonify, request, send_from_directory

import config
from vision.face_detector import FaceDetector, TemporalFilter
from vision.face_recognizer import FaceRecognizer
from vision.action_detector import ActionDetector
from voice.listener import VoiceListener
from voice.speaker import VoiceSpeaker
from voice.chatbot import Chatbot

# Setup logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RobotWebServer")

app = Flask(__name__)

# Global variables for sharing state and locks
latest_frame_bytes = None
frame_lock = threading.Lock()
system_unlocked = not getattr(config, "TARGET_USER", None)
last_seen_target_user = 0.0
latest_mood = "Neutral"
latest_action = "Coding/Typing"
recording_action = None
recording_samples_count = 0
recording_lock = threading.Lock()

client_queues = []
client_lock = threading.Lock()

# Instances
speaker = VoiceSpeaker()
chatbot = Chatbot()

def add_event(event_type, event_data):
    """Pushes a structured event to all connected SSE clients."""
    event = {
        "timestamp": time.time(),
        "type": event_type,
        "data": event_data
    }
    with client_lock:
        for q in client_queues:
            q.put(event)

def process_command(command, speaker, chatbot):
    """Core command processing logic."""
    import re
    logger.info(f"Processing command: '{command}'")
    response_text = ""
    command_clean = command.strip().lower()
    
    current_frame = None
    with frame_lock:
        current_frame = latest_frame_bytes
        
    global latest_mood, latest_action
    # 1. Check Chatbot (Gemini / Local Q&A Grounded) FIRST
    chatbot_response = chatbot.get_response(command, image_bytes=current_frame, mood=latest_mood, action=latest_action)
    is_default_warning = "verify the Gemini API key" in chatbot_response or chatbot_response.startswith("I heard you say:")
    
    if chatbot_response and not is_default_warning:
        response_text = chatbot_response
    else:
        # 2. Check standard commands using word boundaries to avoid substring match bugs (e.g. matching 'hi' in 'vihil')
        def has_word(word_pattern, text):
            return re.search(r'\b' + word_pattern + r'\b', text) is not None

        if "bring me" in command_clean or "get me" in command_clean:
            item = command.replace("bring me", "").replace("get me", "").strip()
            response_text = f"Okay, I will navigate to find the {item} and bring it to you."
            
        elif has_word("locate", command_clean) or command_clean.startswith("find the"):
            item = command.replace("find the", "").replace("locate", "").strip()
            response_text = f"Initiating object detection search loop for {item}."
            
        elif has_word("patrol", command_clean):
            response_text = "Starting room patrol sequence. Monitoring for changes."
            
        elif any(has_word(w, command_clean) for w in ["who are you", "your name"]):
            response_text = "I am your mobile robot assistant, powered by a Raspberry Pi."
            
        elif any(has_word(w, command_clean) for w in ["stop", "halt"]):
            response_text = "Stopping all current operations."
            
        elif any(has_word(w, command_clean) for w in ["hello", "hi", "hey"]):
            response_text = "Hello! How can I help you today?"
            
        elif any(has_word(w, command_clean) for w in ["how are you", "how's it going"]):
            response_text = "I am doing great, thank you for asking. I am ready to assist you."
            
        elif any(has_word(w, command_clean) for w in ["thank you", "thanks"]):
            response_text = "You are very welcome!"
            
        else:
            # Fallback to the chatbot response warning
            response_text = chatbot_response

    if response_text:
        speaker.speak(response_text)
        add_event("response", {"text": response_text})


def voice_listener_worker(listener, speaker, chatbot):
    """Background thread that continuously listens for audio input."""
    global system_unlocked
    logger.info("Voice listener thread started.")
    add_event("status", {"mic_status": "calibrating", "info": "Calibrating microphone..."})
    listener.calibrate()
    add_event("status", {"mic_status": "listening", "info": "Microphone ready"})
    
    last_lock_log = 0
    
    while True:
        try:
            # Always active: mic is never locked. We no longer pause voice listener thread here.
                
            add_event("status", {"mic_status": "idle", "info": "Listening for wake word..."})
            text = listener.listen(timeout=None)
            if not text:
                continue
            
            wake_detected, command = listener.check_and_extract_command(text)
            if wake_detected:
                logger.info("Wake word detected!")
                add_event("voice_alert", {"info": "Wake word detected", "text": text})
                if command:
                    logger.info(f"Command extracted immediately: {command}")
                    add_event("voice_input", {"text": command})
                    process_command(command, speaker, chatbot)
                else:
                    speaker.speak("Yes, I am listening.")
                    add_event("status", {"mic_status": "active_listening", "info": "Listening for command..."})
                    command = listener.listen(timeout=config.COMMAND_TIMEOUT)
                    if command:
                        logger.info(f"Command received: {command}")
                        add_event("voice_input", {"text": command})
                        process_command(command, speaker, chatbot)
                    else:
                        speaker.speak("I didn't hear a command.")
                        add_event("voice_alert", {"info": "Command timeout", "text": ""})
        except Exception as e:
            logger.error(f"Error in voice listener thread: {e}")
            add_event("error", {"info": f"Microphone error: {str(e)}"})
            time.sleep(1)


def video_capture_worker():
    """Background thread for camera frame capture, face detection/recognition, expression detection, and greeting logic."""
    global latest_frame_bytes, system_unlocked, last_seen_target_user, latest_mood, latest_action, recording_action, recording_samples_count
    logger.info("Video capture thread started.")
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    temporal_filter = TemporalFilter(size=20)
    action_detector = ActionDetector()
    
    last_greeted = {}
    last_greeted_mood = {}
    pending_greetings = {}
    last_logged_prediction = {}
    greeting_cooldown = config.GREETING_COOLDOWN
    last_expression_run = 0.0
    cached_mood = "Neutral"
    cached_action = "Coding/Typing"
    last_announced_action = "Coding/Typing"
    last_action_announcement_time = {}
    last_broadcasted_action = "Coding/Typing"
    
    active_index = config.CAMERA_INDEX
    backend = cv2.CAP_DSHOW if os.name == 'nt' else None
    
    def get_capture(index):
        if backend is not None:
            return cv2.VideoCapture(index, backend)
        return cv2.VideoCapture(index)

    while True:
        cap = None
        # Try preferred index first
        temp_cap = get_capture(active_index)
        if temp_cap.isOpened():
            ret, _ = temp_cap.read()
            if ret:
                cap = temp_cap
            else:
                temp_cap.release()
                
        if cap is None:
            # Auto-detect other indexes
            for idx in range(6):
                if idx == active_index:
                    continue
                temp_cap = get_capture(idx)
                if temp_cap.isOpened():
                    ret, _ = temp_cap.read()
                    if ret:
                        logger.info(f"Auto-detected active camera at index: {idx}")
                        cap = temp_cap
                        active_index = idx
                        break
                    temp_cap.release()
                    
        if cap is None:
            logger.warning("Could not open any camera stream! Retrying in 5 seconds...")
            add_event("error", {"info": "Webcam not available. Retrying..."})
            if not system_unlocked:
                logger.info("Unlocking voice module for voice-only fallback mode.")
                system_unlocked = True
                add_event("status", {"mic_status": "listening", "info": "System Unlocked (Voice-only mode)"})
            time.sleep(5)
            continue
            
        logger.info(f"Camera stream opened successfully on index {active_index}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH if hasattr(cv2, 'CAP_PROP_FRAME_WIDTH') else 3, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT if hasattr(cv2, 'CAP_PROP_FRAME_HEIGHT') else 4, config.FRAME_HEIGHT)

        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera. Disconnected.")
                time.sleep(0.1)
                break
                
            gray_frame, faces = detector.detect_faces(frame)
            
            # Run action detection
            cached_action = action_detector.detect(frame, gray_frame, faces)
            
            # If a training session is active, collect visual samples
            global recording_action, recording_samples_count
            if recording_action is not None:
                feat = action_detector.get_feature_vector()
                if feat:
                    action_detector.save_sample(recording_action, feat)
                    recording_samples_count += 1
                    add_event("training_progress", {
                        "action": recording_action,
                        "count": recording_samples_count,
                        "total": 150
                    })
                    if recording_samples_count >= 150:
                        logger.info(f"Web training complete for '{recording_action}'")
                        speaker.speak(f"Action training complete for {recording_action}.")
                        add_event("training_complete", {"action": recording_action})
                        recording_action = None
            
            # Apply single person detection (keep only the largest face)
            if getattr(config, "SINGLE_PERSON_ONLY", False) and len(faces) > 0:
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                faces = [largest_face]
                
            detected_names = []
            now = time.time()
            if len(faces) > 0:
                for face in faces:
                    x, y, w, h = face
                    
                    # Clip face box coordinates to frame boundaries to prevent empty crops
                    img_h, img_w = gray_frame.shape[:2]
                    x_start = max(0, x)
                    y_start = max(0, y)
                    x_end = min(img_w, x + w)
                    y_end = min(img_h, y + h)
                    
                    face_crop = gray_frame[y_start:y_end, x_start:x_end]
                    if face_crop.size == 0:
                        continue
                    
                    # Run identity prediction and expression detection at 0.15-second intervals to optimize CPU usage
                    if now - last_expression_run >= 0.15:
                        raw_name, conf = recognizer.predict(frame, face)
                        raw_mood = detector.detect_expression(face_crop)
                        cached_name = raw_name
                        cached_conf = conf
                        cached_mood = raw_mood
                        last_expression_run = now
                    else:
                        raw_name = cached_name if 'cached_name' in locals() or 'cached_name' in globals() else "Unknown"
                        conf = cached_conf if 'cached_conf' in locals() or 'cached_conf' in globals() else 0.0
                        raw_mood = cached_mood
                    
                    # Smooth identity and expression
                    temporal_filter.update(raw_name, raw_mood)
                    name, mood = temporal_filter.get_smoothed()
                    
                    latest_mood = mood
                    latest_action = cached_action
                    
                    # Log prediction strictly every 1.0 second to satisfy the "each second time" requirement
                    log_key = f"predict_{name}"
                    if log_key not in last_logged_prediction or (now - last_logged_prediction[log_key] >= 1.0):
                        logger.info(f"Face predicted: '{name}' (Raw: '{raw_name}', Conf: {conf:.2f}, Mood: {mood}, Action: {cached_action})")
                        last_logged_prediction[log_key] = now
                        # Broadcast face detection event (including mood) every second
                        add_event("face_detected", {"name": name, "confidence": conf, "mood": mood})
                    
                    # Manage security lock state for target user
                    target_user = getattr(config, "TARGET_USER", None)
                    if target_user and name == target_user:
                        last_seen_target_user = now
                        if not system_unlocked:
                            logger.info(f"[SECURITY] Authorized user '{name}' detected. Unlocking system.")
                            system_unlocked = True
                            add_event("status", {"mic_status": "listening", "info": "System Unlocked (Microphone ready)"})
                            # Keep cooldown across momentary lock/unlock cycles
                            pass
                    
                    if name != "Unknown":
                        label_text = f"{name} ({mood}) [{cached_action}] [Conf: {int(conf * 100)}]"
                        color = (0, 255, 0)
                    else:
                        label_text = f"Unknown ({mood}) [{cached_action}]"
                        color = (0, 0, 255)
                    detected_names.append((name, mood, conf))
                    
                    # Draw overlay on frame
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # Decay the filter if no faces are seen for 3 seconds
                if now - temporal_filter.last_seen > 3.0:
                    temporal_filter.clear()
                latest_mood = "Neutral"
                latest_action = "Neutral"
                cached_action = "Neutral"

            # Broadcast action changes immediately
            if cached_action != last_broadcasted_action:
                add_event("action_detected", {"action": cached_action})
                last_broadcasted_action = cached_action

            # Proactive transition announcements
            if cached_action in ["Eating", "Singing", "Walking"] and cached_action != last_announced_action:
                time_since = now - last_action_announcement_time.get(cached_action, 0.0)
                if time_since > 90.0 and not speaker.is_speaking:
                    logger.info(f"Proactive action speech triggered: '{cached_action}'")
                    last_announced_action = cached_action
                    last_action_announcement_time[cached_action] = now
                    
                    templates = {
                        "Eating": [
                            "I see you are eating, Janvi. Enjoy your food and stay healthy!",
                            "A good meal is fuel for great code. Enjoy your meal, Janvi Shah."
                        ],
                        "Singing": [
                            "You are singing beautifully, Janvi Shah! Music is good for the soul.",
                            "I love hearing you sing, Janvi. That sounds lovely!"
                        ],
                        "Walking": [
                            "I see you are walking, Janvi. Let's keep moving forward!",
                            "Walking is great for the mind, Janvi Shah. Where are we heading?"
                        ]
                    }
                    import random
                    speech_text = random.choice(templates[cached_action])
                    speaker.speak(speech_text)
                    add_event("greeting", {"text": speech_text})
            elif cached_action in ["Coding/Typing", "Neutral"]:
                last_announced_action = cached_action

            # We no longer lock the system when target user is not seen. Mic remains active.

            # Handle greetings
            now = time.time()
            for name, mood, conf in detected_names:
                # Greet any detected user (including Unknown/guests)
                time_since_last = (now - last_greeted[name]) if name in last_greeted else float('inf')
                is_new_mood = (name in last_greeted_mood) and (last_greeted_mood[name] != mood)
                
                mood_change_allowed = is_new_mood and (time_since_last > 60.0)
                standard_allowed = time_since_last > greeting_cooldown
                
                if standard_allowed or mood_change_allowed:
                    if name not in pending_greetings:
                        logger.info(f"Queued greeting for '{name}' with mood '{mood}' (standard_allowed={standard_allowed}, mood_change_allowed={mood_change_allowed})")
                        pending_greetings[name] = (now, mood, conf)
                        add_event("face_detected", {"name": name, "confidence": conf, "mood": mood})
                    elif pending_greetings[name][1] != mood:
                        logger.info(f"Updated pending greeting mood for '{name}' to '{mood}' (preserving original queue time)")
                        first_queued_time = pending_greetings[name][0]
                        pending_greetings[name] = (first_queued_time, mood, conf)
                        add_event("face_detected", {"name": name, "confidence": conf, "mood": mood})

            if pending_greetings:
                last_added_time = max(val[0] for val in pending_greetings.values())
                if now - last_added_time >= getattr(config, "GREETING_STABILIZATION_TIME", 1.0) and not speaker.is_speaking:
                    names_to_greet = list(pending_greetings.keys())
                    
                    if len(names_to_greet) == 1:
                        n = names_to_greet[0]
                        _, m, c = pending_greetings[n]
                        mood_clean = m.strip().lower()
                        
                        # 1. Target user specific profound expression greetings
                        if n.lower() == getattr(config, "TARGET_USER", "janvi shah").lower():
                            greetings = getattr(config, "EXPRESSION_GREETINGS", {})
                            if "happy" in mood_clean:
                                greeting_text = greetings.get("happy", "A smile looks great on you!")
                            elif "surprised" in mood_clean:
                                greeting_text = greetings.get("surprised", "What a pleasant surprise!")
                            elif "focused" in mood_clean or "angry" in mood_clean:
                                greeting_text = greetings.get("angry", "Take a breath, how can I help?")
                            elif "sad" in mood_clean or "tired" in mood_clean:
                                greeting_text = greetings.get("sad", "I'm here for you if you need anything.")
                            else:
                                greeting_text = greetings.get("neutral", "Hello Janvi Shah!")
                                
                        # 2. Guest/Unknown user expression greetings
                        elif n == "Unknown":
                            greetings = getattr(config, "GUEST_EXPRESSION_GREETINGS", {})
                            if "happy" in mood_clean:
                                greeting_text = greetings.get("happy", "Hello guest, you look happy today!")
                            elif "surprised" in mood_clean:
                                greeting_text = greetings.get("surprised", "Hello guest, you look surprised!")
                            elif "focused" in mood_clean or "angry" in mood_clean:
                                greeting_text = greetings.get("angry", "Hello guest, you look focused or angry. Everything okay?")
                            elif "sad" in mood_clean or "tired" in mood_clean:
                                greeting_text = greetings.get("sad", "Hello guest, you look a bit sad or tired today.")
                            else:
                                greeting_text = greetings.get("neutral", "Hello! Welcome.")
                                
                        # 3. Other recognized users expression greetings
                        else:
                            greetings = getattr(config, "GENERAL_EXPRESSION_GREETINGS", {})
                            if "happy" in mood_clean:
                                template = greetings.get("happy", "A smile looks great on you, {name}!")
                            elif "surprised" in mood_clean:
                                template = greetings.get("surprised", "What a pleasant surprise, {name}!")
                            elif "focused" in mood_clean or "angry" in mood_clean:
                                template = greetings.get("angry", "Take a breath, {name}. How can I help?")
                            elif "sad" in mood_clean or "tired" in mood_clean:
                                template = greetings.get("sad", "I'm here for you, {name}.")
                            else:
                                template = greetings.get("neutral", "Hello {name}!")
                            greeting_text = template.format(name=n)
                    elif len(names_to_greet) == 2:
                        n1, n2 = names_to_greet
                        friendly_n1 = "guest" if n1 == "Unknown" else n1
                        friendly_n2 = "guest" if n2 == "Unknown" else n2
                        greeting_text = f"Hello {friendly_n1} and {friendly_n2}, nice to see you."
                    else:
                        friendly_names = ["guest" if name == "Unknown" else name for name in names_to_greet]
                        names_str = ", ".join(friendly_names[:-1]) + ", and " + friendly_names[-1]
                        greeting_text = f"Hello {names_str}, nice to see you."
                    
                    for name in names_to_greet:
                        last_greeted[name] = now
                        last_greeted_mood[name] = pending_greetings[name][1]
                        
                    pending_greetings.clear()
                    
                    logger.info(f"Speaking greeting: '{greeting_text}'")
                    speaker.speak(greeting_text)
                    add_event("greeting", {"text": greeting_text})
                    
            # Encode frame
            ret_enc, jpeg = cv2.imencode('.jpg', frame)
            if ret_enc:
                with frame_lock:
                    latest_frame_bytes = jpeg.tobytes()
                    
            time.sleep(0.03)
            
        cap.release()
        with frame_lock:
            latest_frame_bytes = None
        time.sleep(1)

# --- FLASK ENDPOINTS ---

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

def gen_frames():
    global latest_frame_bytes
    while True:
        with frame_lock:
            frame = latest_frame_bytes
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/events')
def events():
    """Server-Sent Events stream for real-time frontend updates."""
    q = queue.Queue()
    with client_lock:
        client_queues.append(q)
        
    def event_stream():
        try:
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connect', 'info': 'SSE Stream Active'})}\n\n"
            while True:
                try:
                    event = q.get(timeout=1.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    yield "data: {\"type\": \"ping\"}\n\n"
        finally:
            with client_lock:
                client_queues.remove(q)
                
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/api/train', methods=['POST'])
def train_action_api():
    """Triggers visual action training for a specific action name."""
    global recording_action, recording_samples_count
    data = request.json or {}
    action = data.get("action", "").strip()
    
    if not action:
        return jsonify({"status": "error", "message": "Missing action name"}), 400
        
    with recording_lock:
        if recording_action is not None:
            return jsonify({"status": "error", "message": f"Already training action: {recording_action}"}), 400
        recording_action = action
        recording_samples_count = 0
        
    logger.info(f"Triggered action training session via API for: '{action}'")
    add_event("status", {"mic_status": "idle", "info": f"Recording samples for '{action}'..."})
    
    # Speak countdown to let the user get ready
    speaker.speak(f"Please perform the action {action} in front of the camera. Recording starts now.")
    
    return jsonify({"status": "ok"})

@app.route('/api/users', methods=['GET'])
def list_users():
    """Lists all registered face folders (users)."""
    recognizer = FaceRecognizer()
    users = list(recognizer.label_to_name.values())
    return jsonify({"users": users})

@app.route('/api/greet', methods=['POST'])
def trigger_greet():
    """Manually commands the speaker to speak a greeting for a specific name."""
    data = request.json or {}
    name = data.get("name", "User").strip()
    if name.lower() == getattr(config, "TARGET_USER", "janvi shah").lower():
        greeting_text = "Welcome back, Janvi Shah! It is an absolute honor to assist you today. I hope you are having a productive day."
    else:
        greeting_text = f"Hello {name}, nice to see you."
    speaker.speak(greeting_text)
    add_event("greeting", {"text": greeting_text})
    return jsonify({"status": "ok"})

@app.route('/api/speak', methods=['POST'])
def trigger_speak():
    """Instructs the speaker to speak custom text."""
    data = request.json or {}
    text = data.get("text", "").strip()
    if text:
        speaker.speak(text)
        add_event("response", {"text": text})
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "message": "Empty text"}), 400

@app.route('/api/command', methods=['POST'])
def run_command_api():
    """Allows typing text commands from the web dashboard."""
    data = request.json or {}
    command = data.get("command", "").strip()
    if command:
        add_event("voice_input", {"text": f"[Web UI] {command}"})
        # Process command async to avoid blocking HTTP response
        threading.Thread(target=process_command, args=(command, speaker, chatbot)).start()
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "message": "Empty command"}), 400

@app.route('/api/status', methods=['GET'])
def get_status():
    """Retrieves current robot config statuses."""
    return jsonify({
        "camera_active": latest_frame_bytes is not None,
        "chatbot_enabled": chatbot.enabled,
        "single_person_mode": getattr(config, "SINGLE_PERSON_ONLY", True),
        "target_user": getattr(config, "TARGET_USER", "None")
    })

def main():
    # Start Video capture background thread
    video_thread = threading.Thread(target=video_capture_worker, daemon=True)
    video_thread.start()
    
    # Start Voice listener background thread
    listener = VoiceListener()
    voice_thread = threading.Thread(target=voice_listener_worker, args=(listener, speaker, chatbot), daemon=True)
    voice_thread.start()
    
    # Greet on boot
    speaker.speak("Web control server starting up.")
    
    # Run server locally (port 5000)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()
