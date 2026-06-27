import os
import sys

# Suppress OpenCV videoio warnings and verbose logging
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_LOG_LEVEL"] = "0"

import cv2
import argparse
import time
import threading
import logging
import queue

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
logger = logging.getLogger("RobotBrain")

# Shared command queue to pass voice commands to the main loop thread
command_queue = queue.Queue()
system_unlocked = not getattr(config, "TARGET_USER", None)

def voice_listener_thread(listener, speaker):
    """
    Background thread that continuously listens for wake words and handles command input.
    """
    global system_unlocked
    logger.info("Voice listener thread started.")
    listener.calibrate()
    
    # Track locking log state to avoid console clutter
    last_lock_log = 0
    
    while True:
        try:
            # Always active: mic is never locked. We no longer pause voice listener thread here.
                
            # Step 1: Listen for voice input
            text = listener.listen(timeout=None)
            if not text:
                continue
            
            # Step 2: Check for wake word and see if a command is already present
            wake_detected, command = listener.check_and_extract_command(text)
            if wake_detected:
                logger.info("Wake word detected!")
                if command:
                    logger.info(f"Command extracted immediately: {command}")
                    command_queue.put(command)
                else:
                    speaker.speak("Yes, I am listening.")
                    
                    # Step 3: Listen specifically for the command following the wake word
                    command = listener.listen(timeout=config.COMMAND_TIMEOUT)
                    if command:
                        logger.info(f"Command received: {command}")
                        command_queue.put(command)
                    else:
                        speaker.speak("I didn't hear a command.")
        except Exception as e:
            logger.error(f"Error in voice listener thread: {e}")
            time.sleep(1)

def register_new_user(name, detector):
    """
    Captures 50 face images of a user, crops and resizes them, and saves them for training.
    """
    logger.info(f"Preparing to register user: '{name}'")
    user_dir = os.path.join(config.FACES_DIR, name)
    os.makedirs(user_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Could not open camera for registration!")
        return False
        
    cap.set(cv3_width_prop() if hasattr(cv2, 'CAP_PROP_FRAME_WIDTH') else 3, config.FRAME_WIDTH)
    cap.set(cv3_height_prop() if hasattr(cv2, 'CAP_PROP_FRAME_HEIGHT') else 4, config.FRAME_HEIGHT)
    
    speaker = VoiceSpeaker()
    speaker.speak(f"Please look directly at the camera to register your face, {name}.")
    
    logger.info("Starting capture. Look at the camera...")
    count = 0
    start_time = time.time()
    
    while count < 50:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame.")
            continue
            
        gray_frame, faces = detector.detect_faces(frame)
        
        # Display guidelines on the screen
        preview_frame = frame.copy()
        cv2.putText(preview_frame, f"Capturing: {count}/50. Look at camera.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        detector.draw_faces(preview_frame, faces)
        cv2.imshow("Registering Face - Keep Looking at Camera", preview_frame)
        
        # If we detect exactly one face, crop and save it
        if len(faces) == 1:
            x, y, w, h = faces[0]
            # Crop face region
            face_img = gray_frame[y:y+h, x:x+w]
            # Resize face to a standard size for LBPH consistency
            face_resized = cv2.resize(face_img, (200, 200))
            
            # Save image
            img_path = os.path.join(user_dir, f"face_{count}.jpg")
            cv2.imwrite(img_path, face_resized)
            count += 1
            time.sleep(0.05)  # Add a tiny delay to get slightly different angles
            
        # Stop if 'q' or Esc is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            logger.info("Registration cancelled by user.")
            break
            
        # Timeout safety (2 minutes max)
        if time.time() - start_time > 120:
            logger.warning("Registration timed out.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if count >= 50:
        speaker.speak(f"Capture complete. Training model now.")
        logger.info(f"Successfully captured {count} samples. Retraining face model...")
        recognizer = FaceRecognizer()
        if recognizer.train_model():
            speaker.speak("Face model successfully trained. I can now recognize you.")
            logger.info("Training complete!")
            return True
    else:
        speaker.speak("Failed to capture enough face samples. Registration incomplete.")
        logger.warning(f"Only captured {count}/50 face samples.")
        return False

def cv3_width_prop():
    return cv2.CAP_PROP_FRAME_WIDTH if hasattr(cv2, 'CAP_PROP_FRAME_WIDTH') else 3

def cv3_height_prop():
    return cv2.CAP_PROP_FRAME_HEIGHT if hasattr(cv2, 'CAP_PROP_FRAME_HEIGHT') else 4

def process_command(command, speaker, chatbot, frame_bytes=None, mood="Neutral", action="Coding/Typing"):
    """Core command processing logic."""
    import re
    logger.info(f"Processing command: '{command}'")
    response_text = ""
    command_clean = command.strip().lower()
    
    # 1. Check Chatbot (Gemini / Local Q&A Grounded) FIRST
    chatbot_response = chatbot.get_response(command, image_bytes=frame_bytes, mood=mood, action=action)
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
            response_text = chatbot_response

    if response_text:
        speaker.speak(response_text)


def record_action_samples(action_name):
    """
    Guides the user and records 150 feature vectors for the specified action name.
    """
    logger.info(f"Preparing to train physical action: '{action_name}'")
    detector = FaceDetector()
    action_detector = ActionDetector()
    
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Could not open camera for action training!")
        return False
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH if hasattr(cv2, 'CAP_PROP_FRAME_WIDTH') else 3, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT if hasattr(cv2, 'CAP_PROP_FRAME_HEIGHT') else 4, config.FRAME_HEIGHT)
    
    speaker = VoiceSpeaker()
    speaker.speak(f"Please perform the action {action_name} in front of the camera. Recording starts in 3 seconds.")
    logger.info("Recording starts in 3 seconds. Get ready...")
    time.sleep(3)
    
    speaker.speak("Recording now. Keep performing the action.")
    logger.info("Recording samples... Keep performing the action!")
    
    count = 0
    start_time = time.time()
    
    while count < 150:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame.")
            continue
            
        gray_frame, faces = detector.detect_faces(frame)
        
        # Run detector's internal tracking
        action_detector.detect(frame, gray_frame, faces)
        
        preview_frame = frame.copy()
        cv2.putText(preview_frame, f"Recording {action_name}: {count}/150", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        detector.draw_faces(preview_frame, faces)
        cv2.imshow(f"Action Recording - {action_name}", preview_frame)
        
        feat = action_detector.get_feature_vector()
        if feat:
            action_detector.save_sample(action_name, feat)
            count += 1
            
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            logger.info("Recording cancelled by user.")
            break
            
        # Timeout safety (30 seconds)
        if time.time() - start_time > 30:
            logger.warning("Recording timed out.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if count >= 150:
        speaker.speak(f"Action training complete for {action_name}.")
        logger.info(f"Successfully recorded {count} samples for '{action_name}'!")
        return True
    else:
        speaker.speak("Action training incomplete.")
        logger.warning(f"Only recorded {count}/150 samples.")
        return False


def main():
    global system_unlocked
    parser = argparse.ArgumentParser(description="Raspberry Pi Robot Assistant - Phase 1")
    parser.add_argument("--register", type=str, help="Name of the user to register face samples for.")
    parser.add_argument("--train", action="store_true", help="Force train the face recognition model.")
    parser.add_argument("--no-video", action="store_true", help="Run in headless mode without video display.")
    parser.add_argument("--list", action="store_true", help="List all registered/known users.")
    parser.add_argument("--record-action", type=str, help="Name of the physical action to train (e.g. eating, singing, walking, coding).")
    args = parser.parse_args()

    # Handle List Users mode
    if args.list:
        recognizer = FaceRecognizer()
        if recognizer.label_to_name:
            print("Registered users:")
            for label, name in recognizer.label_to_name.items():
                print(f" - {name} (ID: {label})")
        else:
            print("No registered users found.")
        sys.exit(0)

    # Handle Action Recording mode
    if args.record_action:
        record_action_samples(args.record_action)
        sys.exit(0)

    detector = FaceDetector()

    # Handle Face Registration mode
    if args.register:
        register_new_user(args.register, detector)
        sys.exit(0)

    # Handle Training mode
    if args.train:
        recognizer = FaceRecognizer()
        recognizer.train_model()
        sys.exit(0)

    # Main running mode
    logger.info("Initializing Robot Assistant Main Runner...")
    
    recognizer = FaceRecognizer()
    speaker = VoiceSpeaker()
    listener = VoiceListener()
    chatbot = Chatbot()

    # Greet user on boot
    speaker.speak("Robot system is starting up.")

    # Start background Voice Listener Thread
    voice_thread = threading.Thread(
        target=voice_listener_thread, 
        args=(listener, speaker),
        daemon=True
    )
    voice_thread.start()

    # Open Camera Stream
    cap = None
    active_index = config.CAMERA_INDEX
    backend = cv2.CAP_DSHOW if os.name == 'nt' else None
    
    def get_capture(index):
        if backend is not None:
            return cv2.VideoCapture(index, backend)
        return cv2.VideoCapture(index)

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
        logger.error("Could not open any camera stream!")
        speaker.speak("Warning. I cannot access the camera. Running in voice-only mode.")
        logger.info("Camera not available, but continuing in voice-only mode.")
        system_unlocked = True
        cap = cv2.VideoCapture() # Dummy unopened cap
        
    if cap.isOpened():
        logger.info(f"Camera stream opened successfully on index {active_index}.")
        cap.set(cv3_width_prop(), config.FRAME_WIDTH)
        cap.set(cv3_height_prop(), config.FRAME_HEIGHT)


    logger.info("System is ready. Press Ctrl+C or 'q' in the window to quit.")
    
    # Cooldown mapping to track when we last greeted a recognized user
    # Key: username, Value: timestamp of last greeting
    last_greeted = {}
    last_greeted_mood = {}
    pending_greetings = {}
    last_logged_prediction = {}
    last_seen_target_user = 0.0
    temporal_filter = TemporalFilter(size=20)
    greeting_cooldown = getattr(config, "GREETING_COOLDOWN", 20.0)  # seconds
    last_expression_run = 0.0
    cached_mood = "Neutral"
    cached_action = "Coding/Typing"
    last_announced_action = "Coding/Typing"
    last_action_announcement_time = {}
    action_detector = ActionDetector()

    try:
        while True:
            # 1. Process any incoming voice commands from the background thread
            try:
                # Non-blocking check of the queue
                command = command_queue.get_nowait()
                
                frame_bytes = None
                if cap.isOpened() and 'frame' in locals() and frame is not None:
                    ret_enc, jpeg = cv2.imencode('.jpg', frame)
                    if ret_enc:
                        frame_bytes = jpeg.tobytes()
                        
                threading.Thread(
                    target=process_command, 
                    args=(command, speaker, chatbot, frame_bytes, cached_mood, cached_action), 
                    daemon=True
                ).start()
                command_queue.task_done()
            except queue.Empty:
                pass

            # 2. Vision Processing (if camera is open)
            if cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to grab camera frame.")
                    time.sleep(0.05)
                    continue

                # Run detection
                gray_frame, faces = detector.detect_faces(frame)
                
                # Run action detection
                cached_action = action_detector.detect(frame, gray_frame, faces)
                
                # If single person detection is enabled, keep only the largest face (closest to camera)
                if getattr(config, "SINGLE_PERSON_ONLY", False) and len(faces) > 0:
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    faces = [largest_face]
                
                # Predict identities and detect expressions
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
                        face_resized = cv2.resize(face_crop, (200, 200))
                        
                        # Run identity prediction and expression detection at 0.15-second intervals to optimize CPU usage
                        if now - last_expression_run >= 0.15:
                            raw_name, conf = recognizer.predict(frame, face)
                            raw_mood = detector.detect_expression(face_resized)
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
                        
                        # Log prediction strictly every 1.0 second to satisfy the "each second time" requirement
                        log_key = f"predict_{name}"
                        if log_key not in last_logged_prediction or (now - last_logged_prediction[log_key] >= 1.0):
                            logger.info(f"Face predicted: '{name}' (Raw: '{raw_name}', Conf: {conf:.2f}, Mood: {mood})")
                            last_logged_prediction[log_key] = now
                        
                        # Manage security lock state for target user
                        target_user = getattr(config, "TARGET_USER", None)
                        if target_user and name == target_user:
                            last_seen_target_user = now
                            if not system_unlocked:
                                logger.info(f"[SECURITY] Authorized user '{name}' detected. Unlocking system.")
                                system_unlocked = True
                                # Keep cooldown across momentary lock/unlock cycles
                                pass
                        
                        # If recognized or unknown, manage greeting cooldowns
                        if name != "Unknown":
                            label_text = f"{name} ({mood}) [{cached_action}] [Conf: {int(conf * 100)}]"
                            color = (0, 255, 0) # Green for match
                        else:
                            label_text = f"Unknown ({mood}) [{cached_action}]"
                            color = (0, 0, 255) # Red for unknown
                        detected_names.append((name, mood, conf))
                        
                        # Draw visual boxes and labels on frame
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    if len(faces) == 0:
                        if now - temporal_filter.last_seen > 3.0:
                            temporal_filter.clear()
                        cached_action = "Neutral"

                # Proactive transition announcements
                now = time.time()
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
                elif cached_action in ["Coding/Typing", "Neutral"]:
                    last_announced_action = cached_action

                # We no longer lock the system when target user is not seen. Mic remains active.

                # Queue greetings for newly seen people
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
                        elif pending_greetings[name][1] != mood:
                            # Update the mood, but keep the original queue time so we don't delay the greeting indefinitely
                            first_queued_time = pending_greetings[name][0]
                            logger.info(f"Updated pending greeting mood for '{name}' to '{mood}' (preserving original queue time)")
                            pending_greetings[name] = (first_queued_time, mood, conf)
                    else:
                        # Log cooldown details occasionally (throttled)
                        if f"cooldown_{name}" not in last_logged_prediction or (now - last_logged_prediction[f"cooldown_{name}"] > 5.0):
                            logger.info(f"Greeting for '{name}' is on cooldown. Remaining: {greeting_cooldown - time_since_last:.1f}s")
                            last_logged_prediction[f"cooldown_{name}"] = now

                # If there are pending greetings, wait for them to stabilize
                if pending_greetings:
                    # Check if stabilization time has passed since the last person was added
                    last_added_time = max(val[0] for val in pending_greetings.values())
                    time_diff = now - last_added_time
                    # Log stabilization status occasionally
                    if "stabilization_log" not in last_logged_prediction or (now - last_logged_prediction["stabilization_log"] > 1.0):
                        logger.info(f"Pending greetings: {list(pending_greetings.keys())}, Stabilization: {time_diff:.2f}s / {getattr(config, 'GREETING_STABILIZATION_TIME', 2.0)}s")
                        last_logged_prediction["stabilization_log"] = now

                    if time_diff >= getattr(config, "GREETING_STABILIZATION_TIME", 2.0) and not speaker.is_speaking:
                        # Time to greet everyone in the queue!
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
                        
                        logger.info(f"Triggering unified greeting: '{greeting_text}'")
                        # Register the greeting time
                        for name in names_to_greet:
                            last_greeted[name] = now
                            last_greeted_mood[name] = pending_greetings[name][1]
                            
                        # Clear the pending queue
                        pending_greetings.clear()
                        
                        # Speak the unified greeting
                        speaker.speak(greeting_text)

                # Show Video Feed (if not headless)
                if not args.no_video:
                    cv2.imshow("Robot Vision - Press 'q' to Quit", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):
                        logger.info("Quitting by user keyboard request.")
                        break

            # Small sleep to prevent 100% CPU core usage on the Pi
            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("System interrupted by user.")
    finally:
        # Cleanup
        logger.info("Shutting down robot modules...")
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        speaker.shutdown()
        logger.info("Shutdown complete. Goodbye!")

if __name__ == "__main__":
    main()
