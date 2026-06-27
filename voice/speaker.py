import pyttsx3
import logging
import os
import tempfile
import config
from queue import Queue
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoiceSpeaker")

class VoiceSpeaker:
    def __init__(self):
        self.use_online = config.USE_ONLINE_TTS
        self.offline_engine = None

        # Thread-safe queue to process speech sequentially and avoid collision
        self.speech_queue = Queue()
        self.is_speaking = False
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _speech_worker(self):
        """Worker thread to process the speech queue sequentially."""
        if not self.use_online:
            try:
                # Initialize pyttsx3 engine on the worker thread itself to prevent SAPI5 COM multi-threading errors
                import pyttsx3
                self.offline_engine = pyttsx3.init()
                self.offline_engine.setProperty('rate', config.VOICE_RATE)
                self.offline_engine.setProperty('volume', config.VOICE_VOLUME)
                
                # Try setting a female or more natural voice if available
                voices = self.offline_engine.getProperty('voices')
                if len(voices) > 1:
                    self.offline_engine.setProperty('voice', voices[1].id)
                logger.info("Offline TTS engine (pyttsx3) initialized in worker thread.")
            except Exception as e:
                logger.warning(f"Failed to initialize offline pyttsx3 engine in worker thread: {e}")
                logger.warning("Switching to online gTTS engine (requires internet).")
                self.use_online = True

        while True:
            text = self.speech_queue.get()
            if text is None:
                break
            
            self.is_speaking = True
            logger.info(f"Robot says: '{text}'")
            if self.use_online:
                self._speak_online(text)
            else:
                self._speak_offline(text)
            self.is_speaking = False
            self.speech_queue.task_done()

    def speak(self, text):
        """Queues the provided text to be spoken by the background worker."""
        if not text:
            return
        self.speech_queue.put(text)

    def _speak_offline(self, text):
        try:
            self.offline_engine.say(text)
            self.offline_engine.runAndWait()
        except Exception as e:
            logger.error(f"Offline speech error: {e}")
            # Try to reinitialize
            try:
                self.offline_engine = pyttsx3.init()
                self.offline_engine.say(text)
                self.offline_engine.runAndWait()
            except Exception as re_err:
                logger.error(f"Critical failure speaking offline: {re_err}")

    def _speak_online(self, text):
        try:
            from gtts import gTTS
            import re
            
            # Detect Hindi characters to set language code to 'hi' for natural Indian accent
            lang = 'hi' if re.search(r'[\u0900-\u097f]', text) else 'en'
            logger.info(f"Generating TTS online in '{lang}' language mode.")
            
            tts = gTTS(text=text, lang=lang)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_filename = fp.name
                tts.save(temp_filename)

            # Play the sound
            if os.name == 'nt':  # Windows
                import ctypes
                # Get the short path name to avoid issues with spaces or special characters in the filepath
                def get_short_path_name(long_name):
                    buf = ctypes.create_unicode_buffer(1024)
                    ctypes.windll.kernel32.GetShortPathNameW(long_name, buf, 1024)
                    return buf.value
                
                short_path = get_short_path_name(temp_filename)
                winmm = ctypes.windll.winmm
                
                # Stop and close any previous alias to avoid conflict
                winmm.mciSendStringW("close tts_sound", None, 0, 0)
                open_cmd = f'open "{short_path}" type mpegvideo alias tts_sound'
                if winmm.mciSendStringW(open_cmd, None, 0, 0) == 0:
                    winmm.mciSendStringW('play tts_sound wait', None, 0, 0)
                    winmm.mciSendStringW('close tts_sound', None, 0, 0)
            else:  # Linux / Raspberry Pi
                os.system(f'mpg123 -q "{temp_filename}" > /dev/null 2>&1')
                import time
                time.sleep(1.0)

            try:
                os.remove(temp_filename)
            except Exception:
                pass  # Ignore file lock errors on cleanup
        except Exception as e:
            logger.error(f"Online TTS error: {e}")
            print(f"[Robot voice output failed: {text}]")
            
    def shutdown(self):
        """Clean up offline engine loop."""
        self.speech_queue.put(None)
        if self.offline_engine:
            try:
                self.offline_engine.stop()
            except Exception:
                pass
