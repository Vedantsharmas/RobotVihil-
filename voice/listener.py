import speech_recognition as sr
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoiceListener")

class VoiceListener:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Set dynamic energy threshold parameters for better voice isolation
        self.recognizer.dynamic_energy_threshold = True
        
        try:
            self.microphone = sr.Microphone()
            logger.info("Microphone initialized successfully.")
        except Exception as e:
            logger.error(f"Error accessing microphone: {e}")
            logger.error("Verify that a USB microphone or audio card is connected to the Raspberry Pi.")
            self.microphone = None

    def calibrate(self, duration=2):
        """Calibrates the recognizer threshold to account for ambient room noise."""
        if not self.microphone:
            logger.error("Cannot calibrate: Microphone not available.")
            return False
            
        logger.info(f"Calibrating microphone for {duration} seconds... Please keep quiet.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)
        logger.info(f"Calibration finished. Energy threshold set to: {self.recognizer.energy_threshold:.2f}")
        return True

    def listen(self, timeout=None):
        """
        Listens to the microphone and transcribes speech to text.
        Returns:
            - text: Transcribed text string (lowercase), or None if not recognized
        """
        if not self.microphone:
            logger.error("Cannot listen: Microphone not available.")
            return None

        with self.microphone as source:
            logger.info("Listening for command...")
            try:
                # listen(source, timeout, phrase_time_limit)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                logger.debug("Listening timed out waiting for speech.")
                return None
            except Exception as e:
                logger.error(f"Error while listening: {e}")
                return None

        try:
            logger.info("Processing speech...")
            
            # 1. Attempt Groq Whisper API for ultra-fast, high-quality transcription (supports bilingual/Hinglish)
            groq_api_key = getattr(config, "GROQ_API_KEY", "")
            if groq_api_key:
                try:
                    import requests
                    url = "https://api.groq.com/openai/v1/audio/transcriptions"
                    headers = {
                        "Authorization": f"Bearer {groq_api_key}"
                    }
                    files = {
                        "file": ("speech.wav", audio.get_wav_data(), "audio/wav")
                    }
                    data = {
                        "model": "whisper-large-v3",
                        "response_format": "json",
                        "prompt": "Janvi Shah, assistant robot commands like: hello, stop, bring me, locate, find, patrol, main, Hinglish, Hindi"
                    }
                    logger.info("Transcribing with Groq Whisper-large-v3...")
                    res = requests.post(url, headers=headers, files=files, data=data, timeout=8)
                    if res.status_code == 200:
                        text = res.json().get("text", "").strip()
                        if text:
                            text_lower = text.lower().strip()
                            logger.info(f"Heard (Groq Whisper): '{text_lower}'")
                            return text_lower
                    else:
                        logger.warning(f"Groq API returned status {res.status_code}. Falling back to Google.")
                except Exception as e:
                    logger.warning(f"Groq Whisper transcription failed: {e}. Falling back to Google.")
            
            # 2. Fallback to free Google Speech Recognition
            logger.info("Transcribing with Google STT...")
            text = self.recognizer.recognize_google(audio)
            text_lower = text.lower().strip()
            logger.info(f"Heard (Google STT): '{text_lower}'")
            return text_lower
        except sr.UnknownValueError:
            logger.info("Speech was not understood.")
            return None
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Speech Recognition service; {e}")
            return None

    def check_for_wake_word(self, text):
        """Checks if any wake word defined in config is present in the transcribed text."""
        if not text:
            return False
        for word in config.WAKE_WORDS:
            if word in text:
                return True
        return False

    def check_and_extract_command(self, text):
        """
        Checks if a wake word is present in the text, and extracts any command spoken after it.
        Returns:
            - wake_detected (bool)
            - extracted_command (str)
        """
        if not text:
            return False, ""
            
        text_lower = text.lower().strip()
        for word in config.WAKE_WORDS:
            if word in text_lower:
                idx = text_lower.find(word)
                command = text_lower[idx + len(word):].strip()
                # Strip leading/trailing punctuation characters commonly added by speech transcribers
                command = command.lstrip(',.?!- ')
                return True, command
        return False, ""
