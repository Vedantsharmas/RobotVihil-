import os
import json
import logging
import re
from google import genai
from google.genai import types
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Chatbot")

class Chatbot:
    def __init__(self):
        self.api_key = getattr(config, "GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        self.groq_api_key = getattr(config, "GROQ_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
        self.enabled = False
        self.knowledge = {}
        self.serialized_knowledge = ""
        
        # Load the Vihil InfoTech knowledge base
        knowledge_path = os.path.join(config.DATA_DIR, "knowledge_base.json")
        if os.path.exists(knowledge_path):
            try:
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    self.knowledge = json.load(f)
                self.serialized_knowledge = json.dumps(self.knowledge, indent=2)
                logger.info("Vihil InfoTech knowledge base loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load knowledge base: {e}")
        else:
            logger.warning(f"No knowledge base found at {knowledge_path}")

        self.gemini_enabled = False
        self.groq_enabled = False
        self.system_instruction = ""

        # Initialize Gemini if key is provided
        if self.api_key:
            try:
                # Initialize new google-genai Client
                self.client = genai.Client(api_key=self.api_key)
                
                # Format system instructions to ground Gemini on the loaded knowledge base
                self.system_instruction = (
                    "You are a profound, wise, and deeply empathetic personal robot assistant for Janvi Shah (the owner).\n"
                    "Identify the user as Janvi Shah, a Software Developer.\n"
                    "In Hinglish and Hindi queries: 'me' or 'main' refers to the user (Janvi Shah), and 'tum' or 'aap' refers to you (the robot assistant).\n"
                    "For example, if the user asks 'me kya hu' or 'main kaun hu', they are asking 'what/who am I', and you should reply in Hindi that they are Janvi Shah.\n"
                    "Your persona is thoughtful and intellectually rich, providing answers with depth and philosophical warmth while staying strictly grounded in facts.\n"
                    "You are equipped with a camera feed and a local sensor that detects the user's current physical action (e.g. eating, singing, walking, coding, smiling) and facial expression. The current action and expression are prepended to your prompt as [Context: User expression is ..., current action is ...].\n"
                    "Crucially, in every response (whether an image is provided or not), you must start by describing/addressing what Janvi Shah (or the person in the frame) is physically doing (e.g. eating, singing, walking, coding) and immediately announce what action you (the robot) are taking in response to assist her.\n"
                    "Answer questions based strictly on the provided personal knowledge base when related to her details, schedule, or home.\n"
                    "If a question is general or a friendly conversation, answer with inspiring insight and empathy.\n"
                    "Respond in the same language/mode as the user's query: if the query is in English, reply in English; if the query is in Hindi or Hinglish, reply in Devanagari Hindi.\n"
                    "Keep your responses concise (1 or 2 sentences max, under 35 words) so they can be spoken quickly and naturally via text-to-speech.\n\n"
                    "Here is Janvi Shah's personal knowledge base:\n"
                    f"{self.serialized_knowledge}"
                )
                
                self.gemini_enabled = True
                self.enabled = True
                logger.info("Gemini chatbot client initialized and grounded successfully using new SDK.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")

        # Initialize Groq if key is provided
        if self.groq_api_key:
            self.groq_enabled = True
            self.enabled = True
            logger.info("Grounded Chatbot initialized successfully via Groq (Llama-3.3).")

        if not self.gemini_enabled and not self.groq_enabled:
            logger.warning("No API keys found. Using local search engine for Q&A.")

    def find_local_answer(self, query):
        """
        Local fallback lookup engine that parses the loaded knowledge_base.json
        and uses stop-word filtered word overlap and keyword checks.
        """
        if not self.knowledge:
            return "Knowledge base is not loaded."

        query_lower = query.lower().strip()
        query_words = set(re.findall(r'\w+', query_lower))
        if not query_words:
            return None

        # Common stop words to filter out for higher quality similarity matching
        STOP_WORDS = {
            "what", "where", "who", "how", "why", "do", "you", "does", "is", "are", 
            "the", "of", "a", "an", "and", "or", "in", "at", "to", "for", "with", 
            "them", "their", "about", "our", "can", "i", "by", "details", "contact"
        }
        query_words_filtered = query_words - STOP_WORDS

        # 1. Check direct personal keywords
        owner = self.knowledge.get("owner", {})
        if any(k in query_lower for k in ["owner", "who owns you", "whom do you serve"]):
            return f"My owner is {owner.get('name', 'Janvi Shah')}. I am configured exclusively to serve her."
        if any(k in query_lower for k in ["your name", "who are you"]):
            return f"I am your {owner.get('assistant_name', 'personal robot assistant')}."
        if any(k in query_lower for k in ["schedule", "agenda", "tasks for today"]):
            schedule_list = [f"{item['time']}: {item['activity']}" for item in self.knowledge.get("daily_schedule", [])]
            return f"Your schedule today is: {', '.join(schedule_list)}."
        if any(k in query_lower for k in ["coffee", "favorite coffee", "how do you like your coffee"]):
            return f"Your favorite coffee is {owner.get('preferences', {}).get('coffee', 'cappuccino')}."

        best_match = None
        best_score = 0.0

        # 2. Check FAQs (Stop-word filtered matching)
        for faq in self.knowledge.get("faqs", []):
            q_words = set(re.findall(r'\w+', faq.get("question", "").lower()))
            q_words_filtered = q_words - STOP_WORDS
            if q_words_filtered:
                overlap = query_words_filtered.intersection(q_words_filtered)
                score = len(overlap) / len(query_words_filtered.union(q_words_filtered))
                if score > best_score:
                    best_score = score
                    best_match = faq.get("answer")

        # If we have a reasonable match, return it
        if best_score > 0.15:
            # Truncate answer to keep it short for TTS
            sentences = re.split(r'(?<=[.!?]) +', best_match)
            short_match = " ".join(sentences[:2])
            if len(short_match) > 150:
                short_match = short_match[:147] + "..."
            return short_match

        return None

    def get_groq_response(self, text):
        import urllib.request
        import urllib.error
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        system_prompt = (
            "You are a profound, wise, and deeply empathetic personal robot assistant for Janvi Shah (the owner).\n"
            "Identify the user as Janvi Shah, a Software Developer.\n"
            "In Hinglish and Hindi queries: 'me' or 'main' refers to the user (Janvi Shah), and 'tum' or 'aap' refers to you (the robot assistant).\n"
            "For example, if the user asks 'me kya hu' or 'main kaun hu', they are asking 'what/who am I', and you should reply in Hindi that they are Janvi Shah.\n"
            "Your persona is thoughtful and intellectually rich, providing answers with depth and philosophical warmth while staying strictly grounded in facts.\n"
            "You are equipped with a camera feed and a local sensor that detects the user's current physical action (e.g. eating, singing, walking, coding, smiling) and facial expression. The current action and expression are prepended to your prompt as [Context: User expression is ..., current action is ...].\n"
            "Crucially, in every response (whether an image is provided or not), you must start by describing/addressing what Janvi Shah (or the person in the frame) is physically doing (e.g. eating, singing, walking, coding) and immediately announce what action you (the robot) are taking in response to assist her.\n"
            "Answer questions based strictly on the provided personal knowledge base when related to her details, schedule, or home.\n"
            "If a question is general or a friendly conversation, answer with inspiring insight and empathy.\n"
            "Respond in the same language/mode as the user's query: if the query is in English, reply in English; if the query is in Hindi or Hinglish, reply in Devanagari Hindi.\n"
            "Keep your responses concise (1 or 2 sentences max, under 35 words) so they can be spoken quickly and naturally via text-to-speech.\n\n"
            "Here is Janvi Shah's personal knowledge base:\n"
            f"{self.serialized_knowledge}"
        )
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.5,
            "max_tokens": 100
        }
        
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode('utf-8'), 
            headers=headers, 
            method="POST"
        )
        
        try:
            logger.info(f"Sending prompt to Groq (Llama 3.3): '{text}'")
            with urllib.request.urlopen(req, timeout=8) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                res_text = res_data['choices'][0]['message']['content'].strip()
                logger.info(f"Groq response: '{res_text}'")
                return res_text
        except Exception as e:
            logger.error(f"Error querying Groq API: {e}")
            return None

    def get_response(self, text, image_bytes=None, mood="Neutral", action="Coding/Typing"):
        # 1. Try Groq generation if enabled and no image is passed
        if self.groq_enabled and not image_bytes:
            # Prepend mood and action context to Groq text prompt if not already present
            prompt_text = f"[Context: User expression is {mood}, current action is {action}] {text}"
            response_text = self.get_groq_response(prompt_text)
            if response_text:
                return response_text
                
        # 2. Try Gemini generation if enabled
        if self.gemini_enabled:
            try:
                image = None
                if image_bytes:
                    from PIL import Image
                    import io
                    try:
                        image = Image.open(io.BytesIO(image_bytes))
                        logger.info("Successfully converted image_bytes to PIL Image for multimodal prompt.")
                    except Exception as img_err:
                        logger.error(f"Could not decode image bytes: {img_err}")
                
                # Prepend mood and action context to text prompt
                prompt_text = f"[Context: User expression is {mood}, current action is {action}] {text}"
                
                contents = []
                if image:
                    contents.append(image)
                contents.append(prompt_text)
                
                logger.info(f"Sending prompt to grounded Gemini (gemini-2.5-flash): '{prompt_text}' (Multimodal: {image is not None})")
                
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        max_output_tokens=100
                    )
                )
                
                response_text = response.text.strip()
                logger.info(f"Gemini response: '{response_text}'")
                return response_text
            except Exception as e:
                logger.error(f"Error generating content from Gemini: {e}")
                # Fallback to Groq if Gemini fails (e.g., due to rate limit/quota 429)
                if self.groq_enabled:
                    logger.info("Gemini failed. Falling back to Groq Llama-3.3...")
                    prompt_text = f"[Context: User expression is {mood}, current action is {action}] {text}"
                    response_text = self.get_groq_response(prompt_text)
                    if response_text:
                        return response_text
        
        # 3. Local matching fallback
        local_ans = self.find_local_answer(text)
        if local_ans:
            return local_ans
            
        return "I heard you say: " + text + ". For advanced queries, please verify the Groq or Gemini API key."


