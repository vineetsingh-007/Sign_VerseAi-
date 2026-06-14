import queue
import threading
import time
from speech.text_to_speech import TextToSpeechEngine

class SpeechGenerator:
    """
    Manages asynchronous speech output using a background worker thread
    and a FIFO queue. Prevents real-time computer vision frame drops.
    """
    def __init__(self, voice_gender="Female", rate=150, lang="English"):
        self.speech_queue = queue.Queue()
        self.engine = TextToSpeechEngine(voice_gender=voice_gender, rate=rate, lang=lang)
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        
        # Track last spoken sentence to prevent duplicate vocalizations in rapid succession
        self.last_spoken = ""
        self.last_spoken_time = 0

    def speak(self, text, force=False):
        """
        Adds text to the speech synthesis queue.
        - If 'force' is True, it will bypass duplicate checks.
        """
        if not text or not text.strip():
            return
            
        clean_text = text.strip()
        current_time = time.time()
        
        # Check for duplicates within a 2-second window
        if not force and clean_text == self.last_spoken and (current_time - self.last_spoken_time) < 2.0:
            return # skip duplicate
            
        self.speech_queue.put(clean_text)
        self.last_spoken = clean_text
        self.last_spoken_time = current_time

    def update_settings(self, voice_gender, rate, lang):
        """Updates the speech engine settings dynamically."""
        self.engine.update_settings(voice_gender=voice_gender, rate=rate, lang=lang)

    def _process_queue(self):
        """Background worker loop executing TTS requests."""
        while self.running:
            try:
                # Blocks until an item is available, timeout to allow check for self.running
                text = self.speech_queue.get(timeout=0.5)
                
                # Speak using offline mode as preference, falling back to online
                self.engine.speak(text, method="offline")
                
                # Allow a short cooling down period between sentences
                time.sleep(0.5)
                
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in speech generator worker thread: {e}")
                time.sleep(1.0)

    def stop(self):
        """Stops the speech worker thread cleanly."""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
