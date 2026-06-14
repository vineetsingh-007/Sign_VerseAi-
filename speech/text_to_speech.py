import os
import tempfile
import threading
from gtts import gTTS

class TextToSpeechEngine:
    """
    A robust, hybrid Text-to-Speech Engine.
    - Uses pyttsx3 for offline, low-latency local speech on supported OS.
    - Uses gTTS (Google Translation TTS) for multi-language support and high quality.
    - Generates temp audio files for dashboard playback.
    """
    def __init__(self, voice_gender="Female", rate=150, lang="en"):
        self.voice_gender = voice_gender
        self.rate = rate
        self.lang = lang
        self._init_offline_engine()

    def _init_offline_engine(self):
        import sys
        # Disable offline pyttsx3 on macOS inside Streamlit because AppKit crashes with SIGSEGV in background threads
        if sys.platform == "darwin" and ("streamlit" in sys.modules or os.environ.get("DISABLE_PYTTSX3", "false").lower() == "true"):
            self.offline_engine = None
            self.offline_active = False
            return

        try:
            import pyttsx3
            self.offline_engine = pyttsx3.init()
            self.offline_engine.setProperty("rate", self.rate)
            
            # Select gender voice
            voices = self.offline_engine.getProperty("voices")
            if voices:
                # Typically index 0 is male, index 1 is female on most platforms
                gender_idx = 1 if self.voice_gender.lower() == "female" and len(voices) > 1 else 0
                self.offline_engine.setProperty("voice", voices[gender_idx].id)
            self.offline_active = True
        except Exception as e:
            print(f"Offline TTS initialization failed (will fallback to gTTS): {e}")
            self.offline_engine = None
            self.offline_active = False

    def update_settings(self, voice_gender=None, rate=None, lang=None):
        if voice_gender:
            self.voice_gender = voice_gender
        if rate:
            self.rate = rate
        if lang:
            self.lang = lang
        self._init_offline_engine()

    def speak(self, text, method="offline"):
        """
        Synthesizes text and speaks it out loud. Runs in a background thread to prevent blocking.
        """
        def run():
            if method == "offline" and self.offline_active:
                try:
                    self.offline_engine.say(text)
                    self.offline_engine.runAndWait()
                    return
                except Exception as e:
                    print(f"Offline speech failed: {e}. Falling back to online gTTS...")
            
            # Online gTTS fallback
            try:
                temp_file = self.generate_audio_file(text)
                if temp_file:
                    self._play_file(temp_file)
            except Exception as e:
                print(f"Online TTS speech failed: {e}")

        thread = threading.Thread(target=run)
        thread.start()

    def generate_audio_file(self, text):
        """
        Generates a temporary MP3 file of the text and returns its file path.
        Useful for Streamlit or web interfaces.
        """
        try:
            # Multi-language mapping for gTTS
            lang_map = {
                "english": "en",
                "hindi": "hi",
                "marathi": "mr",
                "spanish": "es",
                "french": "fr",
                "german": "de",
                "arabic": "ar",
                "japanese": "ja"
            }
            gtts_lang = lang_map.get(self.lang.lower(), "en")
            
            # Create gTTS object
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            
            # Save to temporary file in the project's scratch or appDataDir
            temp_dir = tempfile.gettempdir()
            filename = f"signverse_tts_{hash(text) & 0xffffffff}.mp3"
            filepath = os.path.join(temp_dir, filename)
            
            tts.save(filepath)
            return filepath
        except Exception as e:
            print(f"Error generating audio file via gTTS: {e}")
            return None

    def _play_file(self, filepath):
        """Plays an audio file using platform-specific commands."""
        import subprocess
        import sys
        
        try:
            if sys.platform == "darwin": # macOS
                subprocess.Popen(["afplay", filepath])
            elif sys.platform.startswith("linux"): # Linux
                subprocess.Popen(["aplay", filepath])
            elif sys.platform == "win32": # Windows
                os.startfile(filepath)
        except Exception as e:
            print(f"Error playing audio file: {e}")
