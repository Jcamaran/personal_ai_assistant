"""Text-to-Speech using pyttsx3 (offline)

Might implement ElevenLabs TTS API in the future for better quality,
but this will require an internet connection and an API key.
"""
import pyttsx3
import os
from dotenv import load_dotenv

load_dotenv()


class TextToSpeech:
    """
    Text-to-Speech engine using pyttsx3.
    Lazy initializes the engine on first use.
    """
    
    def __init__(self, rate=150, volume=1.0, voice_index=1):
        """
        Initialize TextToSpeech.
        
        Args:
            rate (int): Speech rate (words per minute)
            volume (float): Volume level (0.0 to 1.0)
            voice_index (int): Voice index (0=male, 1=female, varies by system)
        """
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self.engine = None  # Lazy init
    
    def _init_engine(self):
        """Initialize TTS engine on first use (lazy loading)"""
        if self.engine is not None:
            return  # Already initialized
        
        print("Initializing TTS engine...")
        
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Set voice
            voices = self.engine.getProperty('voices')
            if voices and self.voice_index < len(voices):
                self.engine.setProperty('voice', voices[self.voice_index].id)
            
            print("✅ TTS engine initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize TTS engine: {e}")
            raise
    
    def speak(self, text):
        """
        Convert text to speech and play it.
        
        Args:
            text (str): Text to speak
        """
        self._init_engine()
        
        try:
            print(f"🗣️ Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
            
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            # Try to recover by reinitializing
            self.engine = None
            raise
    
    def save_to_file(self, text, output_file="speech.wav"):
        """
        Convert text to speech and save to file.
        
        Args:
            text (str): Text to convert
            output_file (str): Path to save audio file
        """
        self._init_engine()
        
        try:
            print(f"💾 Saving speech to {output_file}")
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()
            
        except Exception as e:
            print(f"❌ Failed to save speech: {e}")
            raise
    
    def set_voice(self, voice_index):
        """
        Change the voice dynamically.
        
        Args:
            voice_index (int): Index of the voice to use
        """
        self.voice_index = voice_index
        
        if self.engine is not None:
            voices = self.engine.getProperty('voices')
            if voices and voice_index < len(voices):
                self.engine.setProperty('voice', voices[voice_index].id)
                print(f"✅ Voice changed to index {voice_index}")
    
    def set_rate(self, rate):
        """
        Change speech rate dynamically.
        
        Args:
            rate (int): New speech rate
        """
        self.rate = rate
        if self.engine is not None:
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume):
        """
        Change volume dynamically.
        
        Args:
            volume (float): New volume (0.0 to 1.0)
        """
        self.volume = volume
        if self.engine is not None:
            self.engine.setProperty('volume', volume)
    
    def __del__(self):
        """Cleanup engine when object is destroyed"""
        if self.engine is not None:
            try:
                self.engine.stop()
            except:
                pass


# For backward compatibility with function-based approach
def speak_text(text):
    """Convenience function for speaking text."""
    tts = TextToSpeech(
        rate=int(os.getenv("TTS_RATE", 150)),
        volume=float(os.getenv("TTS_VOLUME", 1.0)),
        voice_index=1
    )
    tts.speak(text)


def save_speech_to_file(text, output_file="speech.wav"):
    """Convenience function for saving speech to file."""
    tts = TextToSpeech(
        rate=int(os.getenv("TTS_RATE", 150)),
        volume=float(os.getenv("TTS_VOLUME", 1.0)),
        voice_index=1
    )
    tts.save_to_file(text, output_file) 

















