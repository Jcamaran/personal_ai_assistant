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
    Reinitializes engine for each speak call for maximum reliability.
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
        print("✅ TTS configuration loaded")
    
    def _create_engine(self):
        """Create a fresh TTS engine instance."""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
            
            # Set voice
            voices = engine.getProperty('voices')
            if voices and self.voice_index < len(voices):
                engine.setProperty('voice', voices[self.voice_index].id)
            
            return engine
            
        except Exception as e:
            print(f"❌ Failed to create TTS engine: {e}")
            raise
    
    def speak(self, text):
        """
        Convert text to speech and play it.
        Creates a fresh engine each time for reliability.
        
        Args:
            text (str): Text to speak
        """
        engine = None
        try:
            print(f"🗣️ Speaking: {text}")
            
            # Create fresh engine for each call (prevents stuck state)
            engine = self._create_engine()
            engine.say(text)
            engine.runAndWait()
            
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            raise
        finally:
            # Clean up engine
            if engine is not None:
                try:
                    engine.stop()
                    del engine
                except:
                    pass
    
    def save_to_file(self, text, output_file="speech.wav"):
        """
        Convert text to speech and save to file.
        
        Args:
            text (str): Text to convert
            output_file (str): Path to save audio file
        """
        engine = None
        try:
            print(f"💾 Saving speech to {output_file}")
            
            engine = self._create_engine()
            engine.save_to_file(text, output_file)
            engine.runAndWait()
            
        except Exception as e:
            print(f"❌ Failed to save speech: {e}")
            raise
        finally:
            if engine is not None:
                try:
                    engine.stop()
                    del engine
                except:
                    pass
    
    def set_voice(self, voice_index):
        """
        Change the voice for future speech.
        
        Args:
            voice_index (int): Index of the voice to use
        """
        self.voice_index = voice_index
        print(f"✅ Voice will be changed to index {voice_index} on next speak")
    
    def set_rate(self, rate):
        """
        Change speech rate for future speech.
        
        Args:
            rate (int): New speech rate
        """
        self.rate = rate
    
    def set_volume(self, volume):
        """
        Change volume for future speech.
        
        Args:
            volume (float): New volume (0.0 to 1.0)
        """
        self.volume = volume


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

















