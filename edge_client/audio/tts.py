"""Text-to-speech using pyttsx3 (offline)."""
import pyttsx3
import os
from dotenv import load_dotenv

load_dotenv()


class TextToSpeech:
    """pyttsx3-based text-to-speech.

    A fresh engine is created for each call, which avoids the engine
    getting stuck between utterances.
    """

    def __init__(self, rate=150, volume=1.0, voice_index=1):
        """
        Args:
            rate: Speech rate in words per minute
            volume: Volume level from 0.0 to 1.0
            voice_index: System voice index (varies by platform)
        """
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index

    def _create_engine(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)

        voices = engine.getProperty("voices")
        if voices and self.voice_index < len(voices):
            engine.setProperty("voice", voices[self.voice_index].id)

        return engine

    def _run(self, action):
        engine = None
        try:
            engine = self._create_engine()
            action(engine)
            engine.runAndWait()
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass

    def speak(self, text):
        """Convert text to speech and play it."""
        self._run(lambda engine: engine.say(text))

    def save_to_file(self, text, output_file="speech.wav"):
        """Convert text to speech and save it to a file."""
        self._run(lambda engine: engine.save_to_file(text, output_file))

    def set_voice(self, voice_index):
        self.voice_index = voice_index

    def set_rate(self, rate):
        self.rate = rate

    def set_volume(self, volume):
        self.volume = volume


def _default_tts():
    return TextToSpeech(
        rate=int(os.getenv("TTS_RATE", 150)),
        volume=float(os.getenv("TTS_VOLUME", 1.0)),
        voice_index=int(os.getenv("TTS_VOICE_INDEX", 1)),
    )


def speak_text(text):
    """Convenience function for speaking text with default settings."""
    _default_tts().speak(text)


def save_speech_to_file(text, output_file="speech.wav"):
    """Convenience function for saving speech to a file with default settings."""
    _default_tts().save_to_file(text, output_file)
