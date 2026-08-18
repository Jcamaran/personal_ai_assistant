"""Text-to-Speech using Microsoft Edge neural voices, with pyttsx3 fallback."""
import asyncio
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()

DEFAULT_EDGE_VOICE = "en-US-JennyNeural"


def sanitize_for_speech(text: str) -> str:
    """Strip markdown and symbols that TTS would read aloud (asterisk, plus, hash)."""
    if not text:
        return ""

    spoken = text
    spoken = re.sub(
        r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]",
        lambda match: (match.group(2) or match.group(1)).strip(),
        spoken,
    )
    spoken = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", spoken)
    spoken = re.sub(r"```[\s\S]*?```", " ", spoken)
    spoken = re.sub(r"`([^`]+)`", r"\1", spoken)
    spoken = re.sub(r"(\*\*|__)(.*?)\1", r"\2", spoken)
    spoken = re.sub(r"(\*|_)(.*?)\1", r"\2", spoken)
    spoken = re.sub(r"(?m)^#+\s*", "", spoken)
    spoken = re.sub(r"(?m)^\s*[-*+]\s+", "", spoken)
    spoken = re.sub(r"(?m)^\s*\d+[.)]\s+", "", spoken)
    for symbol in ("*", "+", "#", "_", "`", "|", "~", ">", "<"):
        spoken = spoken.replace(symbol, " ")
    spoken = re.sub(r"-{2,}", " ", spoken)
    spoken = re.sub(r"\s+", " ", spoken).strip()
    return spoken


def _run_async(coro):
    """Run an async coroutine from sync code (hotkey callback / main thread)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class TextToSpeech:
    """Speaks text with Microsoft Edge neural voices, falling back to pyttsx3."""

    def __init__(self, rate=150, volume=1.0, voice_index=1, engine=None, voice=None):
        """
        Args:
            rate: Speech rate in words per minute (mapped to Edge rate %).
            volume: Volume level from 0.0 to 1.0
            voice_index: pyttsx3 voice index used only on fallback
            engine: 'edge' or 'pyttsx3'. Defaults to TTS_ENGINE env / edge.
            voice: Edge voice name, e.g. en-US-JennyNeural
        """
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self.engine_name = (engine or os.getenv("TTS_ENGINE", "edge")).strip().lower()
        self.voice = voice or os.getenv("TTS_VOICE", DEFAULT_EDGE_VOICE)
        if self.engine_name in {"edge", "edge-tts", "neural"}:
            self.engine_name = "edge"
        print(f"TTS configuration loaded ({self.engine_name}, voice={self.voice})")

    def _edge_rate(self) -> str:
        percent = int((self.rate / 200.0 - 1.0) * 100)
        percent = max(-50, min(100, percent))
        return f"{percent:+d}%"

    def _edge_volume(self) -> str:
        percent = int((self.volume - 1.0) * 100)
        percent = max(-50, min(100, percent))
        return f"{percent:+d}%"

    async def _synthesize_edge(self, text: str, output_file: str) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self._edge_rate(),
            volume=self._edge_volume(),
        )
        await communicate.save(output_file)

    def _play_mp3(self, path: str) -> None:
        import pygame

        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(max(0.0, min(1.0, self.volume)))
        pygame.mixer.music.play()
        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(20)
        pygame.mixer.music.unload()
        pygame.mixer.quit()

    def _speak_edge(self, spoken: str) -> None:
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            _run_async(self._synthesize_edge(spoken, path))
            self._play_mp3(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _create_pyttsx3_engine(self):
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        voices = engine.getProperty("voices")
        if voices and self.voice_index < len(voices):
            engine.setProperty("voice", voices[self.voice_index].id)
        return engine

    def _speak_pyttsx3(self, spoken: str) -> None:
        engine = None
        try:
            engine = self._create_pyttsx3_engine()
            engine.say(spoken)
            engine.runAndWait()
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass

    def speak(self, text):
        """Convert text to speech and play it."""
        spoken = sanitize_for_speech(text)
        if not spoken:
            print("Nothing to speak after sanitizing markdown")
            return

        print(f"Speaking: {spoken}")

        if self.engine_name == "edge":
            try:
                self._speak_edge(spoken)
                return
            except Exception as e:
                print(f"Edge neural TTS failed ({e}); falling back to pyttsx3")

        self._speak_pyttsx3(spoken)

    def save_to_file(self, text, output_file="speech.wav"):
        """Convert text to speech and save it to a file."""
        spoken = sanitize_for_speech(text)
        if self.engine_name == "edge":
            try:
                _run_async(self._synthesize_edge(spoken, output_file))
                return
            except Exception as e:
                print(f"Edge neural TTS save failed ({e}); falling back to pyttsx3")

        engine = None
        try:
            engine = self._create_pyttsx3_engine()
            engine.save_to_file(spoken, output_file)
            engine.runAndWait()
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass

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
