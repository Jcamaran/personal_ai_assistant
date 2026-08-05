"""Speech-to-text using faster-whisper, with lazy model loading."""
import os
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()


class SpeechRecognizer:
    """Whisper-based speech recognition.

    The model is loaded on first use so importing this module stays cheap.
    """

    def __init__(self, model_size="base", device="auto"):
        """
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: 'auto', 'cuda', or 'cpu'
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self.is_using_gpu = False

    def _load_model(self):
        """Load the Whisper model, preferring GPU when device is 'auto'."""
        if self.model is not None:
            return

        print(f"Loading Whisper model: {self.model_size}...")

        try:
            if self.device == "auto":
                try:
                    self.model = WhisperModel(self.model_size, device="cuda", compute_type="float16")
                    self.is_using_gpu = True
                    print("Whisper loaded on GPU")
                except Exception as e:
                    print(f"GPU not available ({e}), falling back to CPU")
                    self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                    self.is_using_gpu = False
                    print("Whisper loaded on CPU")
            else:
                compute_type = "float16" if self.device == "cuda" else "int8"
                self.model = WhisperModel(self.model_size, device=self.device, compute_type=compute_type)
                self.is_using_gpu = (self.device == "cuda")
                print(f"Whisper loaded on {self.device}")

        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(self, audio_file_path, beam_size=3):
        """Transcribe an audio file to text.

        Falls back to CPU if GPU transcription fails due to missing CUDA
        libraries.
        """
        self._load_model()

        try:
            return self._run_transcription(audio_file_path, beam_size)
        except Exception as e:
            if self.is_using_gpu and "cublas" in str(e).lower():
                print(f"GPU transcription failed ({e}), reloading model on CPU")
                self.model = None
                self.device = "cpu"
                self._load_model()
                return self._run_transcription(audio_file_path, beam_size)
            raise

    def _run_transcription(self, audio_file_path, beam_size):
        segments, info = self.model.transcribe(audio_file_path, beam_size=beam_size)
        transcription = " ".join(segment.text for segment in segments)
        print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
        return transcription.strip()

    def __del__(self):
        if self.model is not None:
            del self.model


def transcribe_audio(audio_file_path):
    """Convenience function that creates a recognizer and transcribes a file."""
    model_size = os.getenv("WHISPER_MODEL", "base")
    recognizer = SpeechRecognizer(model_size=model_size, device="auto")
    return recognizer.transcribe(audio_file_path)
