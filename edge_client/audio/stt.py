"""Speech-to-Text using faster-whisper with lazy loading"""
import os
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()


class SpeechRecognizer:
    """
    Speech-to-Text using Whisper model.
    Lazy loads the model on first use to avoid loading at import time.
    """
    
    def __init__(self, model_size="base", device="auto"):
        """
        Initialize SpeechRecognizer.
        
        Args:
            model_size (str): Whisper model size (tiny, base, small, medium, large)
            device (str): Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_size = model_size
        self.device = device
        self.model = None  # Lazy load
        self.is_using_gpu = False
    
    def _load_model(self):
        """Load Whisper model on first use (lazy loading)"""
        if self.model is not None:
            return  # Already loaded
        
        print(f"Loading Whisper model: {self.model_size}...")
        
        try:
            if self.device == "auto":
                # Try GPU first, fallback to CPU
                try:
                    self.model = WhisperModel(
                        self.model_size,
                        device="cuda",
                        compute_type="float16"
                    )
                    self.is_using_gpu = True
                    print("✅ Whisper loaded on GPU")
                except Exception as e:
                    print(f"GPU not available ({e}), falling back to CPU...")
                    self.model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8"
                    )
                    self.is_using_gpu = False
                    print("✅ Whisper loaded on CPU")
            else:
                # Use specified device
                compute_type = "float16" if self.device == "cuda" else "int8"
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=compute_type
                )
                self.is_using_gpu = (self.device == "cuda")
                print(f"✅ Whisper loaded on {self.device}")
                
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            raise
    
    def transcribe(self, audio_file_path, beam_size=3):
        """
        Transcribe audio file to text.
        
        Args:
            audio_file_path (str): Path to audio file
            beam_size (int): Beam size for decoding (higher = more accurate but slower)
        
        Returns:
            str: Transcribed text
        """
        # Load model if not already loaded
        self._load_model()
        
        print(f"🎯 Transcribing audio...")
        
        # Transcribe
        segments, info = self.model.transcribe(audio_file_path, beam_size=beam_size)
        
        # Combine segments
        transcription = " ".join([segment.text for segment in segments])
        
        print(f"📝 Language: {info.language} (confidence: {info.language_probability:.2f})")
        
        return transcription.strip()
    
    def __del__(self):
        """Cleanup model when object is destroyed"""
        if self.model is not None:
            del self.model


# For backward compatibility with function-based approach
def transcribe_audio(audio_file_path):
    """
    Convenience function for transcribing audio.
    Creates a new SpeechRecognizer instance each time.
    """
    model_size = os.getenv("WHISPER_MODEL", "base")
    recognizer = SpeechRecognizer(model_size=model_size, device="auto")
    return recognizer.transcribe(audio_file_path)