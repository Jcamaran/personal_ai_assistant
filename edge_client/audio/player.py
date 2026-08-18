# Audio playback through the default output device.
import sounddevice as sd
import soundfile as sf


def play_audio_file(file_path: str):
    """Play a WAV (or other soundfile-supported) audio file."""
    data, sample_rate = sf.read(file_path, dtype="float32")
    sd.play(data, sample_rate)
    sd.wait()
