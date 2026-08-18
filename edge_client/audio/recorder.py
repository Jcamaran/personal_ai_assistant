# Microphone recording.
import sounddevice as sd
import numpy as np
import os
from dotenv import load_dotenv
import soundfile as sf

load_dotenv()

AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", 16000))
AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", 1))
MAX_RECORDING_DURATION = int(os.getenv("MAX_RECORDING_DURATION", 30))


def _remove_if_exists(path: str):
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def record_audio(duration=5, output_file="temp_recording.wav"):
    """Record from the microphone for a fixed duration and save as WAV."""
    _remove_if_exists(output_file)

    print(f"Recording for {duration} seconds...")

    audio_data = sd.rec(
        int(duration * AUDIO_SAMPLE_RATE),
        samplerate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
        dtype="int16",
    )
    sd.wait()

    sf.write(output_file, audio_data, AUDIO_SAMPLE_RATE)
    print(f"Recording saved to {output_file}")

    return output_file


def record_until_silence(
    output_file="temp_recording.wav",
    silence_threshold=500,
    silence_duration=1.5,
    max_duration=None,
    block_duration=0.1,
):
    """Record from the microphone until sustained silence is detected.

    Recording stops after `silence_duration` seconds below the RMS
    `silence_threshold` (once speech has started), or when `max_duration`
    is reached. Returns the output file path.
    """
    if max_duration is None:
        max_duration = MAX_RECORDING_DURATION

    _remove_if_exists(output_file)

    block_size = int(AUDIO_SAMPLE_RATE * block_duration)
    max_blocks = int(max_duration / block_duration)
    silence_blocks_needed = int(silence_duration / block_duration)

    recorded_blocks = []
    consecutive_silent = 0
    speech_started = False

    print("Recording... (stops on silence)")

    with sd.InputStream(
        samplerate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
        dtype="int16",
        blocksize=block_size,
    ) as stream:
        for _ in range(max_blocks):
            block, _overflowed = stream.read(block_size)
            recorded_blocks.append(block.copy())

            rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))

            if rms >= silence_threshold:
                speech_started = True
                consecutive_silent = 0
            elif speech_started:
                consecutive_silent += 1
                if consecutive_silent >= silence_blocks_needed:
                    break

    audio_data = np.concatenate(recorded_blocks, axis=0)
    sf.write(output_file, audio_data, AUDIO_SAMPLE_RATE)
    print(f"Recording saved to {output_file}")

    return output_file
