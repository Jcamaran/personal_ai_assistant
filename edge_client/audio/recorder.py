# PyAudio recording logic
# Captures audio input from microphone
import sounddevice as sd
import numpy as np
import wave
import os
from dotenv import load_dotenv
import soundfile as sf


load_dotenv()
AUDIO_SAMPLE_RATE= int(os.getenv("AUDIO_SAMPLE_RATE", 16000))
AUDIO_CHANNELS= int(os.getenv("AUDIO_CHANNELS", 1))
MAX_RECORDING_DURATION= int(os.getenv("MAX_RECORDING_DURATION", 30))



def record_audio(duration = 5, output_file = "temp_recording.wav"):
    """ Record from microphone for given duration and saves audio to output_file in WAV format """
    
    # Delete existing file to ensure clean recording
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
        except:
            pass  
    
    print(f"Recording for {duration} seconds...")
    
    audio_data = sd.rec(int(duration * AUDIO_SAMPLE_RATE), samplerate = AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS, dtype='int16')

    sd.wait()  # Wait until recording is finished

    # save the recorded audio to a WAV file
    sf.write(output_file, audio_data, AUDIO_SAMPLE_RATE)
    
    print(f"✅ Recording saved to {output_file}")

    return output_file


def record_until_silence(output_file = "temp_recording.wav", silence_threshold = 500, silence_duration=2):
    """ Record from microphone until silence is detected and saves audio to output_file in WAV format """
    pass
