from pydub import AudioSegment
import os
from config import UPLOAD_DIR

def normalize_audio(input_path):
    # Load the audio file
    audio = AudioSegment.from_file(input_path)
    
    # Convert to mono and 16kHz (required by Whisper & VAD)
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Output path inside uploads folder
    out_path = os.path.join(UPLOAD_DIR, "normalized.wav")
    
    # Export the normalized audio
    audio.export(out_path, format="wav")

    return out_path
