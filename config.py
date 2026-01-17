import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folders for audio files
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SEGMENT_DIR = os.path.join(BASE_DIR, "segments")

# Audio settings
SAMPLE_RATE = 16000
MIN_SPEECH_SEC = 0.5  # minimum speech duration in seconds

# Model & pipeline settings
WHISPER_MODEL = "base"  # Whisper model: tiny, base, small, medium, large
DEFAULT_SPEAKERS = 2
