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
# Whisper model options: tiny (fastest, least accurate), base (fast, good balance),
# small (slower, more accurate), medium (slow, very accurate), large (slowest, most accurate)
# For better accuracy with meeting audio, consider "small" model
WHISPER_MODEL = "small"  # Whisper model: tiny, base, small, medium, large
DEFAULT_SPEAKERS = 2
