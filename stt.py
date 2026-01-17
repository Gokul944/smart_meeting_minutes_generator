import whisper
from config import WHISPER_MODEL

_model = whisper.load_model(WHISPER_MODEL)

def transcribe(audio_path):
    result = _model.transcribe(audio_path, fp16=False)
    return result["text"]
