import whisper
import re
from config import WHISPER_MODEL

_model = whisper.load_model(WHISPER_MODEL)


def _clean_transcription(text: str) -> str:
    """Post-process transcription to fix common errors and names."""
    if not text:
        return ""

    # Fix some meeting-specific transcription artefacts we saw earlier
    text = re.sub(r"\btheió\b", "the I-O", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+)\s+(\d+)\s+ah\b", r"\1-\2", text)

    # Name correction dictionary (extend this as you see new errors)
    name_fixes = {
        r"\bLexmi\b": "Lekshmi",
        r"\bLexme\b": "Lekshmi",
        r"\bCash\b": "Akash",
        r"\bRiyunima\b": "Arunima",
        r"\bRiyunamma\b": "Arunima",
        r"\bcockle\b": "Gokul",
    }
    for pattern, replacement in name_fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Generic word/phrase fixes
    replacements = {
        r"\bgoing home\b": "going to",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Fix spacing around punctuation
    text = re.sub(r"\s+([.!?,:;])", r"\1", text)
    text = re.sub(r"([.!?])\s*([a-z])", r"\1 \2", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def transcribe(audio_path):
    """
    Transcribe audio with improved Whisper settings.
    Uses language hint and prompt for better accuracy.
    """
    result = _model.transcribe(
        audio_path,
        fp16=False,
        language="en",  # Explicitly set English
        task="transcribe",
        temperature=0.0,  # More deterministic, less creative errors
        beam_size=5,  # Default beam size, explicit for clarity
        condition_on_previous_text=True,  # Use context from previous segments
        initial_prompt=(
            "This is a meeting transcript. "
            "Speakers discuss agenda items, make decisions, and assign action items. "
            "Common Indian names such as Gokul, Dani, Lekshmi, Akash, and Arunima may appear."
        ),
    )

    text = result.get("text", "")
    text = _clean_transcription(text)
    return text
