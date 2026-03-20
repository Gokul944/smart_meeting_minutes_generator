# speaker_embedding.py
from resemblyzer import VoiceEncoder, preprocess_wav
import threading

encoder = None
encoder_lock = threading.Lock()


def _get_encoder():
    """Load embedding encoder on first use (prevents slow Streamlit startup)."""
    global encoder
    if encoder is not None:
        return encoder
    with encoder_lock:
        if encoder is None:
            encoder = VoiceEncoder()
    return encoder

def extract_embeddings(segment_files):
    """
    Extract speaker embeddings from segmented wav files.

    Args:
        segment_files (list[str]): List of paths to .wav segment files

    Returns:
        list: List of 256-dim speaker embeddings
    """
    embeddings = []

    for seg_path in segment_files:
        # Load and preprocess audio
        wav = preprocess_wav(seg_path)

        # Skip empty / very short segments
        if len(wav) == 0:
            continue

        # Generate speaker embedding
        emb = _get_encoder().embed_utterance(wav)
        embeddings.append(emb)

    return embeddings
