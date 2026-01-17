# speaker_embedding.py
from resemblyzer import VoiceEncoder, preprocess_wav

# Load encoder once (important for performance)
encoder = VoiceEncoder()

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
        emb = encoder.embed_utterance(wav)
        embeddings.append(emb)

    return embeddings
