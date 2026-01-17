from speaker_embedding import extract_embeddings
from vad import vad_split
from audio_utils import normalize_audio
from clustering import cluster_speakers

# 1. Normalize audio
wav = normalize_audio("uploads/meeting.wav")

# 2. Split into speech segments
segment_files = vad_split(wav)

# 3. Extract embeddings
embs = extract_embeddings(segment_files)
print("Number of embeddings:", len(embs))
if embs:
    print("Shape of first embedding:", embs[0].shape)

# 4. Cluster speakers (change n_speakers as needed)
n_speakers = 2
labels = cluster_speakers(embs, n_speakers)
print("Speaker labels:", labels)
