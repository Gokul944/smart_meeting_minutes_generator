from audio_utils import normalize_audio
from vad import vad_split
from speaker_embedding import extract_embeddings

def main():
    # 1. Normalize audio
    wav = normalize_audio("uploads/meeting.wav")
    print(" Normalized audio saved at:", wav)

    # 2. Split into speech segments (returns list of .wav files)
    segment_files = vad_split(wav)
    print(f"🎧 Number of segments: {len(segment_files)}")

    if not segment_files:
        print(" No speech segments found")
        return

    # 3. Extract embeddings
    embeddings = extract_embeddings(segment_files)
    print(" Number of embeddings:", len(embeddings))

    if embeddings:
        print(" Shape of first embedding:", embeddings[0].shape)

if __name__ == "__main__":
    main()
