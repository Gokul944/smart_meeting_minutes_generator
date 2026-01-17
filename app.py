import os
from tqdm import tqdm
from config import DEFAULT_SPEAKERS, SEGMENT_DIR
from audio_utils import normalize_audio
from vad import vad_split
from speaker_embedding import extract_embeddings
from clustering import cluster_speakers
from stt import transcribe
from rule_nlp import summarize, extract_actions
import re

# ------------------ METADATA PARSER ------------------
def extract_metadata(transcript: str) -> dict:
    """
    Extract metadata (date, time, location, attendees, agenda) from transcript text.
    Returns a dictionary with keys: 'Date', 'Time', 'Location', 'Attendees', 'Agenda'
    """
    metadata = {}

    # Date
    date_match = re.search(
        r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|"
        r"Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s+of\s+\d{4})?",
        transcript,
        re.IGNORECASE
    )
    metadata["Date"] = date_match.group(0) if date_match else "Not found"

    # Time
    time_match = re.search(
        r"\b(?:exactly\s*)?\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?\b",
        transcript
    )
    metadata["Time"] = time_match.group(0) if time_match else "Not found"

    # Location
    location_match = re.search(
        r"\b(?:meeting (?:at|in|held at|held in)\s+([\w\s]+))",
        transcript,
        re.IGNORECASE
    )
    metadata["Location"] = location_match.group(1) if location_match else "Not found"

    # Attendees
    attendees_match = re.search(
        r"\b(?:attendees (?:are|include)|present)\s*[:\-]?\s*([\w\s,]+)",
        transcript,
        re.IGNORECASE
    )
    if attendees_match:
        attendees_text = attendees_match.group(1)
        attendees_list = re.split(r",| and ", attendees_text)
        metadata["Attendees"] = [name.strip() for name in attendees_list if name.strip()]
    else:
        metadata["Attendees"] = []

    # Agenda
    agenda_match = re.search(
        r"(?:agenda|topics?)\s*(?:for\s*[\w\s]+)?[:\-]?\s*(.+?)(?:\.|$)",
        transcript,
        re.IGNORECASE
    )
    metadata["Agenda"] = agenda_match.group(1).strip() if agenda_match else "Not found"

    return metadata

# ------------------ MAIN PIPELINE ------------------
def process_meeting(audio_path, n_speakers=DEFAULT_SPEAKERS):
    """
    Full meeting minutes pipeline:
    Audio → VAD → Speaker Embeddings → Clustering → STT → Summaries → Metadata
    """

    print("🔄 Processing meeting audio...")

    # 1. Normalize audio (mono, 16kHz)
    wav_path = normalize_audio(audio_path)

    # 2. Voice Activity Detection (returns list of segment .wav files)
    segment_files = vad_split(wav_path)
    if not segment_files:
        raise ValueError("❌ No speech detected in audio")

    print(f"Detected {len(segment_files)} speech segments.")

    # 3. Speaker embeddings (1 per segment)
    embeddings = extract_embeddings(segment_files)

    # 4. Speaker clustering
    labels = cluster_speakers(embeddings, n_speakers)

    # 5. Transcription per segment with progress bar
    speaker_text = {}
    for seg_path, label in tqdm(zip(segment_files, labels),
                                total=len(segment_files),
                                desc="Transcribing segments"):
        speaker = f"Speaker {label + 1}"
        text = transcribe(seg_path)
        speaker_text.setdefault(speaker, []).append(text)

    # 6. Merge speaker text
    for sp in speaker_text:
        speaker_text[sp] = " ".join(speaker_text[sp]).strip()

    # 7. Summaries
    overall_text = " ".join(speaker_text.values())
    overall_summary = summarize(overall_text, 6)
    speaker_summaries = {sp: summarize(txt, 3) for sp, txt in speaker_text.items()}

    # 8. Action items
    actions = extract_actions(overall_text)

    # 9. Extract metadata
    metadata = extract_metadata(overall_text)

    return overall_summary, speaker_summaries, actions, metadata

# ------------------ RUN SCRIPT ------------------
if __name__ == "__main__":

    audio_path = "uploads/meeting.wav"

    if not os.path.exists(audio_path):
        print("❌ Audio file not found:", audio_path)
        exit(1)

    try:
        overall, speaker_summaries, actions, metadata = process_meeting(audio_path)
    except Exception as e:
        print("❌ Error processing meeting:", e)
        exit(1)

    print("\n=============== METADATA ===============\n")
    for key, value in metadata.items():
        if isinstance(value, list):
            print(f"{key}: {', '.join(value) if value else 'Not found'}")
        else:
            print(f"{key}: {value}")

    print("\n=============== MEETING MINUTES ===============\n")
    print("📝 OVERALL SUMMARY:\n", overall)

    print("\n👥 SPEAKER-WISE SUMMARY:\n")
    for sp, txt in speaker_summaries.items():
        print(f"{sp}:")
        print(txt)
        print()

    print("✅ ACTION ITEMS:\n")
    for a in actions:
        print("-", a)
