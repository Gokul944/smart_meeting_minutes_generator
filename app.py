import os
from tqdm import tqdm
from config import DEFAULT_SPEAKERS, SEGMENT_DIR
from audio_utils import normalize_audio
from vad import vad_split
from speaker_embedding import extract_embeddings
from clustering import cluster_speakers
from stt import transcribe
from rule_nlp import summarize, extract_actions, extract_decisions
from llm_minutes import generate_minutes_with_gemini, has_gemini, convert_speaker_summaries_to_reported
import re
from collections import Counter

# ------------------ METADATA PARSER ------------------
def extract_metadata(transcript: str) -> dict:
    """
    Extract metadata (date, time, location, attendees, agenda) from transcript text.
    Returns a dictionary with keys: 'Date', 'Time', 'Location', 'Attendees', 'Agenda'
    """
    metadata = {}

    # Date - improved pattern (year is now optional so "March 26" also matches)
    date_match = re.search(
        r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|"
        r"Sep|Sept|September|Oct|October|Nov|November|Dec|December)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?(?:[,\s]+\d{4})?",
        transcript,
        re.IGNORECASE
    )
    metadata["Date"] = date_match.group(0).strip() if date_match else "Not found"

    # Time - more specific pattern to avoid matching "105 PM" as time
    # Also accept dot as hour:minute separator (e.g. "10.30 am")
    time_patterns = [
        r"\b(?:at|starting at|begins at|for|scheduled for)\s+(?:exactly\s*)?(\d{1,2}(?:[.:]+\d{2})?\s*(?:am|pm|AM|PM|a\.m\.|p\.m\.))\b",
        r"\b(\d{1,2}(?:[.:]+\d{2})?\s*(?:am|pm|AM|PM|a\.m\.|p\.m\.))\b",
    ]
    time_match = None
    for pattern in time_patterns:
        time_match = re.search(pattern, transcript, re.IGNORECASE)
        if time_match:
            # Extract just the time part, not "exactly"
            time_str = time_match.group(1) if time_match.lastindex else time_match.group(0)
            # Validate it's a reasonable time (not "105 PM")
            if re.match(r'\d{1,2}(?:[.:]\d{2})?\s*(?:am|pm|AM|PM|a\.m\.|p\.m\.)', time_str):
                hour_part = re.match(r'(\d{1,2})', time_str)
                if hour_part and int(hour_part.group(1)) <= 12:
                    metadata["Time"] = time_str.strip()
                    break
    if "Time" not in metadata:
        metadata["Time"] = "Not found"

    # Location - improved pattern, exclude times
    location_patterns = [
        r"\b(?:meeting|held|convened)\s+(?:at|in)\s+([A-Z][\w\s]+?)(?:\s+on|\s+at\s+\d|\.|$)",
        r"\b(location|venue)[:\-]?\s+([A-Z][\w\s]+?)(?:\s+on|\s+at\s+\d|\.|$)",
    ]
    location_match = None
    for pattern in location_patterns:
        location_match = re.search(pattern, transcript, re.IGNORECASE)
        if location_match:
            loc = location_match.group(1) if location_match.lastindex >= 1 else location_match.group(2)
            # Filter out if it looks like a time or date
            if not re.search(r'\d{1,2}\s*(?:am|pm|AM|PM)', loc, re.IGNORECASE):
                metadata["Location"] = loc.strip()
                break
    if "Location" not in metadata:
        metadata["Location"] = "Not found"

    # Attendees - improved extraction
    attendees_patterns = [
        r"\b(?:attendees|participants|present|attending)\s*(?:are|include|were)[:\-]?\s*([A-Z][\w\s,]+?)(?:\s+\.|\s+and\s+the|$)",
        r"\b(?:members|committee members)[:\-]?\s*([A-Z][\w\s,]+?)(?:\s+\.|$)",
    ]
    attendees_list = []
    for pattern in attendees_patterns:
        attendees_match = re.search(pattern, transcript, re.IGNORECASE)
        if attendees_match:
            attendees_text = attendees_match.group(1)
            # Split by comma or "and"
            parts = re.split(r',|\s+and\s+', attendees_text)
            attendees_list = [name.strip() for name in parts if name.strip() and len(name.strip()) > 2]
            if attendees_list:
                break
    # Fallback: infer attendees as capitalized names mentioned in transcript
    if not attendees_list:
        # Find capitalized words that look like names (e.g. Gokul, Dani, Akash)
        name_candidates = re.findall(r"\b([A-Z][a-z]+)\b", transcript)
        # Ignore common capitalized words that are not person names
        ignore_names = {
            "I", "We", "You", "They", "He", "She", "It",
            "AI", "Hey", "Let", "Lets", "Let’s",
            "And", "But", "Or", "If", "When", "What", "That",
            "This", "Those", "These", "There", "Here",
            "Have", "Has", "Will", "Can", "May", "Must",
            "Yes", "No", "Okay", "Ok", "Well", "Thanks", "Thank",
            # Months (already used for Date)
            "Jan", "January", "Feb", "February", "Mar", "March", "Apr", "April",
            "May", "Jun", "June", "Jul", "July", "Aug", "August",
            "Sep", "Sept", "September", "Oct", "October", "Nov", "November",
            "Dec", "December",
            # Common non-name words that can appear capitalized at sentence starts
            "So", "Confident",
        }
        # If you already know the likely attendee names, allow them even if they appear once.
        # This prevents returning an empty attendee list when each name only appears a single time.
        likely_person_names = {
            "Gokul",
            "Dani",
            "Akash",
            "Arunima",
            "Lekshmi",
        }
        # Only keep likely person names:
        # - not in ignore list
        # - looks like a real name (length >= 3)
        # - appears at least twice in the transcript (reduces false positives),
        #   OR is in the likely-person allowlist above.
        counts = Counter(
            n for n in name_candidates
            if n not in ignore_names and len(n) >= 3
        )
        valid = {
            name
            for name, c in counts.items()
            if (c >= 2) or (name in likely_person_names)
        }

        # Preserve the order of first appearance in the transcript
        ordered = []
        seen = set()
        for name in name_candidates:
            if name in valid and name not in seen:
                seen.add(name)
                ordered.append(name)

        attendees_list = ordered

    metadata["Attendees"] = attendees_list if attendees_list else []

    # Agenda - improved extraction
    agenda_patterns = [
        r"\b(?:agenda|topics?|items?)\s*(?:for|of|today)[:\-]?\s*([A-Z][^.]{10,100}?)(?:\s+\.|$)",
        r"\b(?:discussing|discuss|agenda includes?)[:\-]?\s*([A-Z][^.]{10,100}?)(?:\s+\.|$)",
    ]
    agenda_match = None
    for pattern in agenda_patterns:
        agenda_match = re.search(pattern, transcript, re.IGNORECASE)
        if agenda_match:
            agenda_text = agenda_match.group(1).strip()
            # Clean up common prefixes
            agenda_text = re.sub(r'^(?:for|of|today|the)\s+', '', agenda_text, flags=re.IGNORECASE)
            if len(agenda_text) > 10:  # Must be meaningful length
                metadata["Agenda"] = agenda_text
                break

    # Fallback: infer agenda from first sentence mentioning meeting / seminar / presentation / topic
    if "Agenda" not in metadata or metadata["Agenda"] == "Not found":
        # Rough sentence split
        sentences = re.split(r"(?<=[.!?])\s+", transcript)
        keywords = ("meeting", "seminar", "presentation", "topic", "agenda", "discuss")
        for sent in sentences:
            lower = sent.lower()
            if any(k in lower for k in keywords):
                # Remove greeting like "Hey Dani,"
                cleaned = re.sub(r"^hey\s+\w+,\s*", "", sent.strip(), flags=re.IGNORECASE)
                metadata["Agenda"] = cleaned.strip()
                break

    if "Agenda" not in metadata:
        metadata["Agenda"] = "Not found"

    return metadata


def build_structured_minutes(
    metadata: dict,
    overall_text: str,
    actions: list[str],
) -> str:
    """
    Build a single structured minutes string in the format:

    Date:
    Time:
    Location:
    Attendees:
    Agenda:

    MEETING MINUTES:
    - ...

    DECISIONS:
    - ...

    ACTION ITEMS:
    - Name: Task
    """
    date = metadata.get("Date", "Not found")
    time = metadata.get("Time", "Not found")
    location = metadata.get("Location", "Not found")
    attendees = metadata.get("Attendees", []) or []
    agenda = metadata.get("Agenda", "Not found")

    # Key points for MEETING MINUTES (short extractive summary)
    key_points_raw = summarize(overall_text, 5)
    key_points: list[str] = []
    if key_points_raw:
        # Phrases that are usually just rhetorical / confirmation, not real content
        approval_starts = (
            "that sounds",  # that sounds perfect, sounds good, etc.
            "sounds good",
            "sounds great",
            "that's great",
            "that's good",
            "okay",
            "ok",
            "alright",
            "right,",
        )
        rhetorical_question_starts = (
            "what do you think",
            "any questions",
            "any question",
            "questions from",
        )

        for part in re.split(r"(?<=[.!?])\s+", key_points_raw):
            cleaned = part.strip().rstrip(".").strip()
            if not cleaned:
                continue
            words = cleaned.split()
            if len(words) < 4:
                continue

            lower = cleaned.lower()
            # Drop obvious rhetorical questions like "What do you think?"
            if cleaned.endswith("?") and lower.startswith(rhetorical_question_starts):
                continue
            # Drop pure approval phrases like "That sounds perfect, Gokul"
            if lower.startswith(approval_starts):
                continue

            key_points.append(cleaned)

    # Decisions
    decisions = extract_decisions(overall_text)

    # Turn action sentences into simple "Name: task" style when we can
    action_items: list[str] = []
    for s in actions:
        sent = s.strip()
        lower = sent.lower()

        # Pattern: "Akash can handle the technical part"
        m_handle = re.search(r"\b([A-Z][a-z]+)\s+can handle\s+(.*)", sent)
        if m_handle:
            name = m_handle.group(1)
            task = m_handle.group(2).rstrip(". ")
            action_items.append(f"{name}: {task}")
            continue

        # Pattern: "Arunima is good at designing the slides"
        m_good_at = re.search(r"\b([A-Z][a-z]+)\s+is good at\s+(.*)", sent)
        if m_good_at:
            name = m_good_at.group(1)
            task = m_good_at.group(2).rstrip(". ")
            action_items.append(f"{name}: {task}")
            continue

        # Generic "Name will ..." pattern
        m_will = re.search(r"\b([A-Z][a-z]+)\s+will\s+(.*)", sent)
        if m_will:
            name = m_will.group(1)
            task = m_will.group(2).rstrip(". ")
            action_items.append(f"{name}: {task}")
            continue

        # Fallback: keep as plain bullet
        action_items.append(sent.rstrip("."))

    # Fallback if no actions extracted at all
    if not action_items:
        action_items = []

    # Build final string
    lines: list[str] = []
    lines.append(f"Date: {date if date != 'Not found' else 'Not Mentioned'}")
    lines.append(f"Time: {time if time != 'Not found' else 'Not Mentioned'}")
    lines.append(f"Location: {location if location != 'Not found' else 'Not Mentioned'}")

    # Attendees
    if attendees:
        lines.append("Attendees:")
        for name in attendees:
            lines.append(f"- {name}")
    else:
        lines.append("Attendees: Not Mentioned")

    lines.append("")  # blank line

    # Agenda
    lines.append("Agenda:")
    lines.append(agenda if agenda != "Not found" else "Not Mentioned")
    lines.append("")

    # Meeting minutes
    lines.append("MEETING MINUTES:")
    if key_points:
        for p in key_points:
            lines.append(f"- {p}")
    else:
        lines.append("- Not available")
    lines.append("")

    # Decisions
    lines.append("DECISIONS:")
    if decisions:
        for d in decisions:
            lines.append(f"- {d}")
    else:
        lines.append("- Not explicitly stated")
    lines.append("")

    # Action items
    lines.append("ACTION ITEMS:")
    if action_items:
        for a in action_items:
            lines.append(f"- {a}")
    else:
        lines.append("- None captured")

    return "\n".join(lines)

# ------------------ MAIN PIPELINE ------------------
def process_meeting(audio_path, n_speakers=DEFAULT_SPEAKERS, use_llm: bool = False, gemini_api_key=None):
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

    # 6. Merge speaker text with better spacing
    for sp in speaker_text:
        # Join segments with space, ensuring proper sentence boundaries
        merged = " ".join(speaker_text[sp])
        # Clean up multiple spaces
        merged = re.sub(r'\s+', ' ', merged).strip()
        speaker_text[sp] = merged

    # 7. Summaries
    overall_text = " ".join(speaker_text.values())
    overall_text = re.sub(r'\s+', ' ', overall_text).strip()
    # Use slightly shorter summaries so we focus on purpose instead of full dialogue
    overall_summary = summarize(overall_text, 4)
    speaker_summaries = {sp: summarize(txt, 2) for sp, txt in speaker_text.items()}

    # 7b. Convert speaker summaries to reported speech (if LLM enabled)
    if use_llm and has_gemini(gemini_api_key):
        speaker_summaries = convert_speaker_summaries_to_reported(
            speaker_summaries, api_key=gemini_api_key
        )

    # 8. Action items
    actions = extract_actions(overall_text)

    # 9. Extract metadata
    metadata = extract_metadata(overall_text)

    # 10. Build structured minutes text
    minutes_text = ""
    gemini_error = None
    if use_llm and has_gemini(gemini_api_key):
        # Build transcript with speaker labels so LLM can convert dialogue → "what happened"
        transcript_with_speakers = "\n\n".join(
            f"{speaker}:\n{text}" for speaker, text in speaker_text.items() if text
        ).strip() or overall_text
        minutes_text, gemini_error = generate_minutes_with_gemini(
            transcript_with_speakers, metadata, api_key=gemini_api_key
        )

    # Fallback to rule-based minutes if LLM is disabled or failed/empty
    if not minutes_text:
        minutes_text = build_structured_minutes(metadata, overall_text, actions)

    return overall_summary, speaker_summaries, actions, metadata, minutes_text, gemini_error

# ------------------ RUN SCRIPT ------------------
if __name__ == "__main__":

    audio_path = "uploads/meeting.wav"

    if not os.path.exists(audio_path):
        print("❌ Audio file not found:", audio_path)
        exit(1)

    try:
        overall, speaker_summaries, actions, metadata, minutes_text, gemini_error = process_meeting(audio_path)
    except Exception as e:
        print("❌ Error processing meeting:", e)
        exit(1)

    if gemini_error:
        print("⚠️ Gemini API failed (rule-based minutes used):", gemini_error)

    print("\n=============== METADATA ===============\n")
    for key, value in metadata.items():
        if isinstance(value, list):
            print(f"{key}: {', '.join(value) if value else 'Not found'}")
        else:
            print(f"{key}: {value}")

    print("\n=============== MEETING MINUTES (STRUCTURED) ===============\n")
    print(minutes_text)

    print("\n👥 SPEAKER-WISE SUMMARY (RAW EXTRACTIVE):\n")
    for sp, txt in speaker_summaries.items():
        print(f"{sp}:")
        print(txt)
        print()