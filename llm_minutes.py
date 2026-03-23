import logging
from typing import Dict, Optional

from config import GEMINI_API_KEY, GEMINI_MODEL

# Prefer new Google Gen AI SDK (google-genai). Fall back to deprecated google.generativeai.
_genai_new = None
_genai_old = None
try:
    from google import genai as _genai_new
except ImportError:
    _genai_new = None
try:
    import google.generativeai as _genai_old
except ImportError:
    _genai_old = None
_using_new_sdk = _genai_new is not None


def has_gemini(api_key: Optional[str] = None) -> bool:
    """Return True if Gemini is configured and a client library is available."""
    key = (api_key or GEMINI_API_KEY) if api_key is not None else GEMINI_API_KEY
    if not key or not str(key).strip():
        return False
    return _using_new_sdk or _genai_old is not None


def _get_api_key(api_key: Optional[str]) -> str:
    """Return the API key to use (passed-in or from config)."""
    return (api_key or GEMINI_API_KEY or "").strip()


def generate_minutes_with_gemini(transcript: str, metadata: Dict, api_key: Optional[str] = None):
    """
    Use Google Gemini to generate high-quality, structured meeting minutes.

    Returns (minutes_text, error_message). On success error_message is None.
    api_key: optional override; if not set, uses GEMINI_API_KEY from environment/.env.
    """
    if not has_gemini(api_key):
        msg = "Gemini not available: no API key or install google-genai (or google-generativeai)."
        logging.warning(msg)
        return "", msg

    key = _get_api_key(api_key)
    if not key:
        return "", "API key is empty."

    date = metadata.get("Date", "Not Mentioned")
    time = metadata.get("Time", "Not Mentioned")
    location = metadata.get("Location", "Not Mentioned")
    attendees = metadata.get("Attendees") or []
    agenda = metadata.get("Agenda", "Not Mentioned")
    attendees_hint = ", ".join(attendees) if attendees else "Not Mentioned"

    user_prompt = f"""
Convert the following meeting transcript into formal meeting MINUTES (what HAPPENED in the meeting), not a transcript of who said what.

CRITICAL: Write what HAPPENED — outcomes, decisions, discussion points, agreements. Do NOT list "Speaker 1 said X, Speaker 2 said Y". Do NOT quote speakers. Convert dialogue into third-person / outcome style (e.g. "The team agreed to...", "It was decided that...", "Discussion covered...").

Rules:
1. MEETING MINUTES section = what happened (topics discussed, agreements, outcomes). Not "X said ..." — convert to "The group discussed...", "It was agreed that...", etc.
2. Do NOT quote or paraphrase each speaker; summarize the substance.
3. Write in clear bullet points.
4. Create proper Action Items with assigned names where possible.
5. If Date/Time/Location are not mentioned, write "Not Mentioned".
6. If Attendees are not clear, use "Not Mentioned".
7. Keep the tone neutral and professional.

Use EXACTLY this format:

Date: <date or Not Mentioned>
Time: <time or Not Mentioned>
Location: <location or Not Mentioned>
Attendees:
- Name 1
- Name 2

Agenda:
- Point 1
- Point 2

MEETING MINUTES:
- Point 1
- Point 2

DECISIONS:
- Decision 1

ACTION ITEMS:
- Name: Task

Example (convert speaker dialogue → what happened):

Transcript:
"Speaker 1: Let's build a website. Speaker 2: I will do the design."

Wrong (do NOT do this): "Speaker 1 suggested building a website. Speaker 2 said she would design."
Correct (what HAPPENED):
MEETING MINUTES:
- Discussed building a new website.
- Design responsibility was assigned.

ACTION ITEMS:
- Speaker 2: Design the website.

Now convert the transcript below into minutes in this style (what happened, not who said what).

Known metadata (may be incomplete):
Date: {date}
Time: {time}
Location: {location}
Attendees (from audio analysis): {attendees_hint}
Agenda (from audio analysis): {agenda}

Transcript:
\"\"\"{transcript.strip()}\"\"\"
"""

    try:
        if _using_new_sdk:
            client = _genai_new.Client(api_key=key)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
            )
            text = (getattr(response, "text", None) or "").strip()
        else:
            _genai_old.configure(api_key=key)
            model = _genai_old.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(user_prompt)
            text = (response.text or "").strip()

        if not text:
            logging.warning("Gemini returned empty text; falling back to rule-based minutes.")
            return "", "Gemini returned empty response."
        return text, None
    except Exception as e:
        logging.error("Error calling Gemini API: %s", e)
        return "", str(e)


def convert_speaker_summaries_to_reported(
    speaker_summaries: Dict[str, str],
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Convert speaker-wise summaries from direct speech to reported (third-person) speech
    using Gemini. Returns the original summaries unchanged on failure.
    """
    if not speaker_summaries or not has_gemini(api_key):
        return speaker_summaries

    key = _get_api_key(api_key)
    if not key:
        return speaker_summaries

    # Build a single prompt with all speakers
    speaker_block = "\n".join(
        f"{speaker}: {text}" for speaker, text in speaker_summaries.items() if text
    )

    prompt = f"""Convert the following speaker-wise meeting summaries from DIRECT speech to REPORTED (third-person) speech.

Rules:
1. Convert first-person ("I will...", "I can...") to third-person ("Offered to...", "Volunteered to...").
2. Convert questions ("Have you decided...?") to reported form ("Asked whether the team had decided...").
3. Keep it concise — short bullet-style sentences, no unnecessary words.
4. Do NOT add new information. Only rephrase what is already there.
5. Return EXACTLY the same speaker labels (e.g. "Speaker 1:", "Speaker 2:") followed by the converted text.
6. Do NOT include any extra commentary, headings, or markdown formatting.

Input:
{speaker_block}

Output (same format, reported speech):"""

    try:
        if _using_new_sdk:
            client = _genai_new.Client(api_key=key)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            text = (getattr(response, "text", None) or "").strip()
        else:
            _genai_old.configure(api_key=key)
            model = _genai_old.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            text = (response.text or "").strip()

        if not text:
            return speaker_summaries

        # Parse the response back into a dict
        import re
        converted = {}
        # Match lines like "Speaker 1: ..." or "Speaker 2: ..."
        pattern = re.compile(r"^(Speaker\s+\d+)\s*:\s*(.+)", re.MULTILINE)
        for match in pattern.finditer(text):
            label = match.group(1).strip()
            content = match.group(2).strip()
            converted[label] = content

        # Only use converted if we got results for all speakers
        if len(converted) >= len(speaker_summaries):
            return converted
        else:
            # Partial match — merge with originals
            result = dict(speaker_summaries)
            result.update(converted)
            return result

    except Exception as e:
        logging.error("Error converting speaker summaries to reported speech: %s", e)
        return speaker_summaries
