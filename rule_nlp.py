import re
from collections import Counter
from typing import List, Tuple


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "for", "of", "to", "with",
    "is", "are", "was", "were", "be", "being", "been", "this", "that", "these", "those",
    "it", "its", "as", "by", "from", "we", "you", "i", "they", "he", "she", "our", "their",
    "his", "her", "there", "here", "so", "just", "very", "really", "then", "now", "well",
}

_FILLER_STARTS = (
    "thank you", "thanks everyone", "thanks, everyone", "i'll start again", "i will start again",
    "um", "uh", "well,", "okay,", "ok,", "so,", "alright,", "right,", "yeah,",
    "excuse me", "excuse",
)

# Procedural phrases that indicate meeting flow, NOT action items
_PROCEDURAL_PATTERNS = [
    r"then we'll go to", r"then we will go to", r"we'll go to", r"we will go to",
    r"next will be", r"next is", r"moving on to", r"let's move to",
    r"going to move", r"i'm going to come", r"i'm going to move",
    r"we'll go into", r"we will go into", r"then we'll go", r"then we will go",
    r"going to approval", r"go to approval", r"finish a business", r"nothing else on",
    r"go to the proof", r"go to and", r"starting at exactly",
]

# Real action keywords - must be followed by actual tasks
_REAL_ACTION_PATTERNS = [
    r"will (?:complete|finish|submit|send|review|update|create|implement|fix|address|resolve|provide|deliver|prepare|draft|finalize|bring|return)",
    r"should (?:complete|finish|submit|send|review|update|create|implement|fix|address|resolve|provide|deliver|prepare|draft|finalize)",
    r"need to (?:complete|finish|submit|send|review|update|create|implement|fix|address|resolve|provide|deliver|prepare|draft|finalize)",
    r"needs to (?:complete|finish|submit|send|review|update|create|implement|fix|address|resolve|provide|deliver|prepare|draft|finalize)",
    r"must (?:complete|finish|submit|send|review|update|create|implement|fix|address|resolve|provide|deliver|prepare|draft|finalize)",
    r"going to (?:complete|finish|submit|send|review|update|create|implement|fix|address|resolve|provide|deliver|prepare|draft|finalize)",
    r"plan to (?:complete|finish|submit|send|review|update|create|implement|fix|address|resolve|provide|deliver|prepare|draft|finalize)",
    r"decided to (?:complete|finish|submit|send|review|update|create|implement|fix|address|resolve|provide|deliver|prepare|draft|finalize)",
    # More generic patterns used in informal meetings
    r"\bcan handle\b",
    r"\bis good at\b",
    r"\bi can\b",
    r"\blet's\b",
    r"will bring.*back",
    r"will return.*to",
    r"action item",
    r"follow up",
    r"follow-up",
    r"next step",
    r"next steps",
    r"assign.*to",
    r"responsible for",
]


def _clean_text(text: str) -> str:
    """Clean and normalize text before processing."""
    # Fix transcription errors first
    text = re.sub(r'\btheió\b', 'the I-O', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d+)\s+(\d+)\s+ah\b', r'\1-\2', text)
    text = re.sub(r'\bgoing home\b', 'going to', text, flags=re.IGNORECASE)
    
    # Normalize spacing
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.!?,])', r'\1', text)
    text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)
    
    # Remove standalone "ah", "um", "uh" that are transcription artifacts
    text = re.sub(r'\b(ah|um|uh)\s+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+(ah|um|uh)\b', ' ', text, flags=re.IGNORECASE)
    
    return text.strip()


def _is_incomplete_sentence(sentence: str) -> bool:
    """Detect incomplete/fragmented sentences that shouldn't be in summaries."""
    lowered = sentence.lower().strip()
    tokens = sentence.split()
    
    # Too short
    if len(tokens) < 6:
        return True
    
    # Check for transcription errors that indicate incomplete sentences
    # Patterns like "going home after I did theió 3 2 ah"
    if re.search(r'\b\d+\s+\d+\s+ah\b', lowered) or re.search(r'\btheió\b', lowered):
        return True
    
    # Starts with incomplete verb forms (fragments) - be very aggressive
    incomplete_starts = [
        "define", "talking", "discussing", "mentioning", "referring",
        "places where", "updated so", "excuse me", "excuse",
        "going home",  # Common transcription error
    ]
    if any(lowered.startswith(start) for start in incomplete_starts):
        # Only allow if it's clearly a complete, long sentence
        if len(tokens) < 12 or not re.search(r'[.!?]$', sentence):
            return True
    
    # Has too many "um", "uh" patterns
    um_count = len(re.findall(r'\b(um|uh)\b', lowered))
    if um_count > len(tokens) * 0.15:  # More strict
        return True
    
    # Pattern: "word, word, word" without proper sentence structure
    comma_count = len(re.findall(r',', sentence))
    if comma_count > len(tokens) * 0.4:  # Too many commas relative to words
        # Check if it has verbs
        has_verb = re.search(r'\b(is|are|was|were|will|should|can|may|has|have|had|do|does|did)\b', lowered)
        if not has_verb:
            return True
    
    # Sentence that's just a list without verbs
    if re.match(r'^[\w\s,]+$', sentence):
        content_words = [w for w in tokens if w.lower() not in _STOPWORDS]
        if len(content_words) < 4:
            return True
        # Check for verb presence
        has_verb = re.search(r'\b(is|are|was|were|will|should|can|may|has|have|had|do|does|did|discuss|talk|decide|agree)\b', lowered)
        if not has_verb and len(tokens) < 10:
            return True
    
    # Check for question fragments that are incomplete
    if sentence.endswith('?') and len(tokens) < 8:
        if any(lowered.startswith(start) for start in ["any questions", "any question", "questions from"]):
            # Allow short questions if they're complete
            if len(tokens) < 6:
                return True
    
    return False


def _merge_fragments(sentences: List[str]) -> List[str]:
    """Intelligently merge related sentence fragments."""
    if not sentences:
        return []
    
    merged = []
    i = 0
    
    while i < len(sentences):
        current = sentences[i].strip()
        
        # Try to merge with next sentence if current is incomplete
        if i + 1 < len(sentences):
            next_sent = sentences[i + 1].strip()
            
            # Merge if current doesn't end properly and next continues the thought
            if (not current.endswith(('.', '!', '?')) or 
                _is_incomplete_sentence(current) or
                len(current.split()) < 8):
                
                # Check if merging makes sense
                combined = f"{current} {next_sent}".strip()
                
                # Don't merge if combined would be too long
                if len(combined.split()) <= 40:
                    merged.append(combined)
                    i += 2
                    continue
        
        merged.append(current)
        i += 1
    
    return merged


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences with intelligent fragment handling."""
    text = _clean_text(text)
    if not text:
        return []
    
    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', text)
    sentences = [p.strip() for p in parts if p.strip() and len(p.strip().split()) >= 3]
    
    # Merge fragments intelligently
    sentences = _merge_fragments(sentences)
    
    # Filter out incomplete sentences
    complete_sentences = []
    for s in sentences:
        if not _is_incomplete_sentence(s):
            complete_sentences.append(s)
    
    return complete_sentences if complete_sentences else sentences


def _sentence_tokens(sentence: str) -> List[str]:
    """Extract word tokens from sentence."""
    return re.findall(r'\b\w+\b', sentence.lower())


def _is_procedural(sentence: str) -> bool:
    """Check if sentence is procedural flow, not meaningful content."""
    lowered = " " + sentence.lower() + " "
    return any(re.search(pattern, lowered) for pattern in _PROCEDURAL_PATTERNS)


def _is_filler(sentence: str) -> bool:
    """Check if sentence is filler/greeting."""
    lowered = sentence.lower().lstrip()
    return lowered.startswith(_FILLER_STARTS)


def _sentence_quality_score(sentence: str) -> float:
    """Score sentence quality (higher = better)."""
    tokens = _sentence_tokens(sentence)
    content_words = [w for w in tokens if w not in _STOPWORDS]
    
    # Must have sufficient content
    if len(content_words) < 6:
        return 0.0
    
    # Must have verbs (indicates complete thought)
    verbs = [
        "is", "are", "was", "were", "will", "should", "can", "may", "must",
        "discuss", "discussed", "talk", "talked", "decide", "decided",
        "agree", "agreed", "propose", "proposed", "suggest", "suggested",
        "update", "updated", "change", "changed", "add", "added", "remove", "removed",
        "has", "have", "had", "do", "does", "did", "make", "made", "take", "took",
        "see", "saw", "know", "knew", "think", "thought", "say", "said", "tell", "told",
    ]
    has_verb = any(w in tokens for w in verbs)
    
    if not has_verb:
        # Without verbs, sentence is likely incomplete
        if len(tokens) < 15:
            return 0.05  # Very low score
        return 0.2  # Still low even if long
    
    # Penalize very long sentences (likely run-ons)
    if len(tokens) > 50:
        return 0.3
    
    # Penalize excessive filler words
    filler_words = ["then", "so", "well", "now", "just", "um", "uh", "like", "you know"]
    filler_count = sum(1 for w in tokens if w in filler_words)
    if filler_count > len(tokens) * 0.2:
        return 0.15
    
    # Penalize sentences that are mostly stopwords
    if len(content_words) / len(tokens) < 0.4:
        return 0.2
    
    # Content word density
    density = len(content_words) / len(tokens) if tokens else 0
    
    # Bonus for having proper sentence structure
    structure_bonus = 1.0
    if not sentence.endswith(('.', '!', '?')):
        structure_bonus = 0.6  # Lower bonus for incomplete punctuation
    
    # Bonus for having both subject and verb indicators
    has_subject_indicators = any(w in tokens for w in ["we", "they", "it", "this", "that", "the", "a"])
    if has_subject_indicators and has_verb:
        structure_bonus *= 1.2
    
    return density * structure_bonus


def summarize(text: str, n: int = 5) -> str:
    """
    Improved extractive summarizer that filters fragmented transcriptions
    and returns only complete, meaningful sentences.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    
    # Filter and score sentences
    meaningful_sentences = []
    for s in sentences:
        # Skip procedural, filler, and incomplete sentences
        if _is_procedural(s) or _is_filler(s) or _is_incomplete_sentence(s):
            continue
        
        quality = _sentence_quality_score(s)
        if quality > 0.25:  # Higher threshold - only good quality sentences
            meaningful_sentences.append((s, quality))
    
    if not meaningful_sentences:
        # Fallback: be less strict but still filter incomplete
        for s in sentences:
            if not _is_procedural(s) and not _is_filler(s) and not _is_incomplete_sentence(s):
                quality = _sentence_quality_score(s)
                if quality > 0.15:  # Still require decent quality
                    meaningful_sentences.append((s, quality))
    
    if not meaningful_sentences:
        return ""
    
    # If we have fewer than n sentences, return all
    if len(meaningful_sentences) <= n:
        result = ". ".join(s.strip().rstrip(".") for s, _ in meaningful_sentences)
        return (result + ".").strip()
    
    # Build word frequency for scoring
    all_words = []
    for s, _ in meaningful_sentences:
        tokens = _sentence_tokens(s)
        all_words.extend([w for w in tokens if w not in _STOPWORDS])
    
    if not all_words:
        result = ". ".join(s.strip().rstrip(".") for s, _ in meaningful_sentences[:n])
        return (result + ".").strip()
    
    freq = Counter(all_words)
    
    # Score each sentence: combine quality score with word frequency
    scored: List[Tuple[int, float]] = []
    for idx, (s, quality) in enumerate(meaningful_sentences):
        tokens = _sentence_tokens(s)
        content = [w for w in tokens if w not in _STOPWORDS]
        
        # Frequency-based score
        freq_score = sum(freq.get(w, 0) for w in content) / len(content) if content else 0
        
        # Combined score: quality * (1 + freq_score)
        combined_score = quality * (1.0 + freq_score * 0.1)
        scored.append((idx, combined_score))
    
    # Get top n sentences, maintaining original order
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:n]
    chosen_indices = sorted(idx for idx, _ in top)
    chosen_sentences = [meaningful_sentences[i][0].strip().rstrip(".") for i in chosen_indices]
    
    result = ". ".join(chosen_sentences).strip()
    if not result.endswith("."):
        result += "."
    return result


def extract_actions(text: str) -> List[str]:
    """
    Extract REAL action items - commitments and tasks, NOT procedural statements.
    Much stricter filtering to avoid "then we'll go to X" type statements.
    """
    sentences = _split_sentences(text)
    actions = []
    
    for s in sentences:
        lowered = " " + s.lower().strip() + " "
        
        # Skip procedural statements, filler, and incomplete sentences
        if _is_procedural(s) or _is_filler(s) or _is_incomplete_sentence(s):
            continue
        
        # Must match REAL action patterns (commitments/tasks)
        is_action = any(re.search(pattern, lowered) for pattern in _REAL_ACTION_PATTERNS)
        
        if not is_action:
            continue
        
        # Quality checks
        tokens = _sentence_tokens(s)
        if len(tokens) < 8:  # Require minimum length
            continue
        
        # Must have content words
        content_words = [w for w in tokens if w not in _STOPWORDS]
        if len(content_words) < 5:
            continue
        
        # Must have a verb indicating action
        action_verbs = ["will", "should", "must", "need", "going", "plan", "decided", "bring", "return"]
        if not any(verb in tokens for verb in action_verbs):
            continue
        
        action = s.strip().rstrip(".")
        actions.append(action)
    
    # Remove duplicates
    unique_actions = []
    seen_lower = set()
    for action in actions:
        action_lower = action.lower()
        is_duplicate = False
        for seen in seen_lower:
            words1 = set(_sentence_tokens(action))
            words2 = set(_sentence_tokens(seen))
            if len(words1) > 0 and len(words2) > 0:
                overlap = len(words1 & words2) / max(len(words1), len(words2))
                if overlap > 0.7:
                    is_duplicate = True
                    break
        if not is_duplicate:
            unique_actions.append(action)
            seen_lower.add(action_lower)
    
    return unique_actions[:15]


def extract_decisions(text: str) -> List[str]:
    """
    Extract decision-like sentences, e.g. selections, agreements, conclusions.
    This is simpler and more permissive than extract_actions.
    """
    sentences = _split_sentences(text)
    decisions: List[str] = []

    decision_keywords = [
        "decided", "decide", "agreed", "agreed that", "concluded",
        "will be", "was chosen", "were chosen", "was selected",
        "were selected", "approved", "accepted", "confirmed",
        "resolved", "finalized", "will use", "will go with",
    ]

    for s in sentences:
        lowered = s.lower()

        if _is_procedural(s) or _is_filler(s) or _is_incomplete_sentence(s):
            continue

        if any(k in lowered for k in decision_keywords):
            decisions.append(s.strip().rstrip("."))

    # Deduplicate similar decisions
    unique_decisions: List[str] = []
    seen_lower: set[str] = set()
    for d in decisions:
        key = d.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        unique_decisions.append(d)

    return unique_decisions[:10]
