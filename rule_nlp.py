def summarize(text, n=5):
    sentences = text.split(".")
    return ". ".join(sentences[:n]).strip()

def extract_actions(text):
    keywords = ["will", "should", "need to", "must"]
    return [
        s.strip()
        for s in text.split(".")
        if any(k in s.lower() for k in keywords)
    ]
