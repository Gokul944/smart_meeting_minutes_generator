"""
Quick test to verify your Gemini API key works.
Run from project folder:  python test_gemini_api.py

Uses the new google-genai package if installed; otherwise google-generativeai.
Install:  pip install google-genai
"""
import sys

# Load config (and .env if python-dotenv is installed)
from config import GEMINI_API_KEY, GEMINI_MODEL

# Prefer new SDK (google-genai)
try:
    from google import genai
    _use_new = True
except ImportError:
    _use_new = False
    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: No Gemini SDK installed.")
        print("  Run:  pip install google-genai")
        print("  (Or:  pip install google-generativeai  for the older SDK)")
        sys.exit(1)


def main():
    key = (GEMINI_API_KEY or "").strip()
    if not key:
        print("ERROR: No API key found.")
        print("  - Put GEMINI_API_KEY=your_key in the .env file in this folder, or")
        print("  - Set environment variable:  set GEMINI_API_KEY=your_key  (Windows)")
        print("  - Get a key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    print("API key found (length {}). Model: {}. Testing connection...".format(len(key), GEMINI_MODEL))

    try:
        if _use_new:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents="Reply with exactly: API is working",
            )
            text = (getattr(response, "text", None) or "").strip()
        else:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content("Reply with exactly: API is working")
            text = (response.text or "").strip()

        if "API is working" in text or text:
            print("SUCCESS: Gemini API is working.")
            print("Response:", (text or "")[:100])
        else:
            print("WARNING: API responded but with unexpected content:", repr(text))
    except Exception as e:
        print("FAILED: Gemini API error:")
        print(" ", type(e).__name__, ":", e)
        if "API_KEY_INVALID" in str(e) or "invalid" in str(e).lower():
            print("\n  Check that your key is correct and active at https://aistudio.google.com/apikey")
        if "429" in str(e) or "quota" in str(e).lower():
            print("\n  Quota exceeded. Wait or check https://ai.dev/rate-limit")
        if "404" in str(e) or "not found" in str(e).lower():
            print("\n  Model not found. Set GEMINI_MODEL in .env (e.g. gemini-2.0-flash, gemini-2.5-flash).")
        sys.exit(1)


if __name__ == "__main__":
    main()
