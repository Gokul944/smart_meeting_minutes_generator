from stt import transcribe

text = transcribe("uploads/normalized.wav")
print("TRANSCRIPTION:")
print(text)
