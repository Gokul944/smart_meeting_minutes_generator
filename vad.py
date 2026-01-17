import webrtcvad
import numpy as np
import os
from pydub import AudioSegment
from config import SEGMENT_DIR, SAMPLE_RATE, MIN_SPEECH_SEC


def vad_split(wav_path):
    vad = webrtcvad.Vad(2)  # 0–3 (2 = balanced)

    # Load & normalize audio
    audio = (
        AudioSegment.from_file(wav_path)
        .set_channels(1)
        .set_frame_rate(SAMPLE_RATE)
    )

    samples = np.array(audio.get_array_of_samples())

    frame_size = int(SAMPLE_RATE * 0.03)  # 30 ms
    segments = []
    start = None

    # Ensure clean segments folder
    os.makedirs(SEGMENT_DIR, exist_ok=True)
    for f in os.listdir(SEGMENT_DIR):
        if f.endswith(".wav"):
            os.remove(os.path.join(SEGMENT_DIR, f))

    for i in range(0, len(samples), frame_size):
        frame = samples[i:i + frame_size]
        if len(frame) < frame_size:
            continue

        is_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)

        if is_speech and start is None:
            start = i

        elif not is_speech and start is not None:
            end = i
            duration = (end - start) / SAMPLE_RATE

            if duration >= MIN_SPEECH_SEC:
                # Convert samples → milliseconds
                start_ms = int(start / SAMPLE_RATE * 1000)
                end_ms = int(end / SAMPLE_RATE * 1000)

                seg_audio = audio[start_ms:end_ms]
                seg_path = os.path.join(
                    SEGMENT_DIR, f"seg_{len(segments)}.wav"
                )
                seg_audio.export(seg_path, format="wav")
                segments.append(seg_path)

            start = None

    # Handle trailing speech (important!)
    if start is not None:
        end = len(samples)
        duration = (end - start) / SAMPLE_RATE

        if duration >= MIN_SPEECH_SEC:
            start_ms = int(start / SAMPLE_RATE * 1000)
            end_ms = int(end / SAMPLE_RATE * 1000)

            seg_audio = audio[start_ms:end_ms]
            seg_path = os.path.join(
                SEGMENT_DIR, f"seg_{len(segments)}.wav"
            )
            seg_audio.export(seg_path, format="wav")
            segments.append(seg_path)

    return segments
