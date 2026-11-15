"""Standalone microphone transcription test using faster-whisper.

Install: python -m pip install faster-whisper sounddevice numpy
Run:     python live_whisper_test.py
Stop:    Ctrl+C
"""

import queue
import sys
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 3
MODEL_NAME = "tiny.en"


def main():
    print(f"Loading Whisper model {MODEL_NAME!r} (first run may download it)...", flush=True)
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Audio warning: {status}", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    samples_per_segment = SAMPLE_RATE * SEGMENT_SECONDS
    buffered = np.empty(0, dtype=np.float32)
    segment_number = 0

    print("\nMicrophone active. Speak normally. Press Ctrl+C to stop.\n", flush=True)
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
            while True:
                chunk = audio_queue.get()
                buffered = np.concatenate((buffered, chunk))
                while len(buffered) >= samples_per_segment:
                    audio = buffered[:samples_per_segment]
                    buffered = buffered[samples_per_segment:]
                    segment_number += 1
                    started = time.monotonic()
                    segments, _ = model.transcribe(
                        audio,
                        language="en",
                        beam_size=1,
                        vad_filter=False,
                        condition_on_previous_text=False,
                    )
                    transcript = " ".join(segment.text.strip() for segment in segments).strip()
                    elapsed = time.monotonic() - started
                    print(f"[{segment_number * SEGMENT_SECONDS:>3}s] {transcript or '(no speech detected)'}  (processed in {elapsed:.1f}s)", flush=True)
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    except sd.PortAudioError as exc:
        print(f"\nMicrophone error: {exc}")
        print("Check Windows microphone permissions and the selected default input device.")
        sys.exit(1)


if __name__ == "__main__":
    main()
