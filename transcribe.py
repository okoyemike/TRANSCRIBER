from faster_whisper import WhisperModel

# 1. Load the model
model = WhisperModel("small", device="cpu", compute_type="int8")

# 2. Give it the audio file
segments, info = model.transcribe("test.m4a")

# 3. Print the result
print("Detected language:", info.language)
for segment in segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")