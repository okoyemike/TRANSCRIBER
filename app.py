import os
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
from faster_whisper import WhisperModel

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading model (the first time, it downloads about 1.5 GB)...")
import os
model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=os.cpu_count())
print("Model ready.")

PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Audio Transcriber</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 16px; }
    button { padding: 8px 16px; font-size: 15px; }
    pre { background: #f4f4f4; padding: 12px; white-space: pre-wrap; border-radius: 6px; }
    .error { color: #b00020; }
  </style>
</head>
<body>
  <h1>Audio Transcriber</h1>
  <form method="post" enctype="multipart/form-data"
        onsubmit="document.getElementById('btn').disabled = true;
                  document.getElementById('btn').innerText = 'Transcribing... please wait';">
    <input type="file" name="audio" accept="audio/*,.m4a,.mp3,.wav">
    <button id="btn" type="submit">Transcribe</button>
  </form>

  {% if error %}<p class="error">{{ error }}</p>{% endif %}

  {% if transcript %}
    <h2>Transcript</h2>
    <pre>{{ transcript }}</pre>
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    transcript = None
    error = None

    if request.method == "POST":
        file = request.files.get("audio")
        if not file or file.filename == "":
            error = "Please choose an audio file first."
        else:
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            segments, info = model.transcribe(path, beam_size=1, vad_filter=True, language="en")
            lines = [f"[{s.start:.1f}s - {s.end:.1f}s] {s.text.strip()}" for s in segments]
            transcript = "\n".join(lines)

    return render_template_string(PAGE, transcript=transcript, error=error)


if __name__ == "__main__":
    app.run(debug=False)