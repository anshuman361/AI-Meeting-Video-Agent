from faster_whisper import WhisperModel
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
_model = None

def load_model():
    global _model
    if _model is None:
        print("loading model...")
        _model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",          # Streamlit Cloud has no GPU
            compute_type="int8",   # fastest CPU quantization, minimal accuracy loss
        )
        print("Model loaded successfully")
    return _model

def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    model = load_model()
    task = "translate" if translate else "transcribe"
    segments, info = model.transcribe(chunk_path, task=task, beam_size=5)
    return " ".join(segment.text for segment in segments).strip()

def transcribe_all(chunks: list, translate: bool = False) -> str:
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}")
        text = transcribe_chunk(chunk, translate=translate)
        full_transcript += text + " "
    print("Transcription Completed")
    return full_transcript