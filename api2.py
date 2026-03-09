import io
import os
import re
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gtts import gTTS
from pydub import AudioSegment

import whisper

from main2 import chat


# ---------------------------
# FFmpeg Configuration
# ---------------------------

FFMPEG_PATH = r"C:\ffmpeg-8.0.1-essentials_build\bin"

AudioSegment.converter = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
AudioSegment.ffmpeg    = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
AudioSegment.ffprobe   = os.path.join(FFMPEG_PATH, "ffprobe.exe")


# ---------------------------
# Load Whisper Model
# ---------------------------

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper ready.")


# ---------------------------
# FastAPI App
# ---------------------------

app = FastAPI(title="LYA — Laayn AI Assistant", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------
# Models
# ---------------------------

class ChatRequest(BaseModel):
    message: str

class TTSRequest(BaseModel):
    text: str
    language: str = "en"


# ---------------------------
# Routes
# ---------------------------

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "assistant": "LYA", "model": "gemini-2.0-flash"}


# ---------------------------
# Chat
# ---------------------------

@app.post("/chat")
async def chat_endpoint(body: ChatRequest):

    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    answer = chat(body.message)

    return {"answer": answer}


# ---------------------------
# Text to Speech
# ---------------------------

@app.post("/tts")
async def tts_endpoint(body: TTSRequest):

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    clean = clean_for_tts(body.text)

    try:
        tts = gTTS(text=clean, lang=body.language, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)

        return Response(
            content=buf.read(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=lya.mp3",
                "Cache-Control": "no-cache",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# Speech to Text (Whisper)
# ---------------------------

@app.post("/stt")
async def stt_endpoint(audio: UploadFile = File(...)):

    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio empty")

    if len(audio_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio too large")

    tmp_input = None
    tmp_wav = None

    try:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(audio_bytes)
            tmp_input = f.name

        tmp_wav = tmp_input + ".wav"
        audio_seg = AudioSegment.from_file(tmp_input)
        audio_seg = audio_seg.set_channels(1).set_frame_rate(16000)
        audio_seg.export(tmp_wav, format="wav")

        result = whisper_model.transcribe(tmp_wav)
        text = result["text"].strip()

        return {"text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT error: {str(e)}")

    finally:
        for p in [tmp_input, tmp_wav]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass


# ---------------------------
# Helpers
# ---------------------------

def clean_for_tts(text: str) -> str:
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    return text.strip()


# ---------------------------
# Run Server
# ---------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=9000, reload=True)