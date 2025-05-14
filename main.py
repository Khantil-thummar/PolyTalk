from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import wave
import os
from datetime import datetime
import os

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())

# Mount static folder for assets if needed


RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.wav")

    # Initialize WAV file
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(44100)

    try:
        while True:
            data = await websocket.receive_bytes()
            wf.writeframes(data)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        wf.close()
