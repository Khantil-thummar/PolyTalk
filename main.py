import os
import wave
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

chunk_count = 0
os.makedirs("chunks", exist_ok=True)

@app.get("/")
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()

            filename = f"chunks/chunk_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.wav"
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(data)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
