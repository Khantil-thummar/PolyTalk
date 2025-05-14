import os
import wave
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())

# Change this to control how many chunks to combine
combine_chunks = 2  # Set to None for no combining

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    os.makedirs("chunks", exist_ok=True)

    buffer = bytearray()
    chunk_counter = 0
    file_counter = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk_counter += 1
            buffer.extend(data)

            # If combining is disabled, save each chunk directly
            if combine_chunks is None:
                save_chunk(data, suffix=None)
                continue

            # Combine chunks until the threshold
            if chunk_counter >= combine_chunks:
                save_chunk(buffer, suffix=file_counter)
                buffer.clear()
                chunk_counter = 0
                file_counter += 1

    except WebSocketDisconnect:
        # Save any remaining data if connection closes mid-batch
        if combine_chunks is not None and buffer:
            save_chunk(buffer, suffix=file_counter)
        print("WebSocket disconnected")


def save_chunk(data: bytes, suffix=None):
    filename = (
        f"chunks/chunk_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.wav"
        if suffix is None
        else f"chunks/combined_chunk_{suffix}.wav"
    )

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(data)

    print(f"Saved: {filename}")
