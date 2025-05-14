import os
import wave
import io
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

combine_chunks = None   # Set None to disable combining
SAVE_CHUNKS = False   # Set to False to avoid saving to disk


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
            print("Len of buffer:", len(buffer))

            if combine_chunks is None:
                wav_bytes = convert_chunk_to_wav(data, suffix=None)
                buffer.clear()
                chunk_counter = 0
                continue

            if chunk_counter >= combine_chunks:
                wav_bytes = convert_chunk_to_wav(buffer, suffix=file_counter)
                buffer.clear()
                chunk_counter = 0
                file_counter += 1
            

    except WebSocketDisconnect:
        if combine_chunks is not None and buffer:
            wav_bytes = convert_chunk_to_wav(buffer, suffix=file_counter)
            # Handle last partial chunk if needed
        print("WebSocket disconnected")


def convert_chunk_to_wav(data: bytes, suffix=None) -> bytes:
    filename = (
        f"chunks/chunk_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.wav"
        if suffix is None
        else f"chunks/combined_chunk_{suffix}.wav"
    )

    # Write to in-memory buffer
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(data)
    wav_data = buffer.getvalue()
    if SAVE_CHUNKS:
        with open(filename, 'wb') as f:
            f.write(wav_data)
        print(f"Saved: {filename}")
    else:
        print(f"Processed (not saved): {len(wav_data)} bytes")

    return wav_data
