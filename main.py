import os
import wave
import io
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio
import torch
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(device)
TARGET_LANG = "hin"


os.makedirs("converted_chunks", exist_ok=True)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())

combine_chunks = 2   # Set None to disable combining
SAVE_CHUNKS = True   # Set to False to avoid saving to disk


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



def translate_and_save_wav(original_filepath: str, output_filename: str):
    audio, orig_freq = torchaudio.load(original_filepath)
    audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000)

    # Ensure mono channel
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    audio_inputs = processor(audios=audio, return_tensors="pt").to(device)
    translated_audio = model.generate(**audio_inputs, tgt_lang=TARGET_LANG)[0].cpu().numpy().squeeze()

    sample_rate = model.config.sampling_rate
    output_path = os.path.join("converted_chunks", output_filename)
    sf.write(output_path, translated_audio, sample_rate)
    print(f"Translated and saved: {output_path}")

def convert_chunk_to_wav(data: bytes, suffix=None) -> bytes:
    filename = (
        f"chunks/chunk_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.wav"
        if suffix is None
        else f"chunks/combined_chunk_{suffix}.wav"
    )

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

        # Translate and save translated audio
        translated_name = os.path.basename(filename).replace("chunk", "translated")
        translate_and_save_wav(filename, translated_name)

    else:
        print(f"Processed (not saved): {len(wav_data)} bytes")

    return wav_data
