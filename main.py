import os
import wave
import io
import base64
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
    
    # Flag to track if the WebSocket is open
    is_connected = True

    try:
        while is_connected:
            try:
                data = await websocket.receive_bytes()
                chunk_counter += 1
                buffer.extend(data)
                
                if combine_chunks is None:
                    await convert_chunk_to_wav(data, suffix=None, websocket=websocket)
                    buffer.clear()
                    chunk_counter = 0
                    continue

                if chunk_counter >= combine_chunks:
                    await convert_chunk_to_wav(buffer, suffix=file_counter, websocket=websocket)
                    buffer.clear()
                    chunk_counter = 0
                    file_counter += 1
            except WebSocketDisconnect:
                is_connected = False
                print("WebSocket disconnected while receiving data")
                break

    except Exception as e:
        print(f"Error in WebSocket handler: {e}")
    
    finally:
        if buffer and is_connected:
            try:
                await convert_chunk_to_wav(buffer, suffix=file_counter, websocket=websocket)
            except Exception as e:
                print(f"Error processing final chunk: {e}")
        print("WebSocket connection closed")



async def translate_and_save_wav(original_filepath: str, output_filename: str, websocket: WebSocket = None):
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
    
    # Send translated audio to the client if websocket is provided
    if websocket:
        try:
            # Read the saved audio file and encode it as base64
            with open(output_path, "rb") as audio_file:
                audio_data = audio_file.read()
                base64_audio = base64.b64encode(audio_data).decode('utf-8')
                
                # Send audio data to client
                await websocket.send_json({
                    "type": "translated_audio",
                    "data": base64_audio,
                    "format": "wav"
                })
        except WebSocketDisconnect:
            print("WebSocket disconnected during audio transmission")
        except RuntimeError as e:
            if "after sending 'websocket.close'" in str(e):
                print("WebSocket already closed, cannot send audio")
            else:
                raise
            
    return translated_audio

async def convert_chunk_to_wav(data: bytes, suffix=None, websocket: WebSocket = None):
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
    translated_audio = None

    if SAVE_CHUNKS:
        with open(filename, 'wb') as f:
            f.write(wav_data)
        print(f"Saved: {filename}")

        # Translate and save translated audio
        translated_name = os.path.basename(filename).replace("chunk", "translated")
        translated_audio = await translate_and_save_wav(filename, translated_name, websocket)

    else:
        print(f"Processed (not saved): {len(wav_data)} bytes")

    return wav_data
