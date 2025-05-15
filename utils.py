import os
import wave
import io
import base64
from datetime import datetime
import torchaudio
import torch
import soundfile as sf
from fastapi import WebSocket, WebSocketDisconnect

async def convert_chunk_to_wav(data: bytes, suffix=None, websocket: WebSocket = None, 
                               save_chunks=True, translate_func=None):
    """
    Convert audio data to WAV format and optionally translate it.
    
    Args:
        data: Raw audio bytes
        suffix: Optional suffix for the filename
        websocket: WebSocket connection to send the translated audio to
        save_chunks: Whether to save the audio chunks to disk
        translate_func: Function to use for translation
    """
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

    if save_chunks:
        os.makedirs("chunks", exist_ok=True)
        with open(filename, 'wb') as f:
            f.write(wav_data)
        print(f"Saved: {filename}")

        # Translate and save translated audio if translate_func is provided
        if translate_func:
            translated_name = os.path.basename(filename).replace("chunk", "translated")
            translated_audio = await translate_func(filename, translated_name, websocket)
    else:
        print(f"Processed (not saved): {len(wav_data)} bytes")

    return wav_data

async def send_audio_to_client(websocket: WebSocket, audio_path: str):
    """
    Send audio data to the client over WebSocket.
    
    Args:
        websocket: WebSocket connection to send the audio to
        audio_path: Path to the audio file to send
    """
    if not websocket:
        return
        
    try:
        # Read the saved audio file and encode it as base64
        with open(audio_path, "rb") as audio_file:
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