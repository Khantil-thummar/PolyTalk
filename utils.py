import os
import wave
import io
from datetime import datetime
import torchaudio
import torch
import soundfile as sf

async def convert_chunk_to_wav(data: bytes, suffix=None, save_chunks=True, translate_func=None):
    """
    Convert audio data to WAV format and optionally translate it.
    
    Args:
        data: Raw audio bytes
        suffix: Optional suffix for the filename
        save_chunks: (unused, kept for compatibility)
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

    # Always save the audio chunk to disk
    os.makedirs("chunks", exist_ok=True)
    with open(filename, 'wb') as f:
        f.write(wav_data)
    print(f"Saved: {filename}")

    # Always translate the audio using the saved file if translate_func is provided
    translated_audio = None
    if translate_func:
        translated_name = os.path.basename(filename).replace("chunk", "translated")
        translated_audio = await translate_func(filename, translated_name)

    return wav_data 