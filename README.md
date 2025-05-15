# Real-time Voice Translator

A real-time voice translation application that converts spoken language to a target language using Meta's Seamless M4T v2 model.

## Overview

This application captures audio from a user's microphone through a web browser, streams it to a FastAPI backend server, processes it through Meta's Seamless M4T v2 model for translation, and then returns the translated audio to be played in the browser.

## Features

- Real-time audio capture from browser
- Streaming audio processing
- Translation to target language (default: Hindi)
- Sequential playback of translated audio
- Robust error handling and connection management

## Project Structure

```
.
├── config.py                # Application configuration
├── main.py                  # Main FastAPI application
├── translator.py            # Translation module
├── utils.py                 # Utility functions
├── websocket_handler.py     # WebSocket connection management
└── static/                  # Static web files
    └── index.html          # Web interface
```

## Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch
- FastAPI
- Torchaudio
- Transformers
- SoundFile

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd voice-translator
   ```

2. Install dependencies:
   ```
   pip install fastapi uvicorn torch torchaudio transformers soundfile
   ```

3. Run the application:
   ```
   python main.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:8000/
   ```

## Configuration

You can customize the application behavior by modifying the `config.py` file:

- `MODEL_NAME`: The HuggingFace model to use for translation
- `TARGET_LANGUAGE`: Target language for translation (language code)
- `COMBINE_CHUNKS`: Number of audio chunks to combine before processing
- `SAVE_CHUNKS`: Whether to save audio chunks to disk

## Usage

1. Click "Start Recording" to begin capturing audio from your microphone
2. Speak clearly into your microphone
3. Your speech will be translated to the target language
4. The translated audio will play automatically in your browser
5. Click "Stop Recording" to end the session

## How It Works

1. The frontend captures audio using the WebAudio API
2. Audio is chunked and sent to the backend via WebSocket
3. The backend combines chunks, converts to WAV format
4. The audio is processed through the Seamless M4T v2 model
5. Translated audio is sent back to the client as base64-encoded data
6. The client decodes and plays the audio sequentially

## License

[MIT License](LICENSE)

## Acknowledgements

- [Meta AI](https://ai.meta.com/) for the Seamless M4T v2 model
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [HuggingFace](https://huggingface.co/) for the model hosting
