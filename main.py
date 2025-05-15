import os
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Import configuration
from config import (
    MODEL_NAME, 
    TARGET_LANGUAGE,
    COMBINE_CHUNKS,
    SAVE_CHUNKS,
    HTML_TEMPLATE_DIR,
    CHUNK_DIR,
    OUTPUT_DIR
)

# Import modules
from translator import Translator
from websocket_handler import AudioWebSocketHandler

# Ensure required directories exist
os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize the application
app = FastAPI(title="Real-time Voice Translator")
app.mount("/static", StaticFiles(directory=HTML_TEMPLATE_DIR), name="static")

# Initialize translator
translator = Translator(
    model_name=MODEL_NAME,
    target_lang=TARGET_LANGUAGE
)

# Initialize WebSocket handler
ws_handler = AudioWebSocketHandler(
    translate_func=translator.translate_audio
)
ws_handler.combine_chunks = COMBINE_CHUNKS
ws_handler.save_chunks = SAVE_CHUNKS

@app.get("/")
async def root():
    """Serve the main page"""
    with open(f"{HTML_TEMPLATE_DIR}/index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for audio streaming"""
    await ws_handler.handle_connection(websocket)

# Load the model when the server starts
@app.on_event("startup")
async def startup_event():
    """Initialize resources on server startup"""
    translator.load_model()
    print(f"Server started. Translating to: {TARGET_LANGUAGE}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
