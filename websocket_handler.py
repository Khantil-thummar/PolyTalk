import os
from fastapi import WebSocket, WebSocketDisconnect
from utils import convert_chunk_to_wav
from typing import Callable

class AudioWebSocketHandler:
    def __init__(self, translate_func: Callable):
        """
        Initialize the WebSocket handler.
        
        Args:
            translate_func: Function to use for translation
        """
        self.translate_func = translate_func
        self.combine_chunks = 2  # How many chunks to combine before processing
        # self.save_chunks will be set externally (from main.py)

    async def handle_connection(self, websocket: WebSocket):
        """
        Handle a WebSocket connection for audio streaming.
        
        Args:
            websocket: The WebSocket connection
        """
        await websocket.accept()
        os.makedirs("chunks", exist_ok=True)

        buffer = bytearray()
        chunk_counter = 0
        file_counter = 0

        try:
            while True:
                try:
                    data = await websocket.receive_bytes()
                    chunk_counter += 1
                    buffer.extend(data)
                    
                    if self.combine_chunks is None:
                        await convert_chunk_to_wav(
                            data, 
                            suffix=None, 
                            save_chunks=self.save_chunks,
                            translate_func=self.translate_func
                        )
                        buffer.clear()
                        chunk_counter = 0
                        continue

                    if chunk_counter >= self.combine_chunks:
                        await convert_chunk_to_wav(
                            buffer, 
                            suffix=file_counter, 
                            save_chunks=self.save_chunks,
                            translate_func=self.translate_func
                        )
                        buffer.clear()
                        chunk_counter = 0
                        file_counter += 1
                except WebSocketDisconnect:
                    print("WebSocket disconnected while receiving data")
                    break
        except Exception as e:
            print(f"Error in WebSocket handler: {e}")
        
        finally:
            if buffer:
                print(f"Processing final buffer of size {len(buffer)}")
                try:
                    await convert_chunk_to_wav(
                        buffer, 
                        suffix=file_counter, 
                        save_chunks=self.save_chunks,
                        translate_func=self.translate_func
                    )
                except Exception as e:
                    print(f"Error processing final chunk: {e}")
            print("WebSocket connection closed") 