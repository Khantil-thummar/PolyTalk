from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pyngrok import ngrok, conf
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()

conf.get_default().auth_token = os.getenv("ngrok_token")



app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

connected_clients = []

# Global variable to store ngrok URL
ngrok_tunnel = ngrok.connect(8000, bind_tls=True)
print("ngrok tunnel URL:", ngrok_tunnel)
@app.get("/")
async def get():
    with open(os.path.join("static", "index.html"), "r") as f:
        html_content = f.read()
    html_content = html_content.replace("{{WEBSOCKET_URL}}", f"wss://{ngrok_tunnel.public_url.replace('https://', '')}/ws")
    return HTMLResponse(content=html_content, status_code=200)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            for client in connected_clients:
                if client != websocket:
                    await client.send_bytes(data)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
