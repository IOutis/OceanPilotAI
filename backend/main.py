from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,Form, UploadFile, File, WebSocket, WebSocketDisconnect
import logging
import sys
import os
from typing import List,Dict
import asyncio
from langchain.chat_models import init_chat_model
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
import copy
SESSIONS: Dict[str, Dict] = {}

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()


# Now you can import the function from the other folder
from data.loader.load import extract_file_metadata

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Standard Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- WebSocket Connection Manager ---
# A robust class to manage all active WebSocket connections.
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New WebSocket connection accepted.")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("WebSocket connection closed.")

    async def broadcast(self, message: dict):
        """Sends a JSON message to all connected clients."""
        logger.info(f"Broadcasting message: {message}")
        for connection in self.active_connections:
            await connection.send_json(message)
            
from langchain_google_genai import ChatGoogleGenerativeAI


class AgentClass:
    def __init__(self):
        self.name = "Gemini Agent"
        self.description = "An agent that uses Gemini API to answer questions."
        self.api_key = GEMINI_API_KEY
        self.llm = init_chat_model(
                "gemini-2.0-flash",
                model_provider="google_genai",
                api_key=self.api_key,
                temperature=0.1
            ) 
    def chat(self, prompt: str) -> str:
        response = self.llm.invoke(
            input=prompt,
            tools=[GenAITool(google_search={})],

        )
        return response          


# Initialize FastAPI app and the Connection Manager
app = FastAPI()
manager = ConnectionManager()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

# 1. Standard file upload endpoint (unchanged)
@app.post("/uploadfile/")
async def create_upload_file(session_id: str = Form(...),file: UploadFile = File(...)):
    metadata = await extract_file_metadata(file)
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {}
        
    # We'll store a history of uploads for each session
    if 'uploads' not in SESSIONS[session_id]:
        SESSIONS[session_id]['uploads'] = []
    
    SESSIONS[session_id]['uploads'].append(metadata)
    logger.info(f"Stored metadata for session {session_id}. Total uploads: {len(SESSIONS[session_id]['uploads'])}")

    return metadata

# 2. WebSocket endpoint for clients to connect to
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # You can add logic here to receive messages from clients if needed
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# 3. New Chat endpoint to trigger the agent loop
@app.post("/chat")
async def chat_handler(payload: dict):
    """
    Receives a message from the user, simulates an LLM agent loop,
    and broadcasts updates via WebSocket.
    """
    user_message = payload.get("message", "")
    active_phase_context = payload.get("context", {})
    session_id = payload.get("session_id", "default_session")
    print(session_id)
    if not session_id or session_id not in SESSIONS:
        await manager.broadcast({
            "type": "agent_response",
            "payload": {"from": "bot", "text": "Error: Your session could not be found. Please try uploading a file again."}
        })
        return {"status": "error_no_session"}

    # --- THE FIX: Retrieve context from the session ---
    full_session_context = SESSIONS[session_id]['uploads'] # Limit context size for performance
    filename = active_phase_context.get("name", "the current data")
    sampled_context = copy.deepcopy(full_session_context)
    trimmed_context = []
    for upload in sampled_context:
        # Check if this is a WOD file preview
        if 'sample_data' in upload and upload['sample_data'] is not None and upload['filename'] == filename:
            trimmed_context.append({
                "filename": upload.get("filename", "unknown"),
                "data_preview": upload.get("sample_data", {})
            })
        
    await manager.broadcast({
        "type": "status_update",
        "payload": {"status": "thinking", "message": f"Thinking about '{filename}'..."}
    })
    print("Full session context:", trimmed_context)
    
    # Step 2: Construct a detailed prompt for the Gemini model
    prompt = f"""
    You are a marine data science assistant.
    Your task is to answer the user's question based on the provided data context from the user's session.
    The session context contains a history of all uploaded files.
    Be concise and helpful.

    **Full Session Data Context (all uploads):**
    {trimmed_context}

    **User's Question:**
    {user_message}
    """
    
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {}
    SESSIONS[session_id]['last_message'] = user_message

    # --- SIMULATED GEMINI AGENT LOOP ---

    # Step 1: Acknowledge and start processing (push update to UI)
    await manager.broadcast({
        "type": "status_update",
        "payload": {"status": "thinking", "message": f"Analyzing your request about '{filename}'..."}
    })
    
    # Step 2: Simulate the delay of an LLM call or tool execution
    # await asyncio.sleep(3) 

    # Step 3: Simulate finding a result (push final update to UI)
    # response_text = f"Based on my analysis of '{filename}', I found that the primary cruise ID is '{active_phase_context.get('data', {}).get('data_preview', {}).get('cruise', 'N/A')}'. The data was collected around latitude {active_phase_context.get('data', {}).get('data_preview', {}).get('latitude', 'N/A')}."
    llm = AgentClass()
    response_text = llm.chat(prompt)
    # # response_text = f"Based on my analysis of '{filename}', I found that the primary cruise ID is '{active_context.get('data', {}).get('data_preview', {}).get('cruise', 'N/A')}'. The data was collected around latitude {active_context.get('data', {}).get('data_preview', {}).get('latitude', 'N/A')}."
    response_text = response_text.content
    await manager.broadcast({
        "type": "agent_response",
        "payload": {"from": "bot", "text": response_text}
    })

    return {"status": "message_processed"}
