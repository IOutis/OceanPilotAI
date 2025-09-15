from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Form, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks
import logging
import sys
import os
from typing import List, Dict, Literal
import asyncio
import threading
import json
import copy
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI

SESSIONS: Dict[str, Dict] = {}

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()

# Now you can import the function from the other folder
from data.loader.load import extract_file_metadata

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Standard Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global reference to store the manager
_global_manager = None
main_loop = None

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    logger.info("FastAPI startup complete - WebSocket ready")
    print("FastAPI startup complete - WebSocket ready")

# --- Simplified WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New WebSocket connection accepted.")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("WebSocket connection closed.")

    async def broadcast(self, message: dict):
        """Sends a JSON message to all connected clients."""
        logger.info(f"🔥 BROADCASTING to {len(self.active_connections)} connections: {message}")
        connections_copy = self.active_connections.copy()
        disconnected_connections = []
        for connection in connections_copy:
            try:
                await connection.send_json(message)
                logger.info("✅ Successfully sent message to connection")
            except Exception as e:
                logger.error(f"❌ Error sending message to connection: {e}")
                disconnected_connections.append(connection)
        # Remove broken connections
        for conn in disconnected_connections:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

    def broadcast_sync(self, message: dict, timeout: float = 5.0):
        """
        Schedule broadcast into FastAPI main loop. Safe to call from threads / sync tools.
        """
        logger.info(f"🔥 SYNC BROADCAST called with message: {message}")
        
        if not self.active_connections:
            logger.warning("⚠️ No active WebSocket connections")
            return

        if main_loop is None:
            logger.error("❌ main_loop is not set. Cannot schedule broadcast.")
            return

        try:
            future = asyncio.run_coroutine_threadsafe(self.broadcast(message), main_loop)
            result = future.result(timeout=timeout)
            logger.info("✅ broadcast_sync completed successfully")
            return result
        except Exception as e:
            logger.error(f"❌ broadcast_sync error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

# Initialize the manager globally
manager = ConnectionManager()
_global_manager = manager

# --- Tool factory with session lookup ---
def create_mapping_tool(session_id: str):
    @tool
    def submit_mapping_suggestion(suggestions: List[Dict]) -> str:
        """
        Send mapping suggestions to the WebSocket clients for this session.
        
        Args:
            suggestions: List of dictionaries, each containing:
                - column_name: str (name of the column)
                - role: str (one of: Ignore, Latitude, Longitude, Date, Time, Depth, Temperature, Salinity, Oxygen, Phosphate, Silicate, Nitrate, Categorical, Numerical)
                - data_type: str (numerical, categorical, datetime, geospatial)
                - suggested_use: str (brief description of usage)
        """
        try:
            logger.info(f"🔥 TOOL CALLED: submit_mapping_suggestion for session {session_id}")
            logger.info(f"🔥 TOOL ARGS: {suggestions}")
            
            sess = SESSIONS.get(session_id)
            if not sess:
                logger.error(f"❌ Session {session_id} not found")
                return f"Error: session {session_id} not found"

            mgr = sess.get("manager")
            if not isinstance(mgr, ConnectionManager):
                logger.error("❌ No manager found for session")
                return "Error: no manager found for session"

            valid_suggestions = {}
            for i, suggestion in enumerate(suggestions):
                logger.info(f"🔥 Processing suggestion {i}: {suggestion}")
                
                if isinstance(suggestion, dict):
                    column_name = suggestion.get("column_name") or suggestion.get("column")
                    role = suggestion.get("role")
                    data_type = suggestion.get("data_type")
                    suggested_use = suggestion.get("suggested_use")
                    
                    if all([column_name, role, data_type, suggested_use]):
                        valid_suggestions[column_name] = {
                            "role": role,
                            "data_type": data_type,
                            "suggested_use": suggested_use
                        }
                        logger.info(f"✅ Valid suggestion added for column: {column_name}")
                    else:
                        logger.warning(f"⚠️ Missing fields in suggestion: {suggestion}")

            if not valid_suggestions:
                logger.warning("⚠️ No valid suggestions to broadcast")
                return "No valid suggestions provided."

            message = {"type": "mapping_suggestion", "payload": valid_suggestions}
            logger.info(f"🔥 About to broadcast message: {message}")
            
            # Store message in session for debugging
            sess["last_tool_message"] = message
            
            # CRITICAL: schedule into the main loop (safe from tool context)
            mgr.broadcast_sync(message)
            
            logger.info(f"✅ Tool completed successfully for session {session_id}")
            return f"Successfully submitted {len(valid_suggestions)} mapping suggestions"
            
        except Exception as e:
            logger.error(f"❌ Tool failed with error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: {e}"
    
    return submit_mapping_suggestion

# --- Modern LangChain Agent Implementation ---
class AgentClass:
    def __init__(self, session_id: str):
        self.session_id = session_id
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
        )
        
        # Create tools
        self.tools = [create_mapping_tool(session_id)]
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that can use tools to accomplish tasks."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the tool-calling agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            return_intermediate_steps=True
        )

    async def chat(self, prompt: str):
        """
        Use AgentExecutor to handle the prompt.
        Tool calls will automatically execute and broadcast via WebSocket.
        """
        try:
            logger.info(f"🔥 AGENT CHAT called for session {self.session_id}")
            logger.info(f"🔥 AGENT INPUT: {prompt}")
            
            # Use the agent executor
            result = await self.agent_executor.ainvoke({
                "input": prompt
            })
            
            logger.info(f"🔥 AGENT RESULT: {result}")
            
            # Extract the output
            output = result.get('output', 'Task completed.')
            
            # Log intermediate steps for debugging
            if 'intermediate_steps' in result:
                logger.info(f"🔥 INTERMEDIATE STEPS: {result['intermediate_steps']}")
            
            return output
            
        except Exception as e:
            logger.error(f"❌ Agent chat error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

# --- API Endpoints ---
@app.post("/uploadfile/")
async def create_upload_file(session_id: str = Form(...), file: UploadFile = File(...)):
    metadata = await extract_file_metadata(file)
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"manager": manager}
        
    if 'uploads' not in SESSIONS[session_id]:
        SESSIONS[session_id]['uploads'] = []
    
    SESSIONS[session_id]["manager"] = manager
    SESSIONS[session_id]['uploads'].append(metadata)
    logger.info(f"Stored metadata for session {session_id}. Total uploads: {len(SESSIONS[session_id]['uploads'])}")
    # logger.info(f"Meta data: {metadata}")   
    return metadata

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    session_id = None
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"🔥 WebSocket received: {data}")
            try:
                msg = json.loads(data)
                # First handshake with session_id
                if "session_id" in msg:
                    session_id = msg["session_id"]
                    if session_id not in SESSIONS:
                        SESSIONS[session_id] = {}
                    SESSIONS[session_id]["manager"] = manager
                    SESSIONS[session_id]["socket"] = websocket
                    logger.info(f"✅ WebSocket bound to session {session_id}")
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Invalid JSON received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        if session_id and session_id in SESSIONS:
            SESSIONS[session_id].pop("socket", None)
            logger.info(f"🔥 WebSocket for session {session_id} disconnected")

@app.get("/test_broadcast")
async def test_broadcast():
    """Test endpoint for WebSocket broadcasting"""
    test_message = {
        "type": "mapping_suggestion",
        "payload": {
            "test_column": {
                "role": "Temperature",
                "data_type": "numerical", 
                "suggested_use": "Test mapping suggestion"
            }
        }
    }
    await manager.broadcast(test_message)
    return {"status": "test_broadcast_sent"}

@app.get("/test_sync")
async def test_sync():
    """Test the sync broadcasting"""
    test_message = {
        "type": "mapping_suggestion",
        "payload": {
            "test_sync_column": {
                "role": "Salinity",
                "data_type": "numerical", 
                "suggested_use": "Test sync broadcast"
            }
        }
    }
    
    # Test sync version
    manager.broadcast_sync(test_message)
    
    return {"status": "sync_test_sent"}

@app.get("/debug_session/{session_id}")
async def debug_session(session_id: str):
    """Debug endpoint to check session state"""
    session = SESSIONS.get(session_id, {})
    return {
        "session_id": session_id,
        "session_keys": list(session.keys()),
        "has_manager": "manager" in session,
        "manager_connections": len(session.get("manager", {}).get("active_connections", [])) if "manager" in session else 0,
        "last_tool_message": session.get("last_tool_message"),
        "uploads_count": len(session.get("uploads", []))
    }

@app.post("/chat")
async def chat_handler(payload: dict, background_tasks: BackgroundTasks):
    """Main chat handler"""
    try:
        user_message = payload.get("message", "")
        active_phase_context = payload.get("context", {})
        active_view = payload.get("view", "")
        session_id = payload.get("session_id", "default_session")
        
        logger.info(f"🔥 CHAT HANDLER: session={session_id}, view={active_view}")
        logger.info(f"🔥 USER MESSAGE: {user_message}")
        
        if not session_id or session_id not in SESSIONS:
            await manager.broadcast({
                "type": "agent_response",
                "payload": {"from": "bot", "text": "Error: Your session could not be found. Please try uploading a file again."}
            })
            return {"status": "error_no_session"}

        # Get session context
        full_session_context = SESSIONS[session_id]['uploads']
        filename = active_phase_context.get("name", "the current data")
        sampled_context = copy.deepcopy(full_session_context)
        trimmed_context = []
        
        for upload in sampled_context:
            if 'sample_data' in upload and upload['sample_data'] is not None and upload['filename'] == filename:
                trimmed_context.append({
                    "filename": upload.get("filename", "unknown"),
                    "data_preview": upload.get("sample_data", {})
                })
        
        await manager.broadcast({
            "type": "status_update",
            "payload": {"status": "thinking", "message": f"Thinking about '{filename}'..."}
        })

        if active_view == 'mapping' and active_phase_context:
            active_filename = active_phase_context.get('data', {}).get('filename')
            server_side_data = next((upload for upload in full_session_context if upload.get('filename') == active_filename), None)
            columns = []
            if server_side_data:
                data_for_ui = server_side_data.get('data', [])
                if data_for_ui and isinstance(data_for_ui, list) and len(data_for_ui) > 0:
                    columns = list(data_for_ui[0].keys())
            
            logger.info(f"🔥 MAPPING MODE: columns={columns}")
            
            prompt = f"""
You are an expert data analyst specializing in marine science data. 

The user has uploaded a file with these columns: {columns}

You MUST use the submit_mapping_suggestion tool to provide your analysis for each column. This is absolutely required.

For each column, determine:
- role: One of ["Ignore", "Latitude", "Longitude", "Date", "Time", "Depth", "Temperature", "Salinity", "Oxygen", "Phosphate", "Silicate", "Nitrate", "Categorical", "Numerical"]
- data_type: One of ["numerical", "categorical", "datetime", "geospatial"]  
- suggested_use: Brief description of how this column should be used

Create a list with ALL columns and call the submit_mapping_suggestion tool once with all suggestions.

User's request: "{user_message}"
"""
        else:
            prompt = f"""
You are a marine data science assistant.
Answer the user's question based on the provided data context.

Active context: {active_phase_context}
Full Session Data Context: {trimmed_context}
User's Question: {user_message}
"""
        
        # Initialize agent for this session
        agent = AgentClass(session_id)
        SESSIONS[session_id]['last_message'] = user_message

        await manager.broadcast({
            "type": "status_update", 
            "payload": {"status": "thinking", "message": f"Analyzing your request about '{filename}'..."}
        })
        
        logger.info("🔥 Calling agent...")
        response_text = await agent.chat(prompt)
        logger.info(f"🔥 Agent response: {response_text}")
        
        content = str(response_text) or "I've processed your request."
        
        logger.info(f"🔥 Final response content: {content}")
        logger.info(f"🔥 Session keys: {list(SESSIONS[session_id].keys())}")
        
        await manager.broadcast({
            "type": "agent_response",
            "payload": {"from": "bot", "text": content}
        })

        return {"status": "message_processed"}
        
    except Exception as e:
        logger.error(f"❌ Chat handler error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        await manager.broadcast({
            "type": "agent_response", 
            "payload": {"from": "bot", "text": f"Error processing your request: {str(e)}"}
        })
        return {"status": "error", "message": str(e)}