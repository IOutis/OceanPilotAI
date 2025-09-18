import uuid
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
import time
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List, Any, Optional
from scipy import stats
from scipy.stats import pearsonr
import pandas as pd
import numpy as np


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

# Rate limiting variables
last_request_time = 0
min_request_interval = 2.0  # Minimum 2 seconds between requests

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

    def broadcast_sync(self, message: dict, timeout: float = 10.0):
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



# Add this new analysis data processing endpoint
@app.post("/analysis/process")
async def process_analysis_data(payload: dict):
    """
    Process analysis data and return visualization-ready data
    """
    try:
        session_id = payload.get("session_id")
        source_phase_id = payload.get("source_phase_id")
        analysis_config = payload.get("analysis_config")
        
        if not all([session_id, source_phase_id, analysis_config]):
            return {"error": "Missing required parameters"}
        
        if session_id not in SESSIONS:
            return {"error": f"Session {session_id} not found"}
        
        uploads = SESSIONS[session_id].get('uploads', [])
        source_upload = None
        
        for upload in uploads:
            if upload.get('id') == source_phase_id:
                source_upload = upload
                break
        
        if not source_upload:
            return {"error": f"Source phase {source_phase_id} not found"}
        
        # Use processed data if available, otherwise use original data
        data = source_upload.get('processed_data') or source_upload.get('data')
        mappings = source_upload.get('mappings', {})
        
        if not data:
            return {"error": "No data found"}
        
        df = pd.DataFrame(data)
        
        # Apply mappings
        rename_map = {original_name: role for original_name, role in mappings.items() if role != "Ignore"}
        df.rename(columns=rename_map, inplace=True)
        
        columns_to_drop = [original_name for original_name, role in mappings.items() if role == "Ignore"]
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        # Process the analysis
        processed_data, visualization_config = perform_data_analysis(df, analysis_config)
        
        return {
            "status": "success",
            "data": processed_data,
            "config": visualization_config
        }
        
    except Exception as e:
        logger.error(f"Error processing analysis: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": f"Failed to process analysis: {str(e)}"}

def create_analysis_tool(session_id: str):
    @tool
    def generate_analysis_and_plot(analysis_type: str, x_column: str, y_column: str = None, 
                                 title: str = None, description: str = None) -> str:
        """
        Generate analysis results and visualization configuration based on user query.
        
        Args:
            analysis_type: str (one of: line, scatter, bar, area, histogram, correlation_matrix)
            x_column: str (name of the x-axis column)
            y_column: str (name of the y-axis column, optional for some plot types)
            title: str (title for the visualization)
            description: str (description of the analysis findings)
        """
        try:
            logger.info(f"🔥 ANALYSIS TOOL CALLED: {analysis_type} for session {session_id}")
            
            sess = SESSIONS.get(session_id)
            if not sess:
                logger.error(f"❌ Session {session_id} not found")
                return f"Error: session {session_id} not found"

            mgr = sess.get("manager")
            if not isinstance(mgr, ConnectionManager):
                logger.error("❌ No manager found for session")
                return "Error: no manager found for session"

            # Get the analysis request stored in session
            pending_analysis = sess.get("pending_analysis")
            source_phase_id = None
            
            # Find the active phase or use the most recent upload
            if pending_analysis:
                source_phase_id = pending_analysis.get("source_phase_id")
            else:
                # Default to the most recent upload
                uploads = sess.get('uploads', [])
                if uploads:
                    source_phase_id = uploads[-1].get('id')
            
            if not source_phase_id:
                return "Error: No data source found for analysis"
            
            # Process the analysis data
            analysis_config = {
                "analysis_type": analysis_type,
                "x_column": x_column,
                "y_column": y_column,
                "title": title or f"{analysis_type.title()} Analysis",
                "description": description or f"Analysis of {x_column}" + (f" vs {y_column}" if y_column else "")
            }
            
            # Call the analysis processing function
            import asyncio
            import requests
            
            try:
                # Make a synchronous call to our analysis processing endpoint
                response = requests.post('http://localhost:8000/analysis/process', 
                    json={
                        "session_id": session_id,
                        "source_phase_id": source_phase_id,
                        "analysis_config": analysis_config
                    })
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        # Send the processed data and config to frontend
                        analysis_result = {
                            "type": "analysis_result",
                            "payload": {
                                "analysis_type": analysis_type,
                                "data": result["data"],
                                "config": result["config"]
                            }
                        }
                        mgr.broadcast_sync(analysis_result)
                        return f"Successfully generated {analysis_type} visualization"
                    else:
                        return f"Error processing analysis: {result.get('error', 'Unknown error')}"
                else:
                    return f"Error: Analysis service returned status {response.status_code}"
            
            except Exception as e:
                logger.error(f"Error calling analysis service: {e}")
                return f"Error processing analysis: {str(e)}"
            
        except Exception as e:
            logger.error(f"❌ Analysis tool failed with error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: {e}"
    
    return generate_analysis_and_plot

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

def create_merge_suggestion_tool(session_id: str):
    @tool
    def suggest_merge_strategy(file_info: List[Dict], strategy_recommendation: str, 
                              join_columns: Dict[str, str], reasoning: str, 
                              considerations: str = None) -> str:
        """
        Provide merge strategy suggestions to the user based on dataset analysis.
        
        Args:
            file_info: List of file information dictionaries with id, name, and columns
            strategy_recommendation: str (one of: inner, outer, left, concat)
            join_columns: Dict mapping file_id to recommended join column name
            reasoning: str explaining why this strategy and columns are recommended
            considerations: str optional warnings or things to consider
        """
        try:
            logger.info(f"🔥 MERGE TOOL CALLED for session {session_id}")
            logger.info(f"🔥 MERGE ARGS: strategy={strategy_recommendation}, join_columns={join_columns}")
            
            sess = SESSIONS.get(session_id)
            if not sess:
                logger.error(f"❌ Session {session_id} not found")
                return f"Error: session {session_id} not found"

            mgr = sess.get("manager")
            if not isinstance(mgr, ConnectionManager):
                logger.error("❌ No manager found for session")
                return "Error: no manager found for session"

            # Validate the strategy
            valid_strategies = ['inner', 'outer', 'left', 'concat']
            if strategy_recommendation not in valid_strategies:
                return f"Error: Invalid strategy '{strategy_recommendation}'. Must be one of: {valid_strategies}"

            # Prepare the suggestion payload
            suggestion_payload = {
                "strategy": strategy_recommendation,
                "join_columns": join_columns,
                "reasoning": reasoning,
                "considerations": considerations or "",
                "file_info": file_info
            }

            message = {
                "type": "merge_suggestion", 
                "payload": suggestion_payload
            }
            
            logger.info(f"🔥 About to broadcast merge suggestion: {message}")
            
            # Store message in session for debugging
            sess["last_merge_suggestion"] = message
            
            # Broadcast the suggestion
            mgr.broadcast_sync(message)
            
            logger.info(f"✅ Merge suggestion tool completed successfully for session {session_id}")
            return f"Successfully provided merge strategy recommendation: {strategy_recommendation}"
            
        except Exception as e:
            logger.error(f"❌ Merge suggestion tool failed with error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: {e}"
    
    return suggest_merge_strategy

# --- Rate limiting helper ---  
async def wait_for_rate_limit():
    """Ensure we don't exceed API rate limits"""
    global last_request_time
    
    current_time = time.time()
    time_since_last_request = current_time - last_request_time
    
    if time_since_last_request < min_request_interval:
        sleep_time = min_request_interval - time_since_last_request
        logger.info(f"⏰ Rate limiting: sleeping for {sleep_time:.2f} seconds")
        await asyncio.sleep(sleep_time)
    
    last_request_time = time.time()

# --- Modern LangChain Agent Implementation ---
class AgentClass:
    def __init__(self, session_id: str):
        self.session_id = session_id
        
        # Initialize LLM with better quota-friendly settings
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # Changed from gemini-2.0-flash for better quota
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            max_retries=3,  # Increased retries for rate limit handling
            # request_timeout=30,  # Added timeout
        )
        
        # Create tools
        self.tools = [
            create_mapping_tool(session_id),
            create_analysis_tool(session_id),
            create_merge_suggestion_tool(session_id)  # <-- Add this line
        ]
        
        # Create prompt template with shorter system message
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a marine data science assistant. Use tools when needed."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the tool-calling agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        
        # Create agent executor with conservative settings
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=2,
            return_intermediate_steps=True,
            # early_stopping_method="generate",  # Stop early to save tokens
        )

    async def chat(self, prompt: str):
        """
        Use AgentExecutor to handle the prompt with rate limiting.
        """
        try:
            logger.info(f"🔥 AGENT CHAT called for session {self.session_id}")
            
            # Apply rate limiting
            await wait_for_rate_limit()
            
            # Truncate very long prompts to save tokens
            if len(prompt) > 8000:
                logger.warning("⚠️ Prompt too long, truncating...")
                prompt = prompt[:8000] + "... [truncated for quota management]"
            
            logger.info(f"🔥 AGENT INPUT (length: {len(prompt)})")
            
            # Use the agent executor
            result = await self.agent_executor.ainvoke({
                "input": prompt
            })
            
            logger.info(f"🔥 AGENT RESULT received")
            
            # Extract the output
            output = result.get('output', 'Task completed.')
            
            # Check for quota-related errors
            error_keywords = ['resourceexhausted', 'quota', 'exceeded', 'rate limit', '429']
            if any(keyword in output.lower() for keyword in error_keywords):
                logger.warning(f"⚠️ Quota-related error detected: {output}")
                return "I'm experiencing high demand right now. Please wait a moment and try again, or consider upgrading to a paid plan for faster responses."
            
            return output
            
        except Exception as e:
            logger.error(f"❌ Agent chat error: {e}")
            error_str = str(e).lower()
            
            # Handle specific quota/rate limit errors
            if any(keyword in error_str for keyword in ['quota', 'exceeded', '429', 'resourceexhausted', 'rate limit']):
                return "I've hit the API quota limit. Please wait a few minutes before trying again, or consider upgrading your Google AI Studio plan for higher limits."
            
            # Handle other errors
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"I encountered an error: {str(e)[:200]}. Please try again."
        
class AnalysisAgentClass(AgentClass):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        
        # Add analysis tool to existing tools
        self.tools.append(create_analysis_tool(session_id))
        
        # Update prompt template for analysis context
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a marine data analysis assistant. When users ask questions about their data:

1. Determine the appropriate visualization type (line, scatter, bar, area, histogram)
2. Identify which columns to use for x and y axes
3. Use the generate_analysis_and_plot tool with the correct parameters
4. Provide clear titles and descriptions

Available analysis types:
- line: for time series or sequential data
- scatter: for correlation analysis
- bar: for categorical comparisons
- area: for cumulative or filled area charts
- histogram: for distribution analysis

Always use tools when users ask for plots, charts, or data analysis."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Recreate agent with updated prompt and tools
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            return_intermediate_steps=True,
            early_stopping_method="generate",
        )

def perform_data_analysis(df: pd.DataFrame, analysis_config: Dict) -> tuple:
    """
    Perform the actual data analysis and return processed data and visualization config.
    
    Returns:
        tuple: (processed_data, visualization_config)
    """
    analysis_type = analysis_config["analysis_type"]
    x_column = analysis_config["x_column"]
    y_column = analysis_config.get("y_column")
    
    try:
        # Basic data validation
        # if x_column not in df.columns:
        #     raise ValueError(f"Column '{x_column}' not found in data")
        
        # if y_column and y_column not in df.columns:
        #     raise ValueError(f"Column '{y_column}' not found in data")
        
        # Remove any infinite values and convert to numeric where possible
        df_clean = df.copy()
        
        # Convert columns to numeric if possible
        for col in [x_column, y_column] if y_column else [x_column]:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove rows with NaN values in key columns
        columns_to_check = [x_column, y_column] if y_column else [x_column]
        df_clean = df_clean.dropna(subset=columns_to_check)
        
        if df_clean.empty:
            raise ValueError("No valid data remaining after cleaning")
        
        visualization_config = {
            "type": analysis_type,
            "config": {
                "title": analysis_config.get("title", f"{analysis_type.title()} Analysis"),
                "description": analysis_config.get("description", ""),
                "xAxis": x_column,
                "yAxis": y_column,
                "xAxisLabel": x_column.replace('_', ' ').title(),
                "yAxisLabel": y_column.replace('_', ' ').title() if y_column else ""
            }
        }
        
        # Process data based on analysis type
        if analysis_type == "scatter" or analysis_type == "line":
            if not y_column:
                raise ValueError(f"{analysis_type} plot requires both x and y columns")
            
            processed_data = df_clean[[x_column, y_column]].to_dict('records')
            
            # Add correlation info for scatter plots
            if analysis_type == "scatter" and len(df_clean) > 1:
                corr, p_value = pearsonr(df_clean[x_column], df_clean[y_column])
                visualization_config["config"]["description"] += f" Correlation: {corr:.3f} (p={p_value:.3f})"
        
        elif analysis_type == "bar":
            # For bar charts, group by x_column and aggregate y_column
            if y_column:
                grouped = df_clean.groupby(x_column)[y_column].mean().reset_index()
                processed_data = grouped.to_dict('records')
            else:
                # Count occurrences
                value_counts = df_clean[x_column].value_counts().reset_index()
                value_counts.columns = [x_column, 'count']
                processed_data = value_counts.to_dict('records')
                visualization_config["config"]["yAxis"] = 'count'
                visualization_config["config"]["yAxisLabel"] = 'Count'
        
        elif analysis_type == "area":
            if not y_column:
                raise ValueError("Area plot requires both x and y columns")
            processed_data = df_clean[[x_column, y_column]].sort_values(x_column).to_dict('records')
        
        elif analysis_type == "histogram":
            # Create bins for histogram
            hist_data, bin_edges = np.histogram(df_clean[x_column], bins=20)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            processed_data = [
                {x_column: float(center), 'frequency': int(count)} 
                for center, count in zip(bin_centers, hist_data)
            ]
            visualization_config["config"]["yAxis"] = 'frequency'
            visualization_config["config"]["yAxisLabel"] = 'Frequency'
            visualization_config["type"] = "bar"  # Use bar chart for histogram display
        
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        return processed_data, visualization_config
        
    except Exception as e:
        logger.error(f"Error in data analysis: {e}")
        raise e


@app.get("/analysis/suggestions/{session_id}/{source_phase_id}")
async def get_analysis_suggestions(session_id: str, source_phase_id: str):
    """
    Get suggested analysis queries based on the available data columns
    """
    try:
        if session_id not in SESSIONS:
            return {"error": f"Session {session_id} not found"}
        
        uploads = SESSIONS[session_id].get('uploads', [])
        source_upload = None
        
        for upload in uploads:
            if upload.get('id') == source_phase_id:
                source_upload = upload
                break
        
        if not source_upload:
            return {"error": f"Source phase {source_phase_id} not found"}
        
        mappings = source_upload.get('mappings', {})
        available_columns = [role for role in mappings.values() if role != "Ignore"]
        
        # Generate smart suggestions based on available column types
        suggestions = []
        
        # Basic exploration
        suggestions.append("Show me an overview of the data distribution")
        
        # Marine science specific queries
        if 'Temperature' in available_columns and 'Depth' in available_columns:
            suggestions.append("Plot temperature vs depth profile")
            suggestions.append("Show how temperature changes with depth")
        
        if 'Salinity' in available_columns and 'Temperature' in available_columns:
            suggestions.append("Show temperature-salinity relationship")
            suggestions.append("Create a T-S diagram")
        
        if 'Oxygen' in available_columns:
            suggestions.append("Analyze oxygen distribution")
            if 'Depth' in available_columns:
                suggestions.append("Show oxygen levels by depth")
        
        if 'Date' in available_columns or 'Time' in available_columns:
            suggestions.append("Show temporal trends in the data")
            suggestions.append("Display time series of measurements")
        
        if 'Latitude' in available_columns and 'Longitude' in available_columns:
            suggestions.append("Show spatial distribution of measurements")
        
        # Nutrient analysis
        nutrients = ['Phosphate', 'Silicate', 'Nitrate']
        available_nutrients = [n for n in nutrients if n in available_columns]
        if available_nutrients:
            suggestions.append(f"Compare {' and '.join(available_nutrients)} concentrations")
        
        # General analysis types
        if len(available_columns) >= 2:
            suggestions.append("Show correlations between all variables")
            suggestions.append("Create scatter plots for key relationships")
        
        return {
            "status": "success", 
            "suggestions": suggestions[:8],  # Limit to 8 suggestions
            "available_columns": available_columns
        }
        
    except Exception as e:
        logger.error(f"Error generating analysis suggestions: {e}")
        return {"error": f"Failed to generate suggestions: {str(e)}"}


from fastapi.responses import JSONResponse

# Additional endpoint for statistical summaries
@app.post("/analysis/statistics")
async def get_statistical_summary(payload: dict):
    """
    Get statistical summary of the processed data
    """
    try:
        session_id = payload.get("session_id")
        source_phase_id = payload.get("source_phase_id")
        
        if not all([session_id, source_phase_id]):
            return {"error": "Missing required parameters"}
        
        if session_id not in SESSIONS:
            return {"error": f"Session {session_id} not found"}
        
        uploads = SESSIONS[session_id].get('uploads', [])
        source_upload = None
        
        for upload in uploads:
            if upload.get('id') == source_phase_id:
                source_upload = upload
                break
        
        if not source_upload:
            return {"error": f"Source phase {source_phase_id} not found"}
        
        # Use processed data if available
        data = source_upload.get('processed_data') or source_upload.get('data')
        mappings = source_upload.get('mappings', {})
        
        if not data:
            return {"error": "No data found"}
        
        df = pd.DataFrame(data)
        
        # Apply mappings
        rename_map = {original_name: role for original_name, role in mappings.items() if role != "Ignore"}
        df.rename(columns=rename_map, inplace=True)
        
        columns_to_drop = [original_name for original_name, role in mappings.items() if role == "Ignore"]
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        # Calculate comprehensive statistics
        stats_summary = {}

        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                stats_summary[col] = {
                    "type": "numeric",
                    "count": int(col_data.count()),
                    "mean": make_serializable(col_data.mean()),
                    "std": make_serializable(col_data.std()),
                    "min": make_serializable(col_data.min()),
                    "max": make_serializable(col_data.max()),
                    "median": make_serializable(col_data.median()),
                    "q25": make_serializable(col_data.quantile(0.25)),
                    "q75": make_serializable(col_data.quantile(0.75)),
                    "missing_count": int(df[col].isna().sum()),
                    "missing_percent": make_serializable(df[col].isna().mean() * 100)
                }

        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                value_counts = col_data.value_counts()
                stats_summary[col] = {
                    "type": "categorical",
                    "count": int(col_data.count()),
                    "unique_count": int(value_counts.nunique()),
                    "top_value": make_serializable(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "top_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "missing_count": int(df[col].isna().sum()),
                    "missing_percent": make_serializable(df[col].isna().mean() * 100)
                }

        # Correlation matrix for numeric columns
        correlation_matrix = None
        if len(numeric_cols) > 1:
            corr_df = df[numeric_cols].corr()
            correlation_matrix = {
                "columns": list(corr_df.columns),
                "matrix": [[make_serializable(v) for v in row] for row in corr_df.values.tolist()]
            }

        # --- FIX: Wrap all returned data with make_serializable ---
        result = {
            "status": "success",
            "statistics": {k: {kk: make_serializable(vv) for kk, vv in v.items()} for k, v in stats_summary.items()},
            "correlation_matrix": correlation_matrix,
            "data_shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
            "column_types": {
                "numeric": list(numeric_cols),
                "categorical": list(categorical_cols)
            }
        }
        return JSONResponse(content=result)

        
    except Exception as e:
        logger.error(f"Error generating statistical summary: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": f"Failed to generate statistics: {str(e)}"}

# --- Helper function to optimize data context ---
def optimize_context_for_quota(trimmed_context: List[Dict], max_rows: int = 3) -> List[Dict]:
    """Reduce context size to save tokens"""
    optimized = []
    for context in trimmed_context:
        optimized_context = {
            "filename": context.get("filename", "unknown")
        }
        
        data_preview = context.get("data_preview", {})
        if isinstance(data_preview, list) and len(data_preview) > max_rows:
            optimized_context["data_preview"] = data_preview[:max_rows]
            optimized_context["note"] = f"Showing first {max_rows} rows of {len(data_preview)} total"
        else:
            optimized_context["data_preview"] = data_preview
            
        optimized.append(optimized_context)
    
    return optimized

# --- API Endpoints ---
@app.post("/uploadfile/")
async def create_upload_file(session_id: str = Form(...), file: UploadFile = File(...)):
    metadata = await extract_file_metadata(file)
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"manager": manager}
        
    if 'uploads' not in SESSIONS[session_id]:
        SESSIONS[session_id]['uploads'] = []
    upload_id = str(uuid.uuid4())
    metadata['id'] = upload_id 
    SESSIONS[session_id]["manager"] = manager
    SESSIONS[session_id]['uploads'].append(metadata)
    logger.info(f"Stored metadata for session {session_id}. Total uploads: {len(SESSIONS[session_id]['uploads'])}")
    return metadata

@app.post("/mappings/confirm")
async def confirm_mappings(payload: dict):
    session_id = payload.get("session_id")
    source_phase_id = payload.get("source_phase_id")
    mappings = payload.get("mappings")

    if not all([session_id, source_phase_id, mappings]) or session_id not in SESSIONS:
        return {"status": "error", "message": "Invalid session, phase ID, or mappings."}

    # Find the correct upload in the session and add the mappings to it
    for upload in SESSIONS[session_id].get('uploads', []):
        if upload.get('id') == source_phase_id:
            upload['mappings'] = mappings
            logger.info(f"Saved mappings for phase {source_phase_id} in session {session_id}")
            return {"status": "success", "message": "Mappings saved successfully."}
            
    return {"status": "error", "message": "Source phase not found in session."}

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

@app.get("/quota_status")
async def quota_status():
    """Check current quota usage status"""
    global last_request_time
    return {
        "model": "gemini-1.5-flash",
        "last_request_time": last_request_time,
        "seconds_since_last_request": time.time() - last_request_time,
        "min_interval": min_request_interval,
        "ready_for_request": (time.time() - last_request_time) >= min_request_interval
    }

import pandas as pd
import numpy as np
# @app.post("/preprocess/stats")
# async def get_preprocessing_stats(payload: dict):
#     # It no longer looks up data in the session. It uses the data sent directly from the frontend.
#     file_path = payload.get("file_path")
#     mappings = payload.get("mappings")
#     session_id = payload.get("session_id")
    
#     if not file_path or not mappings:
#         return {"error": "A file path and mappings are required to generate stats."}
    
#     try:
#         # Load the full dataset from the saved file path
#         metadata = SESSIONS.get(session_id, {}).get("uploads", [])
#         data_metadata = None
#         for item in metadata:
#             if item.get("filename") == file_path:
#                 data_metadata = item
#                 break
                
#         if not data_metadata or "data" not in data_metadata:
#             return {"error": "No data found in session metadata."}
            
#         data = data_metadata["data"]
#         df = pd.DataFrame(data)
        
#         # Apply the confirmed mappings to rename columns
#         # We create a "rename map" from the mappings object
#         rename_map = {original_name: role for original_name, role in mappings.items() if role != "Ignore"}
#         df.rename(columns=rename_map, inplace=True)
        
#         # Drop the columns that the user marked as "Ignore"
#         columns_to_drop = [original_name for original_name, role in mappings.items() if role == "Ignore"]
#         df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

#         # Convert pandas objects to JSON-safe types
#         def make_serializable(obj):
#             """Convert pandas/numpy objects to JSON-serializable types"""
#             if pd.isna(obj):
#                 return None
#             elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
#                 return str(obj)
#             elif isinstance(obj, (np.integer, np.int64, np.int32)):
#                 return int(obj)
#             elif isinstance(obj, (np.floating, np.float64, np.float32)):
#                 return float(obj) if not np.isnan(obj) else None
#             elif isinstance(obj, np.ndarray):
#                 return obj.tolist()
#             elif hasattr(obj, 'item'):  # numpy scalars
#                 return obj.item()
#             else:
#                 return obj

#         # Calculate statistics with proper JSON serialization
#         null_counts = df.isnull().sum()
#         null_percentages = (df.isnull().sum() / len(df) * 100).round(2)
        
#         # Get descriptive stats only for numeric columns
#         numeric_df = df.select_dtypes(include=[np.number])
#         desc_stats = numeric_df.describe() if not numeric_df.empty else pd.DataFrame()
        
#         # Get categorical stats
#         categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
#         # Build categorical stats safely
#         categorical_stats = {}
#         for col in categorical_cols:
#             try:
#                 value_counts = df[col].value_counts()
#                 mode_values = df[col].mode()
                
#                 categorical_stats[col] = {
#                     "unique_values": int(df[col].nunique()),
#                     "most_frequent": str(mode_values.iloc[0]) if len(mode_values) > 0 else None,
#                     "frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
#                     "total_non_null": int(df[col].count())
#                 }
#             except (IndexError, ValueError) as e:
#                 # Handle edge cases where column has no valid data
#                 categorical_stats[col] = {
#                     "unique_values": 0,
#                     "most_frequent": None,
#                     "frequency": 0,
#                     "total_non_null": 0
#                 }
        
#         stats = {
#             "null_counts": {k: make_serializable(v) for k, v in null_counts.items()},
#             "null_percentages": {k: make_serializable(v) for k, v in null_percentages.items()},
#             "descriptive_stats": {
#                 col: {stat: make_serializable(desc_stats.loc[stat, col]) 
#                       for stat in desc_stats.index} 
#                 for col in desc_stats.columns
#             } if not desc_stats.empty else {},
#             "categorical_stats": categorical_stats,
#             "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
#             "total_rows": len(df),
#             "total_columns": len(df.columns)
#         }
        
#         return stats
        
#     except Exception as e:
#         logger.error(f"Error generating stats for {file_path}: {e}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         return {"error": f"Failed to generate stats: {str(e)}"}

# Add these new endpoints to your main.py

@app.get("/merge/available/{session_id}")
async def get_available_files_for_merge(session_id: str):
    """
    Get list of files available for merging (those with mappings)
    Uses the same naming logic as analysis endpoints.
    """
    try:
        if session_id not in SESSIONS:
            return {"error": f"Session {session_id} not found"}
        
        uploads = SESSIONS[session_id].get('uploads', [])
        available_files = []
        
        for upload in uploads:
            # Only include files that have mappings and are of type 'ingestion'
            if upload.get('mappings'):
                mapped_columns = [role for role in upload['mappings'].values() if role != "Ignore"]
                # Use consistent naming logic
                name = upload.get('name') or upload.get('filename') or upload.get('original_filename') or f"File_{upload.get('id')}"
                available_files.append({
                    'id': upload['id'],
                    'name': name,
                    'columns': mapped_columns,
                    'total_columns': len(mapped_columns),
                    'has_processed_data': bool(upload.get('processed_data')),
                    'is_merged': upload.get('is_merged', False)
                })
        
        return {
            "status": "success",
            "available_files": available_files,
            "total_available": len(available_files)
        }
        
    except Exception as e:
        logger.error(f"Error getting available files: {e}")
        return {"error": f"Failed to get available files: {str(e)}"}


@app.post("/merge/preview")
async def merge_preview(payload: dict):
    """
    Generate a preview of merged datasets
    """
    try:
        session_id = payload.get("session_id")
        file_ids = payload.get("file_ids", [])
        merge_strategy = payload.get("merge_strategy", "inner")
        join_columns = payload.get("join_columns", {})
        
        if len(file_ids) < 2:
            return {"error": "At least 2 files required for merging"}
        
        if session_id not in SESSIONS:
            return {"error": f"Session {session_id} not found"}
        
        uploads = SESSIONS[session_id].get('uploads', [])
        datasets = []
        
        for file_id in file_ids:
            upload = next((u for u in uploads if u.get('id') == file_id), None)
            if not upload:
                return {"error": f"File {file_id} not found"}
            
            # Use processed data if available, otherwise original data
            data = upload.get('processed_data') or upload.get('data')
            mappings = upload.get('mappings', {})
            
            if not data:
                return {"error": f"No data found for file {file_id}"}
            
            df = pd.DataFrame(data)
            
            # Apply mappings
            rename_map = {orig: role for orig, role in mappings.items() if role != "Ignore"}
            df.rename(columns=rename_map, inplace=True)
            
            # Drop ignored columns
            cols_to_drop = [orig for orig, role in mappings.items() if role == "Ignore"]
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            
            datasets.append({
                'id': file_id,
                'name': upload.get('name', f'Dataset_{file_id}'),
                'df': df
            })
        
        # Perform the merge
        merged_df = merge_datasets(datasets, merge_strategy, join_columns)
        
        # Generate preview with proper serialization
        preview_data = merged_df.head(10).to_dict('records')
        clean_preview_data = []
        for row in preview_data:
            clean_row = {k: make_serializable(v) for k, v in row.items()}
            clean_preview_data.append(clean_row)
        
        return {
            "status": "success",
            "preview": {
                "total_rows": len(merged_df),
                "total_columns": len(merged_df.columns),
                "columns": list(merged_df.columns),
                "sample_data": clean_preview_data
            }
        }
        
    except Exception as e:
        logger.error(f"Error in merge preview: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": f"Merge preview failed: {str(e)}"}

@app.post("/merge/execute")
async def merge_execute(payload: dict):
    """
    Execute the merge and create a new merged dataset
    """
    try:
        session_id = payload.get("session_id")
        file_ids = payload.get("file_ids", [])
        merge_strategy = payload.get("merge_strategy", "inner")
        join_columns = payload.get("join_columns", {})
        
        if len(file_ids) < 2:
            return {"error": "At least 2 files required for merging"}
        
        if session_id not in SESSIONS:
            return {"error": f"Session {session_id} not found"}
        
        uploads = SESSIONS[session_id].get('uploads', [])
        datasets = []
        
        # Same dataset loading logic as preview
        for file_id in file_ids:
            upload = next((u for u in uploads if u.get('id') == file_id), None)
            if not upload:
                return {"error": f"File {file_id} not found"}
            
            # Use processed data if available, otherwise original data
            data = upload.get('processed_data') or upload.get('data')
            mappings = upload.get('mappings', {})
            
            if not data:
                return {"error": f"No data found for file {file_id}"}
            
            df = pd.DataFrame(data)
            
            # Apply mappings
            rename_map = {orig: role for orig, role in mappings.items() if role != "Ignore"}
            df.rename(columns=rename_map, inplace=True)
            
            # Drop ignored columns
            cols_to_drop = [orig for orig, role in mappings.items() if role == "Ignore"]
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            
            datasets.append({
                'id': file_id,
                'name': upload.get('name', f'Dataset_{file_id}'),
                'df': df
            })
        
        # Execute the merge
        merged_df = merge_datasets(datasets, merge_strategy, join_columns)
        
        # Create new merged dataset entry
        merged_id = str(uuid.uuid4())
        file_names = [d['name'] for d in datasets]
        merged_name = f"Merged: {' + '.join(file_names)}"
        
        # Convert merged data with proper serialization
        full_data = merged_df.to_dict('records')
        clean_full_data = []
        for row in full_data:
            clean_row = {k: make_serializable(v) for k, v in row.items()}
            clean_full_data.append(clean_row)
        
        # Sample data for preview
        sample_data = merged_df.head(100).to_dict('records')
        clean_sample_data = []
        for row in sample_data:
            clean_row = {k: make_serializable(v) for k, v in row.items()}
            clean_sample_data.append(clean_row)
        
        merged_metadata = {
            'id': merged_id,
            'type': 'ingestion',  # Keep as ingestion type so it works with existing analysis
            'name': merged_name,
            'filename': merged_name,
            'source_files': file_ids,
            'merge_strategy': merge_strategy,
            'join_columns': join_columns,
            'data': clean_full_data,
            'sample_data': clean_sample_data,
            'columns': list(merged_df.columns),
            'mappings': {col: col for col in merged_df.columns},  # 1:1 mapping since already processed
            'is_merged': True,
            'merge_timestamp': time.time()
        }
        
        SESSIONS[session_id]['uploads'].append(merged_metadata)
        
        return {
            "status": "success",
            "merged_data": merged_metadata
        }
        
    except Exception as e:
        logger.error(f"Error in merge execution: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": f"Merge execution failed: {str(e)}"}

def merge_datasets(datasets, strategy, join_columns):
    """
    Core function to merge datasets based on strategy
    """
    dfs = [d['df'] for d in datasets]
    
    if strategy == 'concat':
        # Simple concatenation
        return pd.concat(dfs, ignore_index=True, sort=False)
    
    elif strategy in ['inner', 'outer', 'left']:
        # Join-based merging
        result = dfs[0]
        
        for i, next_df in enumerate(dfs[1:], 1):
            left_col = join_columns.get(datasets[0]['id'])
            right_col = join_columns.get(datasets[i]['id'])
            
            if not left_col or not right_col:
                raise ValueError(f"Join columns not specified for merge")
            
            result = result.merge(
                next_df, 
                left_on=left_col, 
                right_on=right_col, 
                how=strategy,
                suffixes=('', f'_file{i+1}')
            )
        
        return result
    
    else:
        raise ValueError(f"Unsupported merge strategy: {strategy}")


def make_serializable(obj):
    """Convert pandas/numpy objects to JSON-serializable types"""
    # Handle Series objects first
    if isinstance(obj, pd.Series):
        if len(obj) == 1:
            return make_serializable(obj.iloc[0])
        else:
            return [make_serializable(item) for item in obj]
    
    # Handle scalar pandas NA values
    if pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
        return str(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    else:
        return obj

# Replace the stats calculation section in your get_preprocessing_stats function
@app.post("/preprocess/stats")
async def get_preprocessing_stats(payload: dict):
    # It no longer looks up data in the session. It uses the data sent directly from the frontend.
    file_path = payload.get("file_path")
    mappings = payload.get("mappings")
    session_id = payload.get("session_id")
    
    if not file_path or not mappings:
        return {"error": "A file path and mappings are required to generate stats."}
    
    try:
        # Load the full dataset from the saved file path
        metadata = SESSIONS.get(session_id, {}).get("uploads", [])
        data_metadata = None
        for item in metadata:
            if item.get("filename") == file_path:
                data_metadata = item
                break
                
        if not data_metadata or "data" not in data_metadata:
            return {"error": "No data found in session metadata."}
            
        data = data_metadata["data"]
        df = pd.DataFrame(data)
        
        # Apply the confirmed mappings to rename columns
        # We create a "rename map" from the mappings object
        rename_map = {original_name: role for original_name, role in mappings.items() if role != "Ignore"}
        df.rename(columns=rename_map, inplace=True)
        
        # Drop the columns that the user marked as "Ignore"
        columns_to_drop = [original_name for original_name, role in mappings.items() if role == "Ignore"]
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        # Calculate statistics with proper JSON serialization
        null_counts = df.isnull().sum()
        null_percentages = (df.isnull().sum() / len(df) * 100).round(2)
        
        # Get descriptive stats only for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        desc_stats = numeric_df.describe() if not numeric_df.empty else pd.DataFrame()
        
        # Get categorical stats
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Build categorical stats safely
        categorical_stats = {}
        for col in categorical_cols:
            try:
                value_counts = df[col].value_counts()
                mode_values = df[col].mode()
                
                categorical_stats[col] = {
                    "unique_values": int(df[col].nunique()),
                    "most_frequent": str(mode_values.iloc[0]) if len(mode_values) > 0 else None,
                    "frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "total_non_null": int(df[col].count())
                }
            except (IndexError, ValueError) as e:
                # Handle edge cases where column has no valid data
                categorical_stats[col] = {
                    "unique_values": 0,
                    "most_frequent": None,
                    "frequency": 0,
                    "total_non_null": 0
                }
        
        # Fixed descriptive stats processing
        descriptive_stats = {}
        if not desc_stats.empty:
            for col in desc_stats.columns:
                descriptive_stats[col] = {}
                for stat in desc_stats.index:
                    try:
                        # Get the specific value and ensure it's a scalar
                        stat_value = desc_stats.at[stat, col]  # Use .at for scalar access
                        descriptive_stats[col][stat] = make_serializable(stat_value)
                    except (KeyError, IndexError) as e:
                        logger.warning(f"Could not get stat {stat} for column {col}: {e}")
                        descriptive_stats[col][stat] = None
        
        stats = {
            "null_counts": {k: make_serializable(v) for k, v in null_counts.items()},
            "null_percentages": {k: make_serializable(v) for k, v in null_percentages.items()},
            "descriptive_stats": descriptive_stats,
            "categorical_stats": categorical_stats,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "total_rows": len(df),
            "total_columns": len(df.columns)
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error generating stats for {file_path}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": f"Failed to generate stats: {str(e)}"}
    
from data.loader.preprocess import null_imputation

@app.post("/preprocess/null_imputation")
async def handle_null_imputation(payload: dict):
    """
    Handle null imputation for the dataset based on user choice.
    
    Payload should contain:
    - session_id: str
    - source_phase_id: str  
    - action: str ("continue_without_imputation" or "remove_null_columns")
    - threshold: float (optional, default 0.5 for remove_null_columns action)
    """
    try:
        session_id = payload.get("session_id")
        source_phase_id = payload.get("source_phase_id")
        action = payload.get("action")  # "continue_without_imputation" or "remove_null_columns"
        threshold = payload.get("threshold", 0.5)
        
        if not all([session_id, source_phase_id, action]):
            return {"error": "Missing required parameters: session_id, source_phase_id, or action"}
            
        if session_id not in SESSIONS:
            return {"error": f"Session {session_id} not found"}
            
        # Find the upload metadata in the session
        uploads = SESSIONS[session_id].get('uploads', [])
        target_upload = None
        
        for upload in uploads:
            if upload.get('id') == source_phase_id:
                target_upload = upload
                break
                
        if not target_upload:
            return {"error": f"Source phase {source_phase_id} not found in session"}
            
        # Get the full data and mappings
        full_data = target_upload.get('data')
        mappings = target_upload.get('mappings', {})
        
        if not full_data:
            return {"error": "No data found for this upload"}
            
        # Convert to DataFrame
        df = pd.DataFrame(full_data)
        
        # Apply mappings to rename columns and drop ignored columns
        rename_map = {original_name: role for original_name, role in mappings.items() if role != "Ignore"}
        df.rename(columns=rename_map, inplace=True)
        
        # Drop ignored columns
        columns_to_drop = [original_name for original_name, role in mappings.items() if role == "Ignore"]
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        original_shape = df.shape
        processed_df = df.copy()
        processing_summary = {
            "action_taken": action,
            "original_shape": original_shape,
            "columns_dropped": [],
            "imputation_applied": False
        }
        
        if action == "remove_null_columns":
            # Apply null imputation with the specified threshold
            columns_before = set(processed_df.columns)
            processed_df = null_imputation(processed_df, threshold=threshold)
            columns_after = set(processed_df.columns)
            dropped_columns = list(columns_before - columns_after)
            
            processing_summary.update({
                "columns_dropped": dropped_columns,
                "imputation_applied": True,
                "threshold_used": threshold,
                "final_shape": processed_df.shape
            })
            
        elif action == "continue_without_imputation":
            # No processing, just continue with the current data
            processing_summary.update({
                "final_shape": processed_df.shape,
                "message": "Continued without null imputation"
            })
        
        # Generate sample data for frontend (first 10 rows)
        sample_size = min(10, len(processed_df))
        sample_data = processed_df.head(sample_size).to_dict('records') if not processed_df.empty else []
        
        # Convert pandas objects to JSON-safe types for sample data
        def make_serializable(obj):
            if pd.isna(obj):
                return None
            elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
                return str(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj) if not np.isnan(obj) else None
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return obj
        
        # Clean sample data
        clean_sample_data = []
        for row in sample_data:
            clean_row = {k: make_serializable(v) for k, v in row.items()}
            clean_sample_data.append(clean_row)
        
        # Store the processed data back to session for future use
        target_upload['processed_data'] = processed_df.to_dict('records')
        target_upload['processed_sample_data'] = clean_sample_data
        target_upload['processing_summary'] = processing_summary
        
        # Calculate updated statistics for the processed data
        null_counts = processed_df.isnull().sum()
        null_percentages = (processed_df.isnull().sum() / len(processed_df) * 100).round(2)
        
        updated_stats = {
            "null_counts": {k: make_serializable(v) for k, v in null_counts.items()},
            "null_percentages": {k: make_serializable(v) for k, v in null_percentages.items()},
            "total_rows": len(processed_df),
            "total_columns": len(processed_df.columns),
            "dtypes": {col: str(dtype) for col, dtype in processed_df.dtypes.items()}
        }
        
        return {
            "status": "success",
            "processing_summary": processing_summary,
            "sample_data": clean_sample_data,
            "updated_stats": updated_stats,
            "message": f"Successfully processed data with action: {action}"
        }
        
    except Exception as e:
        logger.error(f"Error in null imputation: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": f"Failed to process null imputation: {str(e)}"}

@app.post("/chat")
async def chat_handler(payload: dict, background_tasks: BackgroundTasks):
    """Main chat handler with quota management"""
    try:
        user_message = payload.get("message", "")
        active_phase_context = payload.get("context", {})
        active_view = payload.get("active_view", "")
        session_id = payload.get("session_id", "default_session")
        logger.info(f"payload received: {list(payload.keys())}")
        
        
        logger.info(f"🔥 CHAT HANDLER: session={session_id}, view={active_view}")
        
        if not session_id or session_id not in SESSIONS:
            await manager.broadcast({
                "type": "agent_response",
                "payload": {"from": "bot", "text": "Error: Your session could not be found. Please try uploading a file again."}
            })
            return {"status": "error_no_session"}

        # Get session context - optimized for quota
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
        
        # Optimize context to save tokens
        trimmed_context = optimize_context_for_quota(trimmed_context, max_rows=3)
        
        await manager.broadcast({
            "type": "status_update",
            "payload": {"status": "thinking", "message": f"Analyzing '{filename}'..."}
        })

        if active_view == 'mapping' and active_phase_context:
            active_filename = active_phase_context.get('data', {}).get('filename')
            server_side_data = next((upload for upload in full_session_context if upload.get('filename') == active_filename), None)
            columns = []
            if server_side_data:
                data_for_ui = server_side_data.get('data', [])
                if data_for_ui and isinstance(data_for_ui, list) and len(data_for_ui) > 0:
                    columns = list(data_for_ui[0].keys())
            
            logger.info(f"🔥 MAPPING MODE: {len(columns)} columns")
            
            # Shortened prompt to save tokens
            prompt = f"""Analyze these columns for marine science data mapping: {columns}

Use submit_mapping_suggestion tool with ALL columns. For each column determine:
- role: ["Ignore", "Latitude", "Longitude", "Date", "Time", "Depth", "Temperature", "Salinity", "Oxygen", "Phosphate", "Silicate", "Nitrate", "Categorical", "Numerical"]
- data_type: ["numerical", "categorical", "datetime", "geospatial"]
- suggested_use: brief description

User request: "{user_message}"
"""

        elif active_view == 'merge' and payload.get('context', {}).get('type') == 'merge':
                logger.info(f"🔥 MERGE MODE: processing merge assistance request")
                
                selected_files_info = payload.get('context', {}).get('selected_files', [])
                current_strategy = payload.get('context', {}).get('merge_strategy', 'inner')
                current_join_columns = payload.get('context', {}).get('join_columns', {})
                
                prompt = f"""You are a marine data merging specialist. Analyze these datasets for optimal merging:

            Selected files for merging:
            {chr(10).join([f"- {f['name']}: columns=[{', '.join(f['columns'])}]" for f in selected_files_info])}

            Current settings:
            - Merge strategy: {current_strategy}
            - Join columns: {current_join_columns}

            User request: "{user_message}"

            Based on the column structures and marine science best practices, provide recommendations using the suggest_merge_strategy tool:

            1. Analyze the compatibility of these datasets
            2. Recommend the best merge strategy:
            - inner: Keep only matching records (for perfect overlap)
            - outer: Keep all records, fill missing values (for comprehensive analysis)  
            - left: Keep all from first file (when first is primary dataset)
            - concat: Stack vertically (when same structure, different time/space)

            3. Suggest appropriate join columns for each file (if not concat)
            4. Consider marine science patterns like:
            - Time series data (Date/Time columns)
            - Spatial data (Lat/Lon coordinates) 
            - Depth profiles (Depth measurements)
            - Station/Sample IDs
            - Common measurement parameters

            Always use the suggest_merge_strategy tool when providing merge recommendations."""
            

        elif active_view == 'analysis' and active_phase_context:
            # NEW: Use AnalysisAgentClass for analysis requests
            logger.info(f"🔥 ANALYSIS MODE: processing request")
            
            # Store the source phase ID in session for the analysis tool to use
            source_phase_id = active_phase_context.get('sourcePhaseId') or active_phase_context.get('id')
            if source_phase_id:
                SESSIONS[session_id]['pending_analysis'] = {'source_phase_id': source_phase_id}
            
            # Use analysis agent
            agent = AgentClass(session_id)
            
            # Get available columns from the source data
            source_upload = None
            for upload in full_session_context:
                if upload.get('id') == source_phase_id:
                    source_upload = upload
                    break
            
            available_columns = []
            original_columns = []
            if source_upload and source_upload.get('mappings'):
                # Get original columns (keys) and mapped roles (values)
                original_columns = [col for col, role in source_upload['mappings'].items() if role != "Ignore"]
                mapped_columns = [role for col, role in source_upload['mappings'].items() if role != "Ignore"]

            # Enhanced prompt for analysis
            prompt = f"""You are analyzing marine science data.
            Here are the available columns for plotting (original column names): {original_columns} {available_columns}
            Mapped roles for reference: {mapped_columns}

            User request: "{user_message}"

            Based on the request, determine the appropriate visualization and use the generate_analysis_and_plot tool with:
            - analysis_type: choose from line, scatter, bar, area, histogram
            - x_column: the original column name for x-axis  
            - y_column: the original column name for y-axis (if needed)
            - title: descriptive title
            - description: brief analysis description

            Always use the tool when users ask for plots, charts, visualizations, or data analysis."""
        else:
            # Shortened prompt for general queries
            prompt = f"""Context: {trimmed_context}
User: {user_message}

Provide a concise, helpful response about this marine data."""
        
        # Initialize agent for this session
        agent = AgentClass(session_id)
        SESSIONS[session_id]['last_message'] = user_message

        await manager.broadcast({
            "type": "status_update", 
            "payload": {"status": "processing", "message": "Processing your request..."}
        })
        
        logger.info("🔥 Calling agent with quota management...")
        response_text = await agent.chat(prompt)
        logger.info(f"🔥 Agent response received (length: {len(str(response_text))})")
        
        content = str(response_text) or "I've processed your request."
        
        await manager.broadcast({
            "type": "agent_response",
            "payload": {"from": "bot", "text": content}
        })

        return {"status": "message_processed"}
        
    except Exception as e:
        logger.error(f"❌ Chat handler error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Better error handling for quota issues
        error_message = "I encountered an error processing your request."
        if any(keyword in str(e).lower() for keyword in ['quota', 'exceeded', '429', 'resourceexhausted']):
            error_message = "I've reached the API quota limit. Please wait a few minutes before trying again."
        
        await manager.broadcast({
            "type": "agent_response", 
            "payload": {"from": "bot", "text": error_message}
        })
        return {"status": "error", "message": str(e)}