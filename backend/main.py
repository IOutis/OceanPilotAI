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
import requests


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
    Process analysis data and return visualization-ready data with insights
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
        print("In process_analysis_data, DataFrame columns:", df.columns.tolist())
       
        # Process the analysis with enhanced insights
        processed_data, visualization_config, statistical_summary, insights = perform_data_analysis(df, analysis_config, mappings)
        
        return {
            "status": "success",
            "data": processed_data,
            "config": visualization_config,
            "statistical_summary": statistical_summary,
            "insights": insights
        }
        
    except Exception as e:
        logger.error(f"Error processing analysis: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": f"Failed to process analysis: {str(e)}"}

# Replace the existing create_analysis_tool function with this:

def create_analysis_tool(session_id: str):
    @tool
    def generate_analysis_and_plot(analysis_type: str, x_column: str, y_column: str = None, 
                                 title: str = None, description: str = None) -> str:
        """
        Generate analysis results and visualization configuration based on user query.
        
        Args:
            analysis_type: str (one of: line, scatter, bar, area, histogram, correlation_matrix)
            x_column: str (name of the x-axis column - can be original name or role)
            y_column: str (name of the y-axis column - can be original name or role, optional for some plot types)
            title: str (title for the visualization)
            description: str (description of the analysis findings)
        """
        try:
            logger.info(f"🔥 ANALYSIS TOOL CALLED: {analysis_type} for session {session_id}")
            logger.info(f"🔥 ANALYSIS COLUMNS: x={x_column}, y={y_column}")
            
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
            
            if pending_analysis:
                source_phase_id = pending_analysis.get("source_phase_id")
            else:
                # Default to the most recent upload
                uploads = sess.get('uploads', [])
                if uploads:
                    source_phase_id = uploads[-1].get('id')
            
            if not source_phase_id:
                return "Error: No data source found for analysis"
            
            # Find the source upload to get mappings
            uploads = sess.get('uploads', [])
            source_upload = None
            for upload in uploads:
                if upload.get('id') == source_phase_id:
                    source_upload = upload
                    break
            
            if not source_upload:
                return "Error: Source upload not found"
            
            # Get mappings for better column resolution
            mappings = source_upload.get('mappings', {})
            
            # Process the analysis data with mappings
            analysis_config = {
                "analysis_type": analysis_type,
                "x_column": x_column,
                "y_column": y_column,
                "title": title or f"{analysis_type.title()} Analysis",
                "description": description or f"Analysis of {x_column}" + (f" vs {y_column}" if y_column else "")
            }
            
            # Call the analysis processing function
            try:
                response = requests.post('http://localhost:8000/analysis/process', 
                    json={
                        "session_id": session_id,
                        "source_phase_id": source_phase_id,
                        "analysis_config": analysis_config
                    })
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        # Send the processed data and config to frontend WITH insights
                        analysis_result = {
                            "type": "analysis_result",
                            "payload": {
                                "analysis_type": analysis_type,
                                "data": result["data"],
                                "config": result["config"],
                                "statistical_summary": result.get("statistical_summary", {}),
                                "insights": result.get("insights", [])
                            }
                        }
                        mgr.broadcast_sync(analysis_result)
                        
                        # Return structured data for LLM to interpret
                        config = result["config"]
                        insights = result.get("insights", [])
                        stats = result.get("statistical_summary", {})
                        data_count = len(result.get("data", []))
                        
                        x_actual = config.get("xAxis", x_column)
                        y_actual = config.get("yAxis", y_column)
                        
                        # Format the statistical results for the LLM
                        stats_text = format_stats_for_llm(stats, analysis_type)
                        insights_text = "\n".join([f"- {insight}" for insight in insights]) if insights else "- No automated insights generated"
                        
                        # Return structured information for the LLM to interpret
                        return f"""I've successfully created a {analysis_type} plot with {data_count} data points showing {x_actual} vs {y_actual}.

STATISTICAL RESULTS:
{stats_text}

AUTOMATED INSIGHTS:
{insights_text}

Based on these statistical results, let me explain what this plot reveals about your marine science data and its significance."""
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

def format_stats_for_llm(stats, analysis_type):
    """Format statistical results for LLM interpretation"""
    if not stats:
        return "No statistical summary available"
    
    formatted = []
    
    # Format based on analysis type
    if analysis_type == "scatter":
        if "correlation" in stats:
            formatted.append(f"Correlation coefficient: {stats['correlation']:.4f}")
        if "p_value" in stats:
            formatted.append(f"P-value: {stats['p_value']:.4f}")
        if "sample_size" in stats:
            formatted.append(f"Sample size: {stats['sample_size']}")
        if "x_mean" in stats and "y_mean" in stats:
            formatted.append(f"Mean values: X={stats['x_mean']:.3f}, Y={stats['y_mean']:.3f}")
        if "x_range" in stats and "y_range" in stats:
            x_min, x_max = stats['x_range']
            y_min, y_max = stats['y_range']
            formatted.append(f"Data ranges: X=[{x_min:.3f} to {x_max:.3f}], Y=[{y_min:.3f} to {y_max:.3f}]")
    
    elif analysis_type == "histogram":
        if "mean" in stats:
            formatted.append(f"Mean: {stats['mean']:.3f}")
        if "median" in stats:
            formatted.append(f"Median: {stats['median']:.3f}")
        if "std_dev" in stats:
            formatted.append(f"Standard deviation: {stats['std_dev']:.3f}")
        if "skewness" in stats:
            formatted.append(f"Skewness: {stats['skewness']:.3f}")
        if "range" in stats and isinstance(stats['range'], list):
            formatted.append(f"Range: {stats['range'][0]:.3f} to {stats['range'][1]:.3f}")
        if "quartiles" in stats:
            q1, q2, q3 = stats['quartiles']
            formatted.append(f"Quartiles: Q1={q1:.3f}, Q2={q2:.3f}, Q3={q3:.3f}")
    
    elif analysis_type == "bar":
        if "groups" in stats:
            formatted.append(f"Number of groups: {stats['groups']}")
        if "total_observations" in stats:
            formatted.append(f"Total observations: {stats['total_observations']}")
    
    # Add any other general statistics
    for key, value in stats.items():
        if key not in ["correlation", "p_value", "sample_size", "x_mean", "y_mean", "x_range", "y_range", 
                      "mean", "median", "std_dev", "skewness", "range", "quartiles", "groups", "total_observations"]:
            if isinstance(value, (int, float)):
                formatted.append(f"{key}: {value}")
            else:
                formatted.append(f"{key}: {str(value)}")
    
    return "\n".join(formatted) if formatted else "No statistical measures calculated"


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
            max_iterations=3,
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

# def perform_data_analysis(df: pd.DataFrame, analysis_config: Dict, mappings: Dict = None) -> tuple:
#     """
#     Perform the actual data analysis and return processed data and visualization config.
#     Accepts both mapped roles and original column names for x_column/y_column.
#     Returns:
#         tuple: (processed_data, visualization_config)
#     """
#     try:
#         print("DataFrame columns:", df.columns.tolist())
#         analysis_type = analysis_config["analysis_type"]
#         x_column = analysis_config["x_column"]
#         y_column = analysis_config.get("y_column")

#         # If mappings are not provided, use empty dict
#         mappings = mappings or {}

#         # Build mapping: original -> role and role -> original
#         original_to_role = {orig: role for orig, role in mappings.items() if role != "Ignore"}
#         role_to_original = {role: orig for orig, role in mappings.items() if role != "Ignore"}

#         # Columns in DataFrame after renaming
#         df_columns = set(df.columns)

#         # Map x_column and y_column to actual DataFrame columns
#         def resolve_column(col):
#             # If col is in DataFrame, use it
#             if col in df_columns:
#                 return col
#             # If col is a mapped role, use it
#             if col in original_to_role.values() and col in df_columns:
#                 return col
#             # If col is an original column name, map to role
#             if col in original_to_role and original_to_role[col] in df_columns:
#                 return original_to_role[col]
#             # If col is a role, map to original and check
#             if col in role_to_original and role_to_original[col] in df_columns:
#                 return role_to_original[col]
#             return None

#         x_col_in_df = resolve_column(x_column)
#         y_col_in_df = resolve_column(y_column) if y_column else None

#         missing = []
#         if not x_col_in_df:
#             missing.append(x_column)
#         if y_column and not y_col_in_df:
#             missing.append(y_column)
#         if missing:
#             raise KeyError(missing)

#         # Remove any infinite values and convert to numeric where possible
#         df_clean = df.copy()

#         # Convert columns to numeric if possible
#         for col in [x_col_in_df, y_col_in_df] if y_col_in_df else [x_col_in_df]:
#             if col in df_clean.columns:
#                 df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

#         # Remove rows with NaN values in key columns
#         # columns_to_check = [x_col_in_df, y_col_in_df] if y_col_in_df else [x_col_in_df]
#         # df_clean = df_clean.dropna(subset=columns_to_check)

#         if df_clean.empty:
#             raise ValueError("No valid data remaining after cleaning")

#         visualization_config = {
#             "type": analysis_type,
#             "config": {
#                 "title": analysis_config.get("title", f"{analysis_type.title()} Analysis"),
#                 "description": analysis_config.get("description", ""),
#                 "xAxis": x_col_in_df,
#                 "yAxis": y_col_in_df,
#                 "xAxisLabel": x_col_in_df.replace('_', ' ').title(),
#                 "yAxisLabel": y_col_in_df.replace('_', ' ').title() if y_col_in_df else ""
#             }
#         }
#         # Process data based on analysis type
#         if analysis_type == "scatter" or analysis_type == "line":
#             if not y_column:
#                 raise ValueError(f"{analysis_type} plot requires both x and y columns")
            
#             processed_data = df_clean[[x_column, y_column]].to_dict('records')
            
#             # Add correlation info for scatter plots
#             if analysis_type == "scatter" and len(df_clean) > 1:
#                 corr, p_value = pearsonr(df_clean[x_column], df_clean[y_column])
#                 visualization_config["config"]["description"] += f" Correlation: {corr:.3f} (p={p_value:.3f})"
        
#         elif analysis_type == "bar":
#             # For bar charts, group by x_column and aggregate y_column
#             if y_column:
#                 grouped = df_clean.groupby(x_column)[y_column].mean().reset_index()
#                 processed_data = grouped.to_dict('records')
#             else:
#                 # Count occurrences
#                 value_counts = df_clean[x_column].value_counts().reset_index()
#                 value_counts.columns = [x_column, 'count']
#                 processed_data = value_counts.to_dict('records')
#                 visualization_config["config"]["yAxis"] = 'count'
#                 visualization_config["config"]["yAxisLabel"] = 'Count'
        
#         elif analysis_type == "area":
#             if not y_column:
#                 raise ValueError("Area plot requires both x and y columns")
#             processed_data = df_clean[[x_column, y_column]].sort_values(x_column).to_dict('records')
        
#         elif analysis_type == "histogram":
#             # Create bins for histogram
#             hist_data, bin_edges = np.histogram(df_clean[x_column], bins=20)
#             bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
#             processed_data = [
#                 {x_column: float(center), 'frequency': int(count)} 
#                 for center, count in zip(bin_centers, hist_data)
#             ]
#             visualization_config["config"]["yAxis"] = 'frequency'
#             visualization_config["config"]["yAxisLabel"] = 'Frequency'
#             visualization_config["type"] = "bar"  # Use bar chart for histogram display
        
#         else:
#             raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
#         return processed_data, visualization_config
        
#     except Exception as e:
#         logger.error(f"Error in data analysis: {e}")
#         raise e
def perform_data_analysis(df: pd.DataFrame, analysis_config: Dict, mappings: Dict = None) -> tuple:
    """
    Enhanced data analysis that works with both original column names and role mappings.
    Returns:
        tuple: (processed_data, visualization_config, statistical_summary, insights)
    """
    try:
        print("DataFrame columns:", df.columns.tolist())
        analysis_type = analysis_config["analysis_type"]
        x_column = analysis_config["x_column"]
        y_column = analysis_config.get("y_column")

        # If mappings are provided, create lookup dictionaries
        if mappings:
            original_to_role = {orig: role for orig, role in mappings.items() if role != "Ignore"}
            role_to_original = {role: orig for orig, role in mappings.items() if role != "Ignore"}
        else:
            original_to_role = {}
            role_to_original = {}

        # Get columns actually available in DataFrame
        df_columns = set(df.columns)

        def resolve_column_name(requested_col):
            """Resolve a column name that could be either original or role name"""
            if requested_col in df_columns:
                return requested_col
            
            if requested_col in role_to_original:
                original_col = role_to_original[requested_col]
                if original_col in df_columns:
                    return original_col
            
            if requested_col in original_to_role:
                role = original_to_role[requested_col]
                if role in df_columns:
                    return role
            
            return None

        # Resolve the actual column names to use
        actual_x_column = resolve_column_name(x_column)
        actual_y_column = resolve_column_name(y_column) if y_column else None

        # Check if columns were found
        missing_columns = []
        if not actual_x_column:
            missing_columns.append(x_column)
        if y_column and not actual_y_column:
            missing_columns.append(y_column)
        
        if missing_columns:
            raise KeyError(f"Columns not found: {missing_columns}")

        # Clean the data
        df_clean = df.copy()

        # Convert to numeric if possible
        for col in [actual_x_column, actual_y_column] if actual_y_column else [actual_x_column]:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        if df_clean.empty:
            raise ValueError("No valid data remaining after cleaning")

        # Initialize containers for insights and statistical summary
        statistical_summary = {}
        insights = []

        def get_display_label(col_name):
            """Get a user-friendly label for display"""
            if mappings and col_name in original_to_role:
                role = original_to_role[col_name]
                return f"{col_name} ({role})"
            return col_name.replace('_', ' ').title()

        visualization_config = {
            "type": analysis_type,
            "config": {
                "title": analysis_config.get("title", f"{analysis_type.title()} Analysis"),
                "description": analysis_config.get("description", ""),
                "xAxis": actual_x_column,
                "yAxis": actual_y_column,
                "xAxisLabel": get_display_label(actual_x_column),
                "yAxisLabel": get_display_label(actual_y_column) if actual_y_column else "",
                "originalXColumn": x_column,
                "originalYColumn": y_column,
            }
        }

        # Process data based on analysis type with enhanced insights
        if analysis_type == "scatter" or analysis_type == "line":
            if not actual_y_column:
                raise ValueError(f"{analysis_type} plot requires both x and y columns")
            
            processed_data = df_clean[[actual_x_column, actual_y_column]].to_dict('records')
            
            # Enhanced analysis for scatter/line plots
            if len(df_clean) > 1:
                corr, p_value = pearsonr(df_clean[actual_x_column], df_clean[actual_y_column])
                statistical_summary = {
                    "correlation": corr,
                    "p_value": p_value,
                    "sample_size": len(df_clean),
                    "x_mean": df_clean[actual_x_column].mean(),
                    "y_mean": df_clean[actual_y_column].mean(),
                    "x_std": df_clean[actual_x_column].std(),
                    "y_std": df_clean[actual_y_column].std(),
                    "x_range": [df_clean[actual_x_column].min(), df_clean[actual_x_column].max()],
                    "y_range": [df_clean[actual_y_column].min(), df_clean[actual_y_column].max()]
                }
                
                # Generate insights based on correlation
                if abs(corr) > 0.7:
                    strength = "strong"
                elif abs(corr) > 0.5:
                    strength = "moderate"
                elif abs(corr) > 0.3:
                    strength = "weak"
                else:
                    strength = "very weak"
                
                direction = "positive" if corr > 0 else "negative"
                
                insights.append(f"There is a {strength} {direction} correlation (r = {corr:.3f}) between {get_display_label(actual_x_column)} and {get_display_label(actual_y_column)}.")
                
                if p_value < 0.05:
                    insights.append(f"The correlation is statistically significant (p = {p_value:.3f}), suggesting a meaningful relationship.")
                else:
                    insights.append(f"The correlation is not statistically significant (p = {p_value:.3f}), suggesting the relationship may be due to random variation.")
                
                # Marine science specific insights
                if 'temperature' in actual_x_column.lower() and 'depth' in actual_y_column.lower():
                    if corr < -0.3:
                        insights.append("This negative relationship between temperature and depth is typical in ocean profiles, showing thermocline structure.")
                elif 'salinity' in actual_x_column.lower() and 'temperature' in actual_y_column.lower():
                    insights.append("This temperature-salinity relationship can reveal water mass characteristics and mixing processes.")
        
        elif analysis_type == "bar":
            if actual_y_column:
                grouped = df_clean.groupby(actual_x_column)[actual_y_column].agg(['mean', 'std', 'count']).reset_index()
                processed_data = grouped.rename(columns={'mean': actual_y_column}).to_dict('records')
                
                # Statistical summary for grouped data
                statistical_summary = {
                    "groups": len(grouped),
                    "total_observations": grouped['count'].sum(),
                    "mean_values": grouped['mean'].tolist(),
                    "std_values": grouped['std'].tolist(),
                    "group_sizes": grouped['count'].tolist()
                }
                
                # Generate insights
                max_group = grouped.loc[grouped['mean'].idxmax(), actual_x_column]
                min_group = grouped.loc[grouped['mean'].idxmin(), actual_x_column]
                overall_mean = grouped['mean'].mean()
                
                insights.append(f"The analysis shows {len(grouped)} different groups with varying {get_display_label(actual_y_column)} values.")
                insights.append(f"'{max_group}' has the highest average value, while '{min_group}' has the lowest.")
                insights.append(f"The overall average across all groups is {overall_mean:.2f}.")
                
            else:
                # Count occurrences
                value_counts = df_clean[actual_x_column].value_counts().reset_index()
                value_counts.columns = [actual_x_column, 'count']
                processed_data = value_counts.to_dict('records')
                
                statistical_summary = {
                    "unique_categories": len(value_counts),
                    "total_observations": value_counts['count'].sum(),
                    "most_common": value_counts.iloc[0][actual_x_column],
                    "most_common_count": value_counts.iloc[0]['count']
                }
                
                insights.append(f"The data contains {len(value_counts)} unique categories in {get_display_label(actual_x_column)}.")
                insights.append(f"'{statistical_summary['most_common']}' is the most frequent category with {statistical_summary['most_common_count']} occurrences.")
                
                visualization_config["config"]["yAxis"] = 'count'
                visualization_config["config"]["yAxisLabel"] = 'Count'
        
        elif analysis_type == "area":
            if not actual_y_column:
                raise ValueError("Area plot requires both x and y columns")
            processed_data = df_clean[[actual_x_column, actual_y_column]].sort_values(actual_x_column).to_dict('records')
            
            # Time series or sequential analysis
            statistical_summary = {
                "data_points": len(df_clean),
                "trend": "increasing" if df_clean[actual_y_column].iloc[-1] > df_clean[actual_y_column].iloc[0] else "decreasing",
                "total_change": df_clean[actual_y_column].iloc[-1] - df_clean[actual_y_column].iloc[0],
                "peak_value": df_clean[actual_y_column].max(),
                "valley_value": df_clean[actual_y_column].min()
            }
            
            insights.append(f"The area plot shows {statistical_summary['trend']} trend over the range of {get_display_label(actual_x_column)}.")
            insights.append(f"Total change from start to end: {statistical_summary['total_change']:.2f}")
        
        elif analysis_type == "histogram":
            # Create bins for histogram
            hist_data, bin_edges = np.histogram(df_clean[actual_x_column], bins=20)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            processed_data = [
                {actual_x_column: float(center), 'frequency': int(count)} 
                for center, count in zip(bin_centers, hist_data)
            ]
            
            # Enhanced statistical analysis for distribution
            data_values = df_clean[actual_x_column].dropna()
            statistical_summary = {
                "mean": data_values.mean(),
                "median": data_values.median(),
                "std_dev": data_values.std(),
                "skewness": data_values.skew(),
                "kurtosis": data_values.kurtosis(),
                "range": [data_values.min(), data_values.max()],
                "quartiles": [data_values.quantile(0.25), data_values.quantile(0.5), data_values.quantile(0.75)]
            }
            
            # Generate distribution insights
            if abs(statistical_summary["skewness"]) < 0.5:
                distribution_shape = "approximately normal"
            elif statistical_summary["skewness"] > 0.5:
                distribution_shape = "right-skewed (positively skewed)"
            else:
                distribution_shape = "left-skewed (negatively skewed)"
            
            insights.append(f"The distribution of {get_display_label(actual_x_column)} appears {distribution_shape}.")
            insights.append(f"Mean: {statistical_summary['mean']:.2f}, Median: {statistical_summary['median']:.2f}, Standard Deviation: {statistical_summary['std_dev']:.2f}")
            
            if abs(statistical_summary['mean'] - statistical_summary['median']) > statistical_summary['std_dev'] * 0.1:
                insights.append("The difference between mean and median suggests some asymmetry in the data distribution.")
            
            visualization_config["config"]["yAxis"] = 'frequency'
            visualization_config["config"]["yAxisLabel"] = 'Frequency'
            visualization_config["type"] = "bar"  # Use bar chart for histogram display
        
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        # Add general data quality insights
        if actual_y_column:
            missing_x = df[actual_x_column].isnull().sum()
            missing_y = df[actual_y_column].isnull().sum()
            if missing_x > 0 or missing_y > 0:
                insights.append(f"Note: {missing_x + missing_y} data points were excluded due to missing values.")
        
        return processed_data, visualization_config, statistical_summary, insights
        
    except Exception as e:
        logger.error(f"Error in data analysis: {e}")
        raise e

# Update the analysis tool to work better with mixed column naming
# Replace the existing create_analysis_tool function with this enhanced version:

def create_analysis_tool(session_id: str):
    @tool
    def generate_analysis_and_plot(analysis_type: str, x_column: str, y_column: str = None, 
                                 title: str = None, description: str = None) -> str:
        """
        Generate analysis results and visualization configuration based on user query.
        
        Args:
            analysis_type: str (one of: line, scatter, bar, area, histogram, correlation_matrix)
            x_column: str (name of the x-axis column - can be original name or role)
            y_column: str (name of the y-axis column - can be original name or role, optional for some plot types)
            title: str (title for the visualization)
            description: str (description of the analysis findings)
        """
        try:
            logger.info(f"🔥 ANALYSIS TOOL CALLED: {analysis_type} for session {session_id}")
            logger.info(f"🔥 ANALYSIS COLUMNS: x={x_column}, y={y_column}")
            
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
            
            if pending_analysis:
                source_phase_id = pending_analysis.get("source_phase_id")
            else:
                # Default to the most recent upload
                uploads = sess.get('uploads', [])
                if uploads:
                    source_phase_id = uploads[-1].get('id')
            
            if not source_phase_id:
                return "Error: No data source found for analysis"
            
            # Find the source upload to get mappings
            uploads = sess.get('uploads', [])
            source_upload = None
            for upload in uploads:
                if upload.get('id') == source_phase_id:
                    source_upload = upload
                    break
            
            if not source_upload:
                return "Error: Source upload not found"
            
            # Get mappings for better column resolution
            mappings = source_upload.get('mappings', {})
            
            # Process the analysis data with mappings
            analysis_config = {
                "analysis_type": analysis_type,
                "x_column": x_column,
                "y_column": y_column,
                "title": title or f"{analysis_type.title()} Analysis",
                "description": description or f"Analysis of {x_column}" + (f" vs {y_column}" if y_column else "")
            }
            
            # Call the analysis processing function
            try:
                response = requests.post('http://localhost:8000/analysis/process', 
                    json={
                        "session_id": session_id,
                        "source_phase_id": source_phase_id,
                        "analysis_config": analysis_config
                    })
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        # Send the processed data and config to frontend WITH insights
                        analysis_result = {
                            "type": "analysis_result",
                            "payload": {
                                "analysis_type": analysis_type,
                                "data": result["data"],
                                "config": result["config"],
                                "statistical_summary": result.get("statistical_summary", {}),
                                "insights": result.get("insights", [])
                            }
                        }
                        mgr.broadcast_sync(analysis_result)
                        
                        # Return enhanced context for deep marine science interpretation
                        config = result["config"]
                        insights = result.get("insights", [])
                        stats = result.get("statistical_summary", {})
                        data_count = len(result.get("data", []))
                        raw_data = result.get("data", [])
                        
                        x_actual = config.get("xAxis", x_column)
                        y_actual = config.get("yAxis", y_column)
                        
                        # Generate enhanced context for LLM interpretation
                        interpretation_context = generate_marine_science_context(
                            analysis_type, x_actual, y_actual, stats, raw_data, mappings
                        )
                        
                        # Return structured information for deep LLM interpretation
                        return f"""VISUALIZATION_CREATED: {analysis_type} plot with {data_count} data points showing {x_actual} vs {y_actual}.

MARINE SCIENCE ANALYSIS CONTEXT:
{interpretation_context}

STATISTICAL RESULTS:
{format_enhanced_stats(stats, analysis_type)}

AUTOMATED INSIGHTS:
{chr(10).join([f"- {insight}" for insight in insights]) if insights else "- No automated insights generated"}

Now provide expert marine science interpretation focusing on:
1. OCEANOGRAPHIC SIGNIFICANCE - What do these patterns reveal about marine processes?
2. DATA QUALITY ASSESSMENT - Are there suspicious patterns, outliers, or measurement issues?
3. ECOLOGICAL IMPLICATIONS - What does this mean for marine life and ecosystem health?
4. ENVIRONMENTAL CONTEXT - How do these values compare to expected marine conditions?
5. RESEARCH IMPLICATIONS - What further investigation or monitoring is recommended?

Be specific about what makes patterns normal, unusual, or concerning in marine science context."""
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

def generate_marine_science_context(analysis_type, x_variable, y_variable, stats, raw_data, mappings):
    """Generate rich context for marine science interpretation"""
    context_lines = []
    
    # Variable identification and expected ranges
    variable_context = identify_marine_variables(x_variable, y_variable)
    context_lines.extend(variable_context)
    
    # Data quality flags based on marine science knowledge
    quality_flags = assess_marine_data_quality(x_variable, y_variable, stats, raw_data)
    if quality_flags:
        context_lines.append("\nDATA QUALITY INDICATORS:")
        context_lines.extend(quality_flags)
    
    # Pattern significance based on analysis type
    pattern_significance = get_pattern_significance(analysis_type, x_variable, y_variable, stats)
    if pattern_significance:
        context_lines.append("\nPATTERN SIGNIFICANCE:")
        context_lines.extend(pattern_significance)
    
    return "\n".join(context_lines)

def identify_marine_variables(x_var, y_var):
    """Identify marine variables and provide expected ranges/context"""
    context = []
    
    variables = [x_var, y_var] if y_var else [x_var]
    
    for var in variables:
        var_lower = var.lower()
        
        if 'oxygen' in var_lower:
            context.append(f"- {var}: Marine dissolved oxygen")
            context.append(f"  • Normal range: 0-15 mg/L (0-10 mL/L)")
            context.append(f"  • Hypoxic: < 2 mg/L, Anoxic: < 0.5 mg/L")
            context.append(f"  • Supersaturation: > 10-12 mg/L (may indicate measurement error or algal bloom)")
            
        elif 'temperature' in var_lower:
            context.append(f"- {var}: Water temperature")
            context.append(f"  • Ocean surface: typically -2°C to 35°C")
            context.append(f"  • Deep water: typically 0-4°C")
            context.append(f"  • Suspicious: values outside -2°C to 40°C range")
            
        elif 'salinity' in var_lower:
            context.append(f"- {var}: Water salinity")
            context.append(f"  • Open ocean: typically 34-37 PSU")
            context.append(f"  • Coastal/estuarine: 0-35 PSU")
            context.append(f"  • Suspicious: values > 42 PSU or negative values")
            
        elif 'depth' in var_lower:
            context.append(f"- {var}: Water depth")
            context.append(f"  • Should be positive values")
            context.append(f"  • Suspicious: negative depths or extremely deep values")
            
        elif any(nutrient in var_lower for nutrient in ['phosphate', 'nitrate', 'silicate']):
            context.append(f"- {var}: Marine nutrient")
            context.append(f"  • Typical range: 0-50 μM (varies by nutrient and location)")
            context.append(f"  • Suspicious: negative values or extremely high concentrations")
            
        elif 'ph' in var_lower:
            context.append(f"- {var}: Seawater pH")
            context.append(f"  • Normal range: 7.5-8.5")
            context.append(f"  • Ocean acidification: declining pH trends")
            context.append(f"  • Suspicious: values outside 6.0-9.0 range")
    
    return context

def assess_marine_data_quality(x_var, y_var, stats, raw_data):
    """Assess data quality based on marine science knowledge"""
    flags = []
    
    # Check for impossible values based on variable type
    if 'oxygen' in x_var.lower():
        if stats.get('range') and len(stats['range']) == 2:
            min_val, max_val = stats['range']
            if min_val < -1:
                flags.append(f"⚠️ Negative oxygen values detected (min: {min_val:.2f}) - likely measurement error")
            if max_val > 20:
                flags.append(f"⚠️ Extremely high oxygen values (max: {max_val:.2f}) - check calibration")
        
        # Check for hypoxic/anoxic conditions
        if stats.get('mean') and stats['mean'] < 2:
            flags.append(f"🔴 Mean oxygen < 2 mg/L indicates hypoxic conditions")
        
        # Check distribution for bimodality (might indicate mixed water masses)
        if 'skewness' in stats and abs(stats['skewness']) > 1.5:
            flags.append(f"📊 Highly skewed distribution (skewness: {stats['skewness']:.2f}) - investigate data sources")
    
    elif 'temperature' in x_var.lower():
        if stats.get('range') and len(stats['range']) == 2:
            min_val, max_val = stats['range']
            if min_val < -3 or max_val > 40:
                flags.append(f"⚠️ Temperature outside typical marine range ({min_val:.1f}°C to {max_val:.1f}°C)")
    
    elif 'salinity' in x_var.lower():
        if stats.get('range') and len(stats['range']) == 2:
            min_val, max_val = stats['range']
            if min_val < 0 or max_val > 42:
                flags.append(f"⚠️ Salinity outside realistic range ({min_val:.1f} to {max_val:.1f} PSU)")
    
    # General quality checks
    if 'sample_size' in stats and stats['sample_size'] < 10:
        flags.append(f"📉 Small sample size (n={stats['sample_size']}) - limited statistical reliability")
    
    if 'std_dev' in stats and 'mean' in stats and stats['mean'] != 0:
        cv = abs(stats['std_dev'] / stats['mean'])
        if cv > 2:
            flags.append(f"📊 High coefficient of variation ({cv:.2f}) - very variable data")
    
    return flags

def get_pattern_significance(analysis_type, x_var, y_var, stats):
    """Determine what patterns might signify in marine science context"""
    significance = []
    
    if analysis_type == "histogram":
        # Distribution shape significance
        if 'skewness' in stats:
            skew = stats['skewness']
            if abs(skew) < 0.5:
                significance.append("- Normal distribution may indicate well-mixed water masses")
            elif skew > 1:
                significance.append("- Right-skewed: few high values, possibly indicating contamination or instrument drift")
            elif skew < -1:
                significance.append("- Left-skewed: few low values, might indicate detection limit issues")
        
        # Bimodal patterns
        if 'oxygen' in x_var.lower():
            significance.append("- Bimodal oxygen distribution could indicate:")
            significance.append("  • Mixing of different water masses")
            significance.append("  • Seasonal stratification effects")
            significance.append("  • Day/night respiration cycles")
    
    elif analysis_type == "scatter":
        # Correlation significance in marine context
        if 'correlation' in stats:
            corr = stats['correlation']
            x_lower, y_lower = x_var.lower(), y_var.lower() if y_var else ""
            
            if 'temperature' in x_lower and 'depth' in y_lower:
                if corr < -0.5:
                    significance.append("- Strong negative T-D correlation indicates thermocline presence")
                elif abs(corr) < 0.3:
                    significance.append("- Weak T-D correlation might indicate mixed water column")
            
            elif 'oxygen' in x_lower and 'depth' in y_lower:
                if corr < -0.5:
                    significance.append("- Oxygen depletion with depth suggests consumption processes")
                elif corr > 0.3:
                    significance.append("- Oxygen increase with depth is unusual - check data quality")
            
            elif 'temperature' in x_lower and 'salinity' in y_lower:
                significance.append("- T-S relationship reveals water mass characteristics")
                if abs(corr) > 0.7:
                    significance.append("- Strong T-S correlation indicates conservative mixing")
    
    return significance

def format_enhanced_stats(stats, analysis_type):
    """Format statistical results with marine science context"""
    if not stats:
        return "No statistical summary available"
    
    formatted = []
    
    # Core statistics with interpretation hints
    for key, value in stats.items():
        if key == "correlation" and value is not None:
            strength = "very strong" if abs(value) > 0.8 else "strong" if abs(value) > 0.6 else "moderate" if abs(value) > 0.4 else "weak"
            formatted.append(f"Correlation: {value:.4f} ({strength} {'positive' if value > 0 else 'negative'} relationship)")
        elif key == "p_value" and value is not None:
            sig = "highly significant" if value < 0.001 else "significant" if value < 0.05 else "not significant"
            formatted.append(f"P-value: {value:.4f} ({sig})")
        elif key == "mean" and value is not None:
            formatted.append(f"Mean: {value:.3f}")
        elif key == "median" and value is not None:
            formatted.append(f"Median: {value:.3f}")
        elif key == "std_dev" and value is not None:
            formatted.append(f"Standard deviation: {value:.3f}")
        elif key == "skewness" and value is not None:
            shape = "normal" if abs(value) < 0.5 else "right-skewed" if value > 0.5 else "left-skewed"
            formatted.append(f"Skewness: {value:.3f} ({shape} distribution)")
        elif key == "range" and isinstance(value, list) and len(value) == 2:
            formatted.append(f"Range: {value[0]:.3f} to {value[1]:.3f} (span: {value[1]-value[0]:.3f})")
        elif key == "sample_size":
            reliability = "high" if value > 100 else "moderate" if value > 30 else "low"
            formatted.append(f"Sample size: {value} ({reliability} statistical reliability)")
        elif key in ["groups", "total_observations"]:
            formatted.append(f"{key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(formatted) if formatted else "No statistical measures calculated"



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
def to_scalar(val):
    """Safely convert pandas/numpy objects to Python scalars for stats."""
    if val is None:
        return None
    if pd.isna(val):
        return None
    if isinstance(val, pd.Series):
        if len(val) == 1:
            return to_scalar(val.iloc[0])
        else:
            return int(len(val))
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return make_serializable(val.item())
        else:
            return int(val.size)
    if hasattr(val, "item") and not isinstance(val, pd.Series):
        return make_serializable(val.item())
    # Handle NaN and inf values
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    try:
        return int(val)
    except Exception:
        try:
            return float(val) if not (isinstance(val, float) and (math.isnan(val) or math.isinf(val))) else None
        except Exception:
            return make_serializable(val)

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
            count_val = to_scalar(col_data.count())
            missing_count = to_scalar(df[col].isna().sum())
            missing_percent = to_scalar(df[col].isna().mean() * 100)
            if len(col_data) > 0:
                stats_summary[col] = {
                    "type": "numeric",
                    "count": make_serializable(count_val),
                    "mean": make_serializable(col_data.mean()),
                    "std": make_serializable(col_data.std()),
                    "min": make_serializable(col_data.min()),
                    "max": make_serializable(col_data.max()),
                    "median": make_serializable(col_data.median()),
                    "q25": make_serializable(col_data.quantile(0.25)),
                    "q75": make_serializable(col_data.quantile(0.75)),
                    "missing_count": make_serializable(missing_count),
                    "missing_percent": make_serializable(missing_percent)
                }

        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            col_data = df[col].dropna()
            value_counts = col_data.value_counts()
            top_value = value_counts.index[0] if len(value_counts) > 0 else None
            top_count = to_scalar(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            unique_count = to_scalar(value_counts.nunique()) if len(value_counts) > 0 else 0
            count_val = to_scalar(col_data.count())
            missing_count = to_scalar(df[col].isna().sum())
            missing_percent = to_scalar(df[col].isna().mean() * 100)
            stats_summary[col] = {
                "type": "categorical",
                "count": make_serializable(count_val),
                "unique_count": make_serializable(unique_count),
                "top_value": make_serializable(top_value),
                "top_count": make_serializable(top_count),
                "missing_count": make_serializable(missing_count),
                "missing_percent": make_serializable(missing_percent)
            }

        # Correlation matrix for numeric columns
        correlation_matrix = None
        if len(numeric_cols) > 1:
            corr_df = df[numeric_cols].corr()
            correlation_matrix = {
                "columns": list(corr_df.columns),
                "matrix": [[make_serializable(v) for v in row] for row in corr_df.values.tolist()]
            }

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
                # Get ORIGINAL column names (keys), not roles (values)
                original_columns = [orig_col for orig_col, role in upload['mappings'].items() if role != "Ignore"]
                
                name = upload.get('name') or upload.get('filename') or upload.get('original_filename') or f"File_{upload.get('id')}"
                available_files.append({
                    'id': upload['id'],
                    'name': name,
                    'columns': original_columns,  # These are now ORIGINAL column names
                    'mappings': upload['mappings'],  # Include mappings for reference
                    'total_columns': len(original_columns),
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
    Generate a preview of merged datasets with original column names preserved
    """
    try:
        session_id = payload.get("session_id")
        file_ids = payload.get("file_ids", [])
        merge_strategy = payload.get("merge_strategy", "inner")
        join_columns = payload.get("join_columns", {})
        preserve_original_names = payload.get("preserve_original_names", True)  # New option
        
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
            
            # NEW: Handle column naming based on preserve_original_names flag
            if preserve_original_names:
                # Keep original column names, just drop ignored columns
                columns_to_drop = [orig for orig, role in mappings.items() if role == "Ignore"]
                df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
                
                # Store the column mappings for metadata
                column_metadata = {orig: role for orig, role in mappings.items() if role != "Ignore"}
            else:
                # Original behavior: rename to roles and drop ignored
                rename_map = {orig: role for orig, role in mappings.items() if role != "Ignore"}
                df.rename(columns=rename_map, inplace=True)
                
                columns_to_drop = [orig for orig, role in mappings.items() if role == "Ignore"]
                df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
                
                column_metadata = {role: role for role in rename_map.values()}
            
            datasets.append({
                'id': file_id,
                'name': upload.get('name', f'Dataset_{file_id}'),
                'df': df,
                'column_metadata': column_metadata,
                'original_mappings': mappings
            })
        
        # Perform the merge with enhanced metadata tracking
        merged_df, merge_metadata = merge_datasets_enhanced(datasets, merge_strategy, join_columns)
        
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
                "sample_data": clean_preview_data,
                "column_metadata": merge_metadata,
                "preserve_original_names": preserve_original_names
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
    Execute the merge and create a new merged dataset with original column names
    """
    try:
        session_id = payload.get("session_id")
        file_ids = payload.get("file_ids", [])
        merge_strategy = payload.get("merge_strategy", "inner")
        join_columns = payload.get("join_columns", {})
        preserve_original_names = payload.get("preserve_original_names", True)
        
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
            
            data = upload.get('processed_data') or upload.get('data')
            mappings = upload.get('mappings', {})
            
            if not data:
                return {"error": f"No data found for file {file_id}"}
            
            df = pd.DataFrame(data)
            
            # Handle column naming consistently with preview
            if preserve_original_names:
                columns_to_drop = [orig for orig, role in mappings.items() if role == "Ignore"]
                df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
                column_metadata = {orig: role for orig, role in mappings.items() if role != "Ignore"}
            else:
                rename_map = {orig: role for orig, role in mappings.items() if role != "Ignore"}
                df.rename(columns=rename_map, inplace=True)
                columns_to_drop = [orig for orig, role in mappings.items() if role == "Ignore"]
                df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
                column_metadata = {role: role for role in rename_map.values()}
            
            datasets.append({
                'id': file_id,
                'name': upload.get('name', f'Dataset_{file_id}'),
                'df': df,
                'column_metadata': column_metadata,
                'original_mappings': mappings
            })
        
        # Execute the merge with enhanced metadata
        merged_df, merge_metadata = merge_datasets_enhanced(datasets, merge_strategy, join_columns)
        
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
        
        # Create enhanced mappings that preserve the original column names and their roles
        if preserve_original_names:
            # For merged dataset, create mappings that show original names -> roles
            enhanced_mappings = merge_metadata.get('column_mappings', {})
        else:
            # Original behavior: 1:1 mapping since columns are already roles
            enhanced_mappings = {col: col for col in merged_df.columns}
        
        merged_metadata = {
            'id': merged_id,
            'type': 'ingestion',
            'name': merged_name,
            'filename': merged_name,
            'source_files': file_ids,
            'merge_strategy': merge_strategy,
            'join_columns': join_columns,
            'data': clean_full_data,
            'sample_data': clean_sample_data,
            'columns': list(merged_df.columns),
            'mappings': enhanced_mappings,
            'column_metadata': merge_metadata,
            'preserve_original_names': preserve_original_names,
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

def merge_datasets_enhanced(datasets, strategy, join_columns):
    """
    Enhanced merge function that preserves original column names and tracks metadata
    """
    dfs = [d['df'] for d in datasets]
    
    # Collect all column metadata for the final result
    all_column_metadata = {}
    source_file_mapping = {}  # Track which columns came from which files
    
    if strategy == 'concat':
        # Simple concatenation with metadata tracking
        result = pd.concat(dfs, ignore_index=True, sort=False)
        
        # For concat, merge all column metadata
        for i, dataset in enumerate(datasets):
            for col, role in dataset['column_metadata'].items():
                if col in result.columns:
                    all_column_metadata[col] = role
                    if col not in source_file_mapping:
                        source_file_mapping[col] = []
                    source_file_mapping[col].append(dataset['name'])
        
    elif strategy in ['inner', 'outer', 'left']:
        # Join-based merging with conflict resolution
        result = dfs[0].copy()
        
        # Initialize metadata from first dataset
        for col, role in datasets[0]['column_metadata'].items():
            if col in result.columns:
                all_column_metadata[col] = role
                source_file_mapping[col] = [datasets[0]['name']]
        
        for i, next_df in enumerate(dfs[1:], 1):
            left_col = join_columns.get(datasets[0]['id'])
            right_col = join_columns.get(datasets[i]['id'])
            
            if not left_col or not right_col:
                raise ValueError(f"Join columns not specified for merge")
            
            # Handle column name conflicts before merge
            right_df = next_df.copy()
            rename_map = {}
            
            for col in right_df.columns:
                if col != right_col and col in result.columns:
                    # Create unique column name
                    new_name = f"{col}_{datasets[i]['name']}"
                    rename_map[col] = new_name
            
            if rename_map:
                right_df.rename(columns=rename_map, inplace=True)
            
            # Perform the merge
            result = result.merge(
                right_df, 
                left_on=left_col, 
                right_on=right_col, 
                how=strategy,
                suffixes=('', f'_from_{datasets[i]["name"]}')
            )
            
            # Update metadata for new columns
            for col, role in datasets[i]['column_metadata'].items():
                final_col_name = rename_map.get(col, col)
                if final_col_name in result.columns:
                    all_column_metadata[final_col_name] = role
                    source_file_mapping[final_col_name] = [datasets[i]['name']]
    
    else:
        raise ValueError(f"Unsupported merge strategy: {strategy}")
    
    merge_metadata = {
        'column_mappings': all_column_metadata,
        'source_files': source_file_mapping,
        'merge_strategy': strategy,
        'join_columns_used': join_columns if strategy != 'concat' else None,
        'total_source_datasets': len(datasets),
        'dataset_names': [d['name'] for d in datasets]
    }
    
    return result, merge_metadata

import pandas as pd
import numpy as np
from typing import Any, Union, List, Dict
import math

def make_serializable(obj: Any) -> Any:
    """
    Convert pandas/numpy objects to JSON-serializable types with comprehensive handling
    """
    # Handle None first
    if obj is None:
        return None
    
    # Handle pandas Series
    if isinstance(obj, pd.Series):
        if len(obj) == 1:
            return make_serializable(obj.iloc[0])
        else:
            return [make_serializable(item) for item in obj.tolist()]
    
    # Handle pandas Index
    if isinstance(obj, pd.Index):
        return [make_serializable(item) for item in obj.tolist()]
    
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return [make_serializable(item) for item in obj.tolist()]
    
    # Handle pandas NA/NaT and numpy NaN
    if pd.isna(obj) or (isinstance(obj, float) and math.isnan(obj)):
        return None
    
    # Handle pandas Timestamp and datetime objects
    if isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
        return str(obj)
    
    # Handle numpy integer types
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, 
                       np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    
    # Handle numpy float types (including checking for NaN/inf)
    if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    
    # Handle numpy bool
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # Handle numpy complex (convert to string representation)
    if isinstance(obj, np.complex128):
        return str(obj)
    
    # Handle regular Python float (check for NaN/inf)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    
    # Handle lists and tuples recursively
    if isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    
    # Handle dictionaries recursively
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    
    # Handle objects with .item() method (numpy scalars)
    if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
        try:
            return make_serializable(obj.item())
        except (ValueError, TypeError):
            return str(obj)
    
    # Handle objects with .tolist() method
    if hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
        try:
            return make_serializable(obj.tolist())
        except (ValueError, TypeError):
            return str(obj)
    
    # For basic Python types, return as-is
    if isinstance(obj, (int, str, bool)):
        return obj
    
    # Last resort: convert to string
    return str(obj)

def safe_df_to_records(df: pd.DataFrame) -> List[Dict]:
    """
    Safely convert DataFrame to records with proper serialization
    """
    if df.empty:
        return []
    
    # Replace inf and -inf with None before conversion
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    
    # Convert to records
    records = df_clean.to_dict('records')
    
    # Apply make_serializable to each record
    serialized_records = []
    for record in records:
        serialized_record = {}
        for key, value in record.items():
            serialized_record[str(key)] = make_serializable(value)
        serialized_records.append(serialized_record)
    
    return serialized_records

# Alternative helper function for column info serialization
def serialize_column_info(info_dict: Dict) -> Dict:
    """
    Safely serialize column information dictionaries
    """
    serialized = {}
    for key, value in info_dict.items():
        serialized[str(key)] = make_serializable(value)
    return serialized

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
        # rename_map = {original_name: role for original_name, role in mappings.items() if role != "Ignore"}
        # df.rename(columns=rename_map, inplace=True)
        
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
        # rename_map = {original_name: role for original_name, role in mappings.items() if role != "Ignore"}
        # df.rename(columns=rename_map, inplace=True)
        
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

# Add these additional imports to your main.py file (some may already exist)
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from fastapi import Query
import re
from datetime import datetime
import operator
import io

# Pydantic models for request/response validation
class FilterCondition(BaseModel):
    column: str
    operator: str = Field(..., description="eq, ne, gt, gte, lt, lte, contains, starts_with, ends_with, in, not_in, is_null, not_null, between")
    value: Optional[Union[str, int, float, List[Union[str, int, float]]]] = None
    case_sensitive: bool = False

class DataPlaygroundRequest(BaseModel):
    session_id: str
    source_phase_id: str
    filters: List[FilterCondition] = []
    search_term: Optional[str] = None
    search_columns: Optional[List[str]] = None
    sort_column: Optional[str] = None
    sort_order: str = Field(default="asc", description="asc or desc")
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=1000)
    columns: Optional[List[str]] = None  # Select specific columns
    group_by: Optional[str] = None
    aggregate_functions: Optional[Dict[str, str]] = None  # {"column": "function"}

class ColumnInfo(BaseModel):
    name: str
    data_type: str
    unique_values: int
    null_count: int
    null_percentage: float
    sample_values: List[Any]
    is_numeric: bool
    is_datetime: bool
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None

@app.get("/playground/{session_id}/{source_phase_id}/info")
async def get_data_info(session_id: str, source_phase_id: str):
    """Get comprehensive information about the dataset"""
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
        
        # Use processed data if available, otherwise use original data
        data = source_upload.get('data')
        # mappings = source_upload.get('mappings', {})
        
        if not data:
            return {"error": "No data found"}
        
        df = pd.DataFrame(data)
        print(df.columns)
        
        # Apply column mappings and filtering
        # if mappings:
        #     columns_to_drop = [original_name for original_name, role in mappings.items() if role == "Ignore"]
        #     df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        # Generate column information
        column_info = []
        for col in df.columns:
            col_data = df[col]
            is_numeric = pd.api.types.is_numeric_dtype(col_data)
            is_datetime = pd.api.types.is_datetime64_any_dtype(col_data)
            
            info = ColumnInfo(
                name=col,
                data_type=str(col_data.dtype),
                unique_values=int(col_data.nunique()),
                null_count=int(col_data.isnull().sum()),
                null_percentage=float(col_data.isnull().mean() * 100),
                sample_values=[make_serializable(v) for v in col_data.dropna().unique()[:10].tolist()],
                is_numeric=is_numeric,
                is_datetime=is_datetime
            )
            
            if is_numeric:
                info.min_value = make_serializable(col_data.min())
                info.max_value = make_serializable(col_data.max())
                info.mean_value = make_serializable(col_data.mean())
            elif is_datetime:
                info.min_value = make_serializable(col_data.min())
                info.max_value = make_serializable(col_data.max())
            
            column_info.append(info.dict())
        
        return {
            "status": "success",
            "dataset_info": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_info": column_info,
                # "memory_usage": df.memory_usage(deep=True).sum(),
                # "dtypes_summary": df.dtypes.value_counts().to_dict()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting data info: {e}")
        return {"error": f"Failed to get data info: {str(e)}"}

@app.post("/playground/data")
async def get_filtered_data(request: DataPlaygroundRequest):
    """Main data playground endpoint with comprehensive filtering and exploration"""
    try:
        session_id = request.session_id
        source_phase_id = request.source_phase_id
        
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
        data = source_upload.get('data')
        # mappings = source_upload.get('mappings', {})
        
        if not data:
            return {"error": "No data found"}
        
        df = pd.DataFrame(data)
        original_count = len(df)
        
        # Apply column mappings
        # if mappings:
        #     columns_to_drop = [original_name for original_name, role in mappings.items() if role == "Ignore"]
        #     df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        # Apply filters
        for filter_condition in request.filters:
            df = apply_filter(df, filter_condition)
        
        # Apply global search
        if request.search_term:
            df = apply_search(df, request.search_term, request.search_columns)
        
        filtered_count = len(df)
        
        # Apply column selection
        if request.columns:
            available_columns = [col for col in request.columns if col in df.columns]
            if available_columns:
                df = df[available_columns]
        
        # Apply grouping and aggregation
        if request.group_by and request.group_by in df.columns:
            df = apply_grouping(df, request.group_by, request.aggregate_functions)
        
        # Apply sorting
        if request.sort_column and request.sort_column in df.columns:
            ascending = request.sort_order.lower() == "asc"
            df = df.sort_values(by=request.sort_column, ascending=ascending, na_position='last')
        
        # Apply pagination
        offset = (request.page - 1) * request.page_size
        total_pages = (len(df) + request.page_size - 1) // request.page_size
        paginated_df = df.iloc[offset:offset + request.page_size]
        
        # Convert to serializable format using safe method
        data_records = safe_df_to_records(paginated_df)
        
        return {
            "status": "success",
            "data": data_records,
            "pagination": {
                "current_page": request.page,
                "page_size": request.page_size,
                "total_rows": len(df),
                "total_pages": total_pages,
                "has_next": request.page < total_pages,
                "has_previous": request.page > 1,
                "showing_from": offset + 1 if len(df) > 0 else 0,
                "showing_to": min(offset + request.page_size, len(df))
            },
            "summary": {
                "original_count": original_count,
                "filtered_count": filtered_count,
                "columns_shown": list(paginated_df.columns) if not paginated_df.empty else [],
                "filters_applied": len(request.filters),
                "search_applied": bool(request.search_term)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in data playground: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": f"Failed to filter data: {str(e)}"}

def safe_df_to_records(df: pd.DataFrame) -> List[Dict]:
    """Safely convert DataFrame to records using existing make_serializable function"""
    records = df.to_dict('records')
    return [make_serializable(record) for record in records]

def apply_filter(df: pd.DataFrame, filter_condition: FilterCondition) -> pd.DataFrame:
    """Apply a single filter condition to the dataframe"""
    column = filter_condition.column
    op = filter_condition.operator
    value = filter_condition.value
    case_sensitive = filter_condition.case_sensitive
    
    if column not in df.columns:
        logger.warning(f"Column {column} not found in dataframe")
        return df
    
    col_data = df[column]
    
    # Handle null/not_null operators first (don't need value conversion)
    if op == "is_null":
        return df[col_data.isnull()]
    elif op == "not_null":
        return df[col_data.notnull()]
    
    # Skip filter if value is None or empty string for other operators
    if value is None or (isinstance(value, str) and value.strip() == ""):
        logger.warning(f"Empty value for filter on column {column}, skipping")
        return df
    
    try:
        # Type conversion logic
        converted_value = convert_value_to_column_type(col_data, value)
        
        if op == "eq":
            return df[col_data == converted_value]
        elif op == "ne":
            return df[col_data != converted_value]
        elif op == "gt":
            return df[col_data > converted_value]
        elif op == "gte":
            return df[col_data >= converted_value]
        elif op == "lt":
            return df[col_data < converted_value]
        elif op == "lte":
            return df[col_data <= converted_value]
        elif op == "contains":
            if case_sensitive:
                return df[col_data.astype(str).str.contains(str(converted_value), na=False)]
            else:
                return df[col_data.astype(str).str.contains(str(converted_value), case=False, na=False)]
        elif op == "starts_with":
            if case_sensitive:
                return df[col_data.astype(str).str.startswith(str(converted_value), na=False)]
            else:
                return df[col_data.astype(str).str.lower().str.startswith(str(converted_value).lower(), na=False)]
        elif op == "ends_with":
            if case_sensitive:
                return df[col_data.astype(str).str.endswith(str(converted_value), na=False)]
            else:
                return df[col_data.astype(str).str.lower().str.endswith(str(converted_value).lower(), na=False)]
        elif op == "in":
            if isinstance(converted_value, list):
                return df[col_data.isin(converted_value)]
            else:
                return df[col_data == converted_value]
        elif op == "not_in":
            if isinstance(converted_value, list):
                return df[~col_data.isin(converted_value)]
            else:
                return df[col_data != converted_value]
        elif op == "between":
            if isinstance(value, list) and len(value) == 2:
                # Convert both values
                min_val = convert_value_to_column_type(col_data, value[0])
                max_val = convert_value_to_column_type(col_data, value[1])
                return df[(col_data >= min_val) & (col_data <= max_val)]
            else:
                logger.warning(f"Between filter requires list of 2 values, got: {value}")
                return df
        else:
            logger.warning(f"Unknown operator: {op}")
            return df
            
    except Exception as e:
        logger.error(f"Error applying filter {op} to column {column}: {e}")
        logger.error(f"Column dtype: {col_data.dtype}, Value: {value}, Type: {type(value)}")
        return df

def convert_value_to_column_type(col_data: pd.Series, value):
    """Convert filter value to match the column's data type"""
    if value is None:
        return None
    
    # Handle list values (for 'in', 'not_in', 'between' operators)
    if isinstance(value, list):
        return [convert_single_value_to_column_type(col_data, v) for v in value]
    
    return convert_single_value_to_column_type(col_data, value)

def convert_single_value_to_column_type(col_data: pd.Series, value):
    """Convert a single value to match the column's data type"""
    if value is None or pd.isna(value):
        return None
    
    col_dtype = col_data.dtype
    
    try:
        # Numeric types
        if pd.api.types.is_numeric_dtype(col_data):
            if pd.api.types.is_integer_dtype(col_data):
                return int(float(str(value)))  # Handle cases like "5.0" -> 5
            else:
                return float(str(value))
        
        # DateTime types
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            if isinstance(value, str):
                return pd.to_datetime(value)
            return value
        
        # Boolean types
        elif pd.api.types.is_bool_dtype(col_data):
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        
        # String/object types - keep as string
        else:
            return str(value)
    
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not convert value '{value}' to column type {col_dtype}: {e}")
        # Fall back to string representation
        return str(value)

def apply_search(df: pd.DataFrame, search_term: str, search_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Apply global search across specified columns or all text columns"""
    if not search_term.strip():
        return df
    
    search_cols = search_columns if search_columns else df.select_dtypes(include=['object', 'string']).columns.tolist()
    search_cols = [col for col in search_cols if col in df.columns]
    
    if not search_cols:
        return df
    
    # Create a mask for rows that match the search term
    mask = pd.Series([False] * len(df), index=df.index)
    
    for col in search_cols:
        try:
            col_mask = df[col].astype(str).str.contains(search_term, case=False, na=False)
            mask = mask | col_mask
        except Exception as e:
            logger.warning(f"Error searching in column {col}: {e}")
            continue
    
    return df[mask]

def apply_grouping(df: pd.DataFrame, group_by: str, aggregate_functions: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Apply grouping and aggregation"""
    if group_by not in df.columns:
        return df
    
    if not aggregate_functions:
        # Default aggregation - count
        return df.groupby(group_by).size().reset_index(name='count')
    
    # Apply specified aggregations
    agg_dict = {}
    for col, func in aggregate_functions.items():
        if col in df.columns:
            if func in ['sum', 'mean', 'median', 'min', 'max', 'std', 'count']:
                agg_dict[col] = func
            else:
                logger.warning(f"Unknown aggregation function: {func}")
    
    if agg_dict:
        try:
            return df.groupby(group_by).agg(agg_dict).reset_index()
        except Exception as e:
            logger.error(f"Error in grouping: {e}")
            return df.groupby(group_by).size().reset_index(name='count')
    
    return df

@app.get("/playground/{session_id}/{source_phase_id}/column/{column_name}/values")
async def get_column_unique_values(
    session_id: str, 
    source_phase_id: str, 
    column_name: str,
    limit: int = Query(default=100, le=1000),
    search: Optional[str] = Query(default=None)
):
    """Get unique values for a specific column (useful for filter dropdowns)"""
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
        
        data = source_upload.get('processed_data') or source_upload.get('data')
        mappings = source_upload.get('mappings', {})
        
        if not data:
            return {"error": "No data found"}
        
        df = pd.DataFrame(data)
        
        # Apply mappings
        if mappings:
            columns_to_drop = [original_name for original_name, role in mappings.items() if role == "Ignore"]
            df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        if column_name not in df.columns:
            return {"error": f"Column {column_name} not found"}
        
        # Get unique values
        unique_values = df[column_name].dropna().unique()
        
        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            unique_values = [v for v in unique_values if search_lower in str(v).lower()]
        
        # Limit results and serialize safely
        limited_values = unique_values[:limit]
        serialized_values = [make_serializable(v) for v in limited_values]
        
        return {
            "status": "success",
            "column": column_name,
            "values": serialized_values,
            "total_unique": len(unique_values),
            "showing": len(serialized_values)
        }
        
    except Exception as e:
        logger.error(f"Error getting column values: {e}")
        return {"error": f"Failed to get column values: {str(e)}"}

@app.post("/playground/export")
async def export_filtered_data(request: DataPlaygroundRequest, 
                              export_format: str = Query(default="csv", description="csv, json, excel")):
    """Export filtered data in various formats"""
    try:
        # Get filtered data (reuse the same logic as get_filtered_data but without pagination)
        request.page_size = 10000  # Large page size for export
        request.page = 1
        
        result = await get_filtered_data(request)
        
        if result.get("status") != "success":
            return result
        
        df_data = result["data"]
        df = pd.DataFrame(df_data)
        
        if export_format == "csv":
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            return {
                "status": "success",
                "data": csv_buffer.getvalue(),
                "filename": f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "content_type": "text/csv"
            }
        elif export_format == "json":
            return {
                "status": "success",
                "data": df.to_json(orient='records', indent=2),
                "filename": f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "content_type": "application/json"
            }
        elif export_format == "excel":
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_data = excel_buffer.getvalue()
            import base64
            return {
                "status": "success",
                "data": base64.b64encode(excel_data).decode(),
                "filename": f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "encoding": "base64"
            }
        else:
            return {"error": f"Unsupported export format: {export_format}"}
    
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return {"error": f"Failed to export data: {str(e)}"}

# Additional helpful endpoints for the data playground

@app.get("/playground/{session_id}/{source_phase_id}/summary")
async def get_data_summary(session_id: str, source_phase_id: str):
    """Get a quick statistical summary of the dataset"""
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
        
        data = source_upload.get('processed_data') or source_upload.get('data')
        mappings = source_upload.get('mappings', {})
        
        if not data:
            return {"error": "No data found"}
        
        df = pd.DataFrame(data)
        
        # Apply mappings
        if mappings:
            columns_to_drop = [original_name for original_name, role in mappings.items() if role == "Ignore"]
            df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        # Calculate summary statistics
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": make_serializable(df.memory_usage(deep=True).sum() / 1024 / 1024),
            "columns": []
        }
        
        for col in df.columns:
            col_data = df[col]
            col_summary = {
                "name": col,
                "dtype": str(col_data.dtype),
                "non_null_count": int(col_data.count()),
                "null_count": int(col_data.isnull().sum()),
                "null_percentage": make_serializable(col_data.isnull().mean() * 100),
                "unique_count": int(col_data.nunique()),
                "is_numeric": pd.api.types.is_numeric_dtype(col_data),
                "is_datetime": pd.api.types.is_datetime64_any_dtype(col_data)
            }
            
            if col_summary["is_numeric"]:
                col_summary.update({
                    "min": make_serializable(col_data.min()),
                    "max": make_serializable(col_data.max()),
                    "mean": make_serializable(col_data.mean()),
                    "std": make_serializable(col_data.std())
                })
            
            summary["columns"].append(col_summary)
        
        return {"status": "success", "summary": summary}
        
    except Exception as e:
        logger.error(f"Error getting data summary: {e}")
        return {"error": f"Failed to get data summary: {str(e)}"}

@app.post("/playground/validate_filters")
async def validate_filters(request: dict):
    """Validate filter conditions before applying them"""
    try:
        session_id = request.get("session_id")
        source_phase_id = request.get("source_phase_id")
        filters = request.get("filters", [])
        
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
        
        data = source_upload.get('processed_data') or source_upload.get('data')
        if not data:
            return {"error": "No data found"}
        
        df = pd.DataFrame(data)
        validation_results = []
        
        for filter_data in filters:
            result = {
                "filter": filter_data,
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            column = filter_data.get("column")
            operator = filter_data.get("operator")
            value = filter_data.get("value")
            
            # Validate column exists
            if column not in df.columns:
                result["valid"] = False
                result["errors"].append(f"Column '{column}' not found")
            else:
                col_data = df[column]
                
                # Validate operator for column type
                if operator in ["gt", "gte", "lt", "lte", "between"] and not pd.api.types.is_numeric_dtype(col_data):
                    result["warnings"].append(f"Numeric comparison on non-numeric column '{column}'")
                
                # Validate value format
                if operator == "between" and not isinstance(value, list):
                    result["valid"] = False
                    result["errors"].append("'between' operator requires a list of 2 values")
                elif operator == "between" and isinstance(value, list) and len(value) != 2:
                    result["valid"] = False
                    result["errors"].append("'between' operator requires exactly 2 values")
                elif operator in ["in", "not_in"] and not isinstance(value, list):
                    result["warnings"].append("'in'/'not_in' operators work best with lists")
            
            validation_results.append(result)
        
        return {"status": "success", "validations": validation_results}
        
    except Exception as e:
        logger.error(f"Error validating filters: {e}")
        return {"error": f"Failed to validate filters: {str(e)}"}

@app.get("/playground/{session_id}/{source_phase_id}/sample")
async def get_data_sample(
    session_id: str, 
    source_phase_id: str,
    sample_size: int = Query(default=10, ge=1, le=1000),
    random_sample: bool = Query(default=False)
):
    """Get a sample of the data for quick preview"""
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
        
        data = source_upload.get('processed_data') or source_upload.get('data')
        mappings = source_upload.get('mappings', {})
        
        if not data:
            return {"error": "No data found"}
        
        df = pd.DataFrame(data)
        
        # Apply mappings
        if mappings:
            columns_to_drop = [original_name for original_name, role in mappings.items() if role == "Ignore"]
            df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        # Get sample
        if random_sample and len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
        else:
            sample_df = df.head(sample_size)
        
        # Convert to safe records
        sample_data = safe_df_to_records(sample_df)
        
        return {
            "status": "success",
            "sample_data": sample_data,
            "sample_size": len(sample_data),
            "total_rows": len(df),
            "columns": list(sample_df.columns),
            "is_random": random_sample and len(df) > sample_size
        }
        
    except Exception as e:
        logger.error(f"Error getting data sample: {e}")
        return {"error": f"Failed to get data sample: {str(e)}"}


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

            Steps to follow:
                1. Use the generate_analysis_and_plot tool with appropriate parameters:
                - analysis_type: choose from line, scatter, bar, area, histogram
                - x_column: use original column name for x-axis  
                - y_column: use original column name for y-axis (if needed)
                - title: descriptive title
                - description: brief analysis description

                2. When the tool returns statistical results, provide comprehensive interpretation covering:
                - What the statistical measures mean in practical terms
                - Marine science context and oceanographic implications  
                - Data quality observations and reliability
                - Ecological or environmental significance
                - Practical applications for research or monitoring


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