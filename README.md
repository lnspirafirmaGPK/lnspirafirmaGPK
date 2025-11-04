## Hi there 👋

<!--
**lnspirafirmaGPK/lnspirafirmaGPK** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
Import uuid
import os  #(NEW) Import os for environment variables
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from openai import AsyncOpenAI  #(CHANGED) Import AsyncOpenAI for FastAPI

# --- 1. Application Setup ---
app = FastAPI(
    title="AI Service Template",
    description="A professional template for deploying AI models (NLP, Video Gen) as a stateless API.",
    version="1.0.0"
)

# --- (NEW) 2. AI Client Setup ---
# Read the API Key from environment variables (Best Practice)
# (NEW) อย่าลืมตั้งค่า: export GEMINI_API_KEY="your_actual_key_here"
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# (CHANGED) We use AsyncOpenAI for compatibility with FastAPI's async functions
ai_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --- 3. Pydantic Data Models (Request/Response Schemas) ---

class MessageInput(BaseModel):
    """Input schema for text analysis."""
    content: str = Field(..., description="The text content to be analyzed.", min_length=1)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class AnalysisResponse(BaseModel):
    """Output schema for text analysis."""
    analysis_id: str = Field(..., description="Unique ID for this analysis request.")
    timestamp: datetime = Field(..., description="Time of the analysis.")
    is_intentful: bool = Field(False, description="Whether the model detected a specific intent.")
    emotional_tone: str = Field("Neutral", description="Detected emotional tone.")
    confidence: float = Field(0.0, description="Confidence score (0.0 to 1.0).")
    entities: List[Dict[str, Any]] = Field([], description="List of recognized entities.")
    
    # (NEW) Add a field to hold the embedding results
    embedding: List[float] = Field([], description="Vector embedding of the content.")


class VideoRequest(BaseModel):
    """Input schema for video generation."""
    prompt: str = Field(..., description="The text prompt to generate video from.", min_length=10)
    style: Optional[str] = Field("cinematic", description="Desired style of the video.")

class VideoResponse(BaseModel):
    """Output schema for video generation."""
    job_id: str = Field(..., description="The unique ID for the generation job.")
    status: str = Field("pending", description="Current status of the job (e.g., pending, processing, completed).")
    estimated_duration_sec: int = Field(60, description="Estimated time in seconds.")
    
# --- 4. AI Service Logic (Placeholders) ---

# (CHANGED) This function is now modified to call the actual AI service
async def analyze_text_placeholder(text: str) -> Dict[str, Any]:
    """
    (Placeholder Function - MODIFIED)
    This function now calls the AI embedding service and provides
    mock data for other fields.
    """
    print(f"Analyzing text and getting embedding for: '{text[:20]}...'")
    
    # --- (NEW) Call AI Service ---
    embedding_vector = []  # Default to empty list
    try:
        # (CHANGED) Await the async client call
        embedding_response = await ai_client.embeddings.create(
            input=text,
            model="gemini-embedding-001"
        )
        embedding_vector = embedding_response.data[0].embedding
        print(f"Successfully retrieved embedding with dimension: {len(embedding_vector)}")
    except Exception as e:
        print(f"Error calling AI embedding service: {e}")
        # In a real system, you might want to log this error
        # We'll continue with an empty embedding
    
    # --- (MODIFIED) Keep mock data for other fields ---
    entities_found = []
    if "agio" in text.lower():
        entities_found.append({"text": "agio", "type": "PERSON", "start": 0, "end": 4})

    is_intentful = len(text) > 10 or bool(entities_found)
    tone = "Positive" if "good" in text.lower() else "Neutral"
    
    return {
        "is_intentful": is_intentful,
        "emotional_tone": tone,
        "confidence": 0.85 if is_intentful else 0.3, # (Changed from random)
        "entities": entities_found,
        "embedding": embedding_vector  # (NEW) Pass the real embedding back
    }

async def generate_video_placeholder(prompt: str, style: str) -> Dict[str, Any]:
    """
    (Placeholder Function - Unchanged)
    This is where you integrate your real Video Generation API.
    """
    # TODO: Replace this with your actual video generation API call.
    print(f"Submitting video job for prompt: '{prompt}' with style: '{style}'")
    job_id = f"vid_job_{uuid.uuid4()}"
    
    return {
        "job_id": job_id,
        "status": "pending",
        "estimated_duration_sec": 90 # (Changed from random)
    }

# --- 5. API Endpoints ---

@app.post("/analyze", 
          response_model=AnalysisResponse,
          summary="Analyze Text Content (with Embedding)", # (Changed)
          description="Analyzes input text for intent, emotion, entities, and generates an embedding.")
async def analyze_message_endpoint(input_data: MessageInput):
    """
    Analyzes a single message using the AI backend.
    """
    if not input_data.content:
        raise HTTPException(status_code=422, detail="Content cannot be empty.")

    try:
        # Call the modified placeholder service
        analysis_results = await analyze_text_placeholder(input_data.content)
        
        # Format the response according to the pydantic model
        return AnalysisResponse(
            analysis_id=f"anl_{uuid.uuid4()}",
            timestamp=datetime.utcnow(),
            is_intentful=analysis_results.get("is_intentful", False),
            emotional_tone=analysis_results.get("emotional_tone", "Neutral"),
            confidence=analysis_results.get("confidence", 0.0),
            entities=analysis_results.get("entities", []),
            embedding=analysis_results.get("embedding", []) # (NEW) Map the embedding
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/generate-video", 
          response_model=VideoResponse,
          summary="Submit Video Generation Job",
          description="Submits a text prompt to the video generation service.")
async def generate_video_endpoint(request: VideoRequest):
    """
    Submits a prompt to generate an AI video.
    Returns a job ID for status tracking.
    """
    try:
        video_job = await generate_video_placeholder(request.prompt, request.style)
        
        return VideoResponse(
            job_id=video_job.get("job_id"),
            status=video_job.get("status", "failed"),
            estimated_duration_sec=video_job.get("estimated_duration_sec", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# (CHANGED) 'random' import is no longer needed