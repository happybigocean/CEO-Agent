import os
import asyncio
import logging
from typing import Optional

from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.os import AgentOS
from fastapi.middleware.cors import CORSMiddleware
from agno.db.postgres import PostgresDb
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector  # Remove SearchType import
from fastapi import FastAPI, HTTPException
from agno.tools.reasoning import ReasoningTools
from agno.models.google import Gemini
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.tools.googlesearch import GoogleSearchTools

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_CONNECTION_STRING = os.getenv("SUPABASE_CONNECTION_STRING")
ENV = os.getenv("ENV", "development")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env")

SUPABASE_DB_URL = (
    SUPABASE_CONNECTION_STRING
)

supabase_db = PostgresDb(
    db_url=SUPABASE_DB_URL,
    id="supabase-main",
    knowledge_table="knowledge_contents",
)

vector_db = PgVector(
    table_name="vectors", 
    db_url=SUPABASE_DB_URL,
    embedder=OpenAIEmbedder(),
)

knowledge = Knowledge(
    name="CEO Knowledge Base",
    description="Comprehensive knowledge base for CEO Agent",
    contents_db=supabase_db,
    vector_db=vector_db,
)

ceo_agent = Agent(
    name="CEO Agent",
    model=Gemini(
        id="gemini-2.5-pro",
        #max_output_tokens=5000,
        search=True,
        ),
    tools=[ReasoningTools()],
    description="You are a news agent that helps users find the latest news.",
    instructions=[
        "You are a CEO assistant",
        "Given a topic by the user, respond with 4 latest news items about that topic.",
        "Search for 10 news items and select the top 4 unique items.",
        "If you find relevant information in the knowledge base, use it to provide highly detailed, thorough, and structured answers with proper citations.",
        "Even for very simple questions, always provide context, explanations, examples, and insights so the user receives maximum clarity.",
    ],
    user_id="ceo_user",
    db=supabase_db,
    knowledge=knowledge,
    num_history_runs=10, 
    markdown=True,
)

# Initialize AgentOS
agent_os = AgentOS(
    os_id="netcorobo",
    description="NetcoRobo Enhanced",
    agents=[ceo_agent],
)

app = agent_os.get_app()

if ENV == "production":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.post("/loadknowledge")
async def load_knowledge():
    """Load knowledge into the database"""
    try:
        logger.info("Starting knowledge loading process...")
        
        await knowledge.clear()
        
        # Load the Thai recipes PDF
        result1 = await knowledge.add_content_async(
            name="Thai Recipes Collection",
            url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",
            metadata={"user_tag": "Thai Recipes", "content_type": "recipes", "source": "PDF"}
        )
        
        logger.info("Knowledge loading completed successfully")
        
        return {
            "status": "success", 
            "message": "Knowledge base loaded successfully",
        }
        
    except Exception as e:
        logger.error(f"Error loading knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading knowledge: {str(e)}")

@app.get("/knowledge/status")
async def get_knowledge_status():
    """Check knowledge base status and list contents"""
    try:
        # Try to get knowledge contents
        contents = []
        try:
            # This might vary based on your agno version
            knowledge_items = await knowledge.get_contents_async(limit=20)
            if knowledge_items:
                contents = [
                    {
                        "name": getattr(item, 'name', 'Unknown'),
                        "id": getattr(item, 'id', None),
                        "metadata": getattr(item, 'metadata', {}),
                        "content_preview": str(getattr(item, 'content', ''))[:200] + "..." if len(str(getattr(item, 'content', ''))) > 200 else str(getattr(item, 'content', ''))
                    }
                    for item in knowledge_items
                ]
        except Exception as e:
            logger.warning(f"Could not retrieve knowledge contents: {e}")
        
        return {
            "status": "success",
            "knowledge_base_name": knowledge.name,
            "description": knowledge.description,
            "total_contents": len(contents),
            "contents": contents
        }
        
    except Exception as e:
        logger.error(f"Error getting knowledge status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting knowledge status: {str(e)}")

@app.post("/knowledge/search")
async def search_knowledge_direct(query: str, limit: int = 5):
    """Direct knowledge search endpoint for testing"""
    try:
        logger.info(f"Searching knowledge base for: '{query}'")
        
        # Perform search
        results = await knowledge.search_async(query=query, limit=limit)
        
        formatted_results = []
        if results:
            for i, result in enumerate(results):
                formatted_results.append({
                    "rank": i + 1,
                    "content": str(getattr(result, 'content', result))[:500] + "..." if len(str(getattr(result, 'content', result))) > 500 else str(getattr(result, 'content', result)),
                    "metadata": getattr(result, 'metadata', {}),
                    "score": getattr(result, 'score', None)
                })
        
        return {
            "status": "success",
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching knowledge: {str(e)}")

# Test endpoint for agent interaction
@app.post("/test/agent")
async def test_agent_knowledge(question: str):
    """Test agent's knowledge base access"""
    try:
        # This would simulate asking the agent directly
        # You might need to implement this based on your agno version
        return {
            "status": "info",
            "message": "Use the main chat endpoint to interact with the agent",
            "suggestion": f"Ask your question '{question}' through the agent interface"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# If you need to run locally, use this:
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Render sets this automatically)
    port = int(os.getenv("PORT", 8000))
    
    # Only use reload in development
    use_reload = ENV == "development"
    
    uvicorn.run(
        "main:app",  # Use import string instead of app object when using reload
        host="0.0.0.0", 
        port=port,
        reload=use_reload
    )
