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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENV = os.getenv("ENV", "development")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

SUPABASE_DB_URL = (
    f"postgresql://postgres.qfelvikfvbbbxijcitgg:IldCjSTVqXfsZ4YZ@aws-1-us-east-2.pooler.supabase.com:6543/postgres"
)

# Initialize database connections
supabase_db = PostgresDb(
    db_url=SUPABASE_DB_URL,
    id="supabase-main",
    knowledge_table="knowledge_contents",
)

# Initialize vector database without SearchType
vector_db = PgVector(
    table_name="vectors", 
    db_url=SUPABASE_DB_URL,
    # Remove search_type parameter - use defaults
)

# Initialize knowledge base
knowledge = Knowledge(
    name="CEO Knowledge Base",
    description="Comprehensive knowledge base for CEO Agent",
    contents_db=supabase_db,
    vector_db=vector_db,
)

# Initialize Agent with corrected model and improved instructions
ceo_agent = Agent(
    name="CEO Agent",
    model=OpenAIChat(
        id="gpt-4o",  # Use valid model ID
        temperature=0.1
    ),
    instructions=[
        "You are a knowledgeable CEO assistant with access to a comprehensive knowledge base.",
        "When answering questions, always search through your knowledge base first.",
        "Use the search_knowledge function to find relevant information.",
        "If you find relevant information in the knowledge base, use it to provide detailed answers with proper citations.",
        "If you cannot find specific information in the knowledge base, clearly state this and offer to help in other ways.",
        "For questions about knowledge base contents, try searching with broad terms like 'content', 'topics', or specific keywords.",
        "Always be helpful and provide the most comprehensive answer possible based on available knowledge."
    ],
    description="Advanced CEO Agent with Knowledge Base Access",
    user_id="ceo_user",
    db=supabase_db,
    knowledge=knowledge,
    add_history_to_context=True,
    num_history_runs=20,
    search_knowledge=True,  # This is crucial for knowledge base access
    markdown=True,
    tools=[ReasoningTools()],
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
        
        # Clear existing knowledge first (optional)
        # await knowledge.clear()
        
        # Load the Thai recipes PDF
        result1 = await knowledge.add_content_async(
            name="Thai Recipes Collection",
            url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",
            metadata={"user_tag": "Thai Recipes", "content_type": "recipes", "source": "PDF"}
        )
        
        # Add CEO best practices content
        result2 = await knowledge.add_content_async(
            name="CEO Leadership Best Practices",
            content="""
# CEO Leadership Best Practices

## Strategic Planning and Vision
- Develop clear, measurable vision statements that inspire teams
- Create 3-5 year strategic roadmaps with quarterly milestones
- Conduct regular market analysis and competitive assessments
- Align organizational goals with market opportunities

## Team Leadership and Management
- Foster psychological safety and open communication
- Implement regular one-on-one meetings with direct reports
- Create clear career development paths for employees
- Recognize and reward high performance consistently

## Decision Making Framework
- Use data-driven approaches for major decisions
- Consider all stakeholder impacts before implementation
- Maintain transparency in decision-making processes
- Document lessons learned from both successes and failures

## Financial Management
- Monitor key financial metrics monthly
- Maintain healthy cash flow and reserves
- Invest in growth opportunities strategically
- Regular financial audits and compliance checks

## Innovation and Growth
- Encourage calculated risk-taking and experimentation
- Allocate budget for research and development
- Stay current with industry trends and technologies
- Build partnerships that enhance competitive advantage
            """,
            metadata={"user_tag": "CEO Guidelines", "content_type": "best_practices", "category": "leadership"}
        )
        
        # Add sample company policies
        result3 = await knowledge.add_content_async(
            name="Company Policies and Procedures",
            content="""
# Company Policies and Procedures

## Remote Work Policy
- Flexible working hours between 8 AM - 6 PM local time
- Mandatory team meetings twice weekly
- Home office equipment allowance provided
- Regular check-ins with managers required

## Performance Review Process
- Quarterly performance reviews for all employees
- 360-degree feedback system implementation
- Goal setting and tracking using OKR framework
- Career development discussions included

## Code of Conduct
- Respect and inclusivity in all interactions
- Confidentiality of company and client information
- Ethical business practices and compliance
- Reporting procedures for violations

## Training and Development
- Annual training budget per employee
- Skill development workshops monthly
- Leadership training programs available
- External conference and certification support
            """,
            metadata={"user_tag": "Company Policies", "content_type": "policies", "category": "operations"}
        )
        
        logger.info("Knowledge loading completed successfully")
        
        return {
            "status": "success", 
            "message": "Knowledge base loaded successfully",
            "loaded_items": [
                {"name": "Thai Recipes Collection", "result": str(result1)[:100]},
                {"name": "CEO Leadership Best Practices", "result": str(result2)[:100]},
                {"name": "Company Policies", "result": str(result3)[:100]}
            ]
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
