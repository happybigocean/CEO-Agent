import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.os import AgentOS
from fastapi.middleware.cors import CORSMiddleware
from agno.db.postgres import PostgresDb
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector, SearchType
from fastapi import FastAPI, HTTPException
from agno.tools.reasoning import ReasoningTools

# Load OpenAI key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENV = os.getenv("ENV", "development") 

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

SUPABASE_DB_URL = (
   f"postgresql://postgres.qfelvikfvbbbxijcitgg:IldCjSTVqXfsZ4YZ@aws-1-us-east-2.pooler.supabase.com:6543/postgres"
)

supabase_db = PostgresDb(
    db_url=SUPABASE_DB_URL,
    id="supabase-main"
)

vector_db = PgVector(table_name="vectors", db_url=SUPABASE_DB_URL, search_type=SearchType.hybrid)

knowledge = Knowledge(
    vector_db=vector_db,
    contents_db=supabase_db
)

# Initialize Agent
ceo_agent = Agent(
    name="CEO Agent",
    model=OpenAIChat(id="gpt-4.1"),
    instructions=[
        "Include sources in your response.",
        "Always search your knowledge before answering the question.",
        "Only include the output in your response. No other text.",
    ],
    description="You are a Assistant based on the knowledge of the database.",
    markdown=True,
    db=supabase_db,
    user_id="ceo_user",
    add_history_to_context=True,
    num_history_runs=10,
    knowledge=knowledge,
    search_knowledge=True,
    tools=[ReasoningTools(add_instructions=True)],
)

agent_os = AgentOS(
    os_id="netcorobo",
    description="NetcoRobo",
    agents=[ceo_agent],
)

app = agent_os.get_app()

@app.post("/loadknowledge")
async def load_knowledge():
    try:
        await knowledge.add_content_async(
            url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"
        )
        return {"status": "success", "message": "Knowledge base loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading knowledge: {str(e)}")

if ENV == "production":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # production domain only
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )






