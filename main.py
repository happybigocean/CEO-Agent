import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.os import AgentOS
from fastapi.middleware.cors import CORSMiddleware
from agno.db.postgres import PostgresDb

# Load OpenAI key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OS_SECURITY_KEY = os.getenv("OS_SECURITY_KEY")
ENV = os.getenv("ENV", "development")  # default to development if not set

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")
if not OS_SECURITY_KEY:
    raise ValueError("OS_SECURITY_KEY not set in .env")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OS_SECURITY_KEY"] = OS_SECURITY_KEY
os.environ.pop("OS_SECURITY_KEY", None)

# Get your Supabase project and password
SUPABASE_PROJECT = os.getenv("SUPABASE_PROJECT")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")

SUPABASE_DB_URL = (
    f"postgresql://postgres:{SUPABASE_PASSWORD}@{SUPABASE_PROJECT}:5432/postgres"
)

supabase_db = PostgresDb(
    db_url=SUPABASE_DB_URL
)

# Initialize Agent
ceo_agent = Agent(
    name="CEO Agent",
    model=OpenAIChat(id="gpt-4.1"),
    instructions=(
        "Always perform a Google Search before answering to ensure information is recent. "
        "Search should prioritize results from the past 24 hours. Use trusted news sources, "
        "financial data providers, and official announcements. Summarize clearly, verify conflicting "
        "claims, flag uncertainty, and provide sources. When asked, include pros & cons, risk "
        "assessment, and recommended actions. Always act with integrity and transparency."
    ),
    markdown=True,
    db=supabase_db,
    session_id="ceo_agent_session",
    user_id="ceo_user",
    add_history_to_context=True,
    num_history_runs=10,
)

cto_agent = Agent(
    name="CTO Agent",
    model=OpenAIChat(id="gpt-4.1"),
    instructions=(
        "Always perform a Google Search before answering to ensure information is recent. "
        "Search should prioritize results from the past 24 hours. Use trusted news sources, "
        "financial data providers, and official announcements. Summarize clearly, verify conflicting "
        "claims, flag uncertainty, and provide sources. When asked, include pros & cons, risk "
        "assessment, and recommended actions. Always act with integrity and transparency."
    ),
    markdown=True,
    db=supabase_db,
    session_id="cto_agent_session",
    user_id="cto_user",
    add_history_to_context=True,
    num_history_runs=10,
)

agent_os = AgentOS(
    os_id="netcorobo",
    description="NetcoRobo",
    agents=[ceo_agent, cto_agent]
)

app = agent_os.get_app()

if ENV == "production":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # production domain only
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )






