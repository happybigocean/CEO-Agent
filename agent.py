import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.playground import Playground
from agno.storage.sqlite import SqliteStorage
from fastapi.middleware.cors import CORSMiddleware
from agno.db.postgres import PostgresDb

# Load OpenAI key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

storage = SqliteStorage(table_name="agent_sessions", db_file="agent.db")
memory = SqliteStorage(table_name="agent_memory", db_file="agent.db")

# Get your Supabase project and password
SUPABASE_PROJECT = os.getenv("SUPABASE_PROJECT")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")

SUPABASE_DB_URL = (
    f"postgresql://postgres:{SUPABASE_PASSWORD}@db.{SUPABASE_PROJECT}:5432/postgres"
)

supabase_db = PostgresDb(db_url=SUPABASE_DB_URL)

ceo_agent = Agent(
  name="CEO Agent",
  model=OpenAIChat(id="gpt-4.1"),
  storage=storage,
  description="CEO Agent",
  add_datetime_to_instructions=True,
  add_history_to_messages=True,
  memory=memory,
  num_history_runs=10,
  markdown=True,
  session_id="ceo_agent_session",
  user_id="ceo_user"
)

cto_agent = Agent(
  name="CTO Agent",
  model=OpenAIChat(id="gpt-4.1"),
  storage=storage,
  description="CTO Agent",
  add_datetime_to_instructions=True,
  add_history_to_messages=True,
  memory=memory,
  num_history_runs=10,
  markdown=True,
  session_id="cto_agent_session",
  user_id="cto_user"
)

playground = Playground(agents=[ceo_agent, cto_agent])
app = playground.get_app()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],   # you can restrict later if needed
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

if __name__ == "__main__":
  playground.serve("agent:app", reload=True)
