import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.playground import Playground
from agno.storage.sqlite import SqliteStorage

# Load OpenAI key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

# Store agent sessions in a SQLite database
storage = SqliteStorage(table_name="agent_sessions", db_file="agent.db")

# Initialize Agent
agent = Agent(
  name="CEO Agent",
  model=OpenAIChat(id="gpt-4.1", api_key=OPENAI_API_KEY),
  storage=storage,
  description="CEO Agent with SQLite long-term memory.",
  add_datetime_to_instructions=True,
  # Add the chat history to the messages
  add_history_to_messages=True,
  # Number of history runs
  num_history_runs=10,
  markdown=True,
)

playground = Playground(agents=[agent])
app = playground.get_app()

if __name__ == "__main__":
  playground.serve("agent:app", reload=True)
