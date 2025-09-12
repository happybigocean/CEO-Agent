import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.os import AgentOS
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer

# Load OpenAI key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OS_SECURITY_KEY = os.getenv("OS_SECURITY_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")
if not OS_SECURITY_KEY:
    raise ValueError("OS_SECURITY_KEY not set in .env")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OS_SECURITY_KEY"] = OS_SECURITY_KEY
os.environ.pop("OS_SECURITY_KEY", None)

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
)

agent_os = AgentOS(
    os_id="netcorobo",
    description="NetcoRobo",
    agents=[ceo_agent, cto_agent],
)

app = agent_os.get_app()



