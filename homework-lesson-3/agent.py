from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
# from langgraph.prebuilt import create_react_agent

from config import SYSTEM_PROMPT, Settings
from tools import TOOLS


settings = Settings()

llm = ChatOpenAI(
    api_key=settings.api_key.get_secret_value(),
    model=settings.model_name,
    timeout=settings.request_timeout,
)

memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=TOOLS,
    prompt=SYSTEM_PROMPT,
    checkpointer=memory,
)


AGENT_CONFIG = {
    "configurable": {"thread_id": "research-agent-cli"},
    "recursion_limit": settings.max_iterations,
}
