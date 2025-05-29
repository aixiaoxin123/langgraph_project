from langgraph.prebuilt import create_react_agent


# 先创建llm
import os
from langchain.chat_models import init_chat_model

API_KEY = "sk-123"

BASE_URL = "https://api.deepseek.com"

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL


llm = init_chat_model("openai:deepseek-chat")


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"




graph = create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt="You are a helpful assistant"
)