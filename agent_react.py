from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Define a simple tool
def get_weather(city: str) -> str:
    """Return a fake weather string for demo purposes."""
    return f"It's always sunny in {city} (demo)."

# Connect LangChain's OpenAI client to Ollama
llm = ChatOpenAI(
    model="llama3-groq-tool-use",   # or any model you’ve pulled into Ollama, e.g. "mistral", "llama2"
    temperature=0,
)

# Create a ReAct agent
agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt="You are a helpful assistant."
)

# Invoke the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Paris?"}]}
)

print(result)
