import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, StateGraph
from langgraph.constants import START, END
from langchain_openai import ChatOpenAI

# 先创建llm
import os
from langchain.chat_models import init_chat_model

API_KEY = "sk-123"

BASE_URL = "https://api.deepseek.com"

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL


llm = init_chat_model("openai:deepseek-chat")


model = llm

class ComplicatedState(MessagesState):
    new_field: str

async def make_graph():
    client = MultiServerMCPClient(
        {
       "bing-cn-mcp-server": {
            "url": "https://mcp.api-inference.modelscope.cn/sse/3a7bdd8299e141",
            "transport": "sse",
        }
        }
    )
    # since get_tools is not an async, it could be called directly
    tools = await client.get_tools()

    print(tools)
    sub_graph_workflow = StateGraph(ComplicatedState)
    sub_graph_workflow.add_node("executor", create_react_agent(model, tools))
    sub_graph_workflow.add_edge(START, "executor")
    sub_graph_workflow.add_edge("executor", END)
    sub_graph = sub_graph_workflow.compile()

    def call_sub_graph(state: MessagesState):
        # this is just a demonstration, the graph could be larger
        # there will be more invoke and more def instead of async def
        return sub_graph.invoke(ComplicatedState(messages=state["messages"], new_field="something"))

    main_graph_workflow = StateGraph(MessagesState)
    main_graph_workflow.add_node("sub_graph", call_sub_graph)
    main_graph_workflow.add_edge(START, "sub_graph")
    main_graph_workflow.add_edge("sub_graph", END)
    main_graph = main_graph_workflow.compile()

    return main_graph

async def main():
    graph = await make_graph() # top level async here
    # result = await graph.ainvoke({"messages": "What is the weather like in New York?"}) # it seems ainvoke is necessary for the later tool invocation
    # print(result)
    bing_response= graph.stream({"messages": "小米发布的芯片叫什么?"}, stream_mode="updates")
    for chunk in bing_response:
        print(chunk)

# Change the transport to 'sse' and start the mcp server first:
# https://github.com/modelcontextprotocol/quickstart-resources/tree/main/weather-server-python

if __name__ == "__main__":
    asyncio.run(main())
