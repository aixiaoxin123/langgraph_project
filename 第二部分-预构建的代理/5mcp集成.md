参考文档：
https://langchain-ai.github.io/langgraph/agents/mcp/



自定义 MCP 服务器¶
要创建自己的 MCP 服务器，您可以使用该库。此库提供了一种定义工具并将其作为服务器运行的简单方法。mcp

安装 MCP 库：


pip install mcp
或者
uv pip install mcp -i https://pypi.tuna.tsinghua.edu.cn/simple/


安装库以在 LangGraph 中使用 MCP 工具：langchain-mcp-adapters


pip install langchain-mcp-adapters
或者
uv pip install langchain-mcp-adapters   -i https://pypi.tuna.tsinghua.edu.cn/simple/


# 此示例有问题的，和别的mcp服务端的工具，无法直接使用，可以获取mcp的工具集合，但是大模型调用mcp工具时，有问题，官方给的代码无法复现；




```python


# 先创建llm
import os
from langchain.chat_models import init_chat_model

API_KEY = "sk-123"

BASE_URL = "https://api.deepseek.com"

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL


llm = init_chat_model("openai:deepseek-chat")


# 获取mcp
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_servers={
       "howtocook-mcp": {
        "url": "https://mcp.api-inference.modelscope.cn/sse/0834c80993e34a",
        "transport": "sse"
         }
}
default_tools=[]
enabled_tools = {}

client=MultiServerMCPClient(mcp_servers)

loaded_tools = default_tools[:]
for tool in await  client.get_tools():
    print(tool.name)
    # if tool.name in enabled_tools:
    tool.description = (
        f"Powered by '{tool.name}'.\n{tool.description}"
    )
    loaded_tools.append(tool)
                    
print(loaded_tools)



# mcp_tools = await client.get_tools()

print(loaded_tools)

# 构建一个agent

from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# tool = TavilySearch(max_results=2)


tools = loaded_tools
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)

graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass





```

    mcp_howtocook_getAllRecipes
    mcp_howtocook_getRecipesByCategory
    mcp_howtocook_recommendMeals
    mcp_howtocook_whatToEat
    [StructuredTool(name='mcp_howtocook_getAllRecipes', description="Powered by 'mcp_howtocook_getAllRecipes'.\n获取所有菜谱", args_schema={'type': 'object', 'properties': {'no_param': {'type': 'string', 'description': '无参数'}}, 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}, response_format='content_and_artifact', coroutine=<function convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x000002D3EC0BC5E0>), StructuredTool(name='mcp_howtocook_getRecipesByCategory', description="Powered by 'mcp_howtocook_getRecipesByCategory'.\n根据分类查询菜谱，可选分类有: 水产, 早餐, 调味料, 甜品, 饮品, 荤菜, 半成品, 汤羹, 主食, 素菜", args_schema={'type': 'object', 'properties': {'category': {'type': 'string', 'enum': ['水产', '早餐', '调味料', '甜品', '饮品', '荤菜', '半成品', '汤羹', '主食', '素菜'], 'description': '菜谱分类名称，如水产、早餐、荤菜、主食等'}}, 'required': ['category'], 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}, response_format='content_and_artifact', coroutine=<function convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x000002D3EC0BD300>), StructuredTool(name='mcp_howtocook_recommendMeals', description="Powered by 'mcp_howtocook_recommendMeals'.\n根据用户的忌口、过敏原、人数智能推荐菜谱，创建一周的膳食计划以及大致的购物清单", args_schema={'type': 'object', 'properties': {'allergies': {'type': 'array', 'items': {'type': 'string'}, 'description': '过敏原列表，如["大蒜", "虾"]'}, 'avoidItems': {'type': 'array', 'items': {'type': 'string'}, 'description': '忌口食材列表，如["葱", "姜"]'}, 'peopleCount': {'type': 'integer', 'minimum': 1, 'maximum': 10, 'description': '用餐人数，1-10之间的整数'}}, 'required': ['peopleCount'], 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}, response_format='content_and_artifact', coroutine=<function convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x000002D3EC0BDD00>), StructuredTool(name='mcp_howtocook_whatToEat', description="Powered by 'mcp_howtocook_whatToEat'.\n不知道吃什么？根据人数直接推荐适合的菜品组合", args_schema={'type': 'object', 'properties': {'peopleCount': {'type': 'integer', 'minimum': 1, 'maximum': 10, 'description': '用餐人数，1-10之间的整数，会根据人数推荐合适数量的菜品'}}, 'required': ['peopleCount'], 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}, response_format='content_and_artifact', coroutine=<function convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x000002D3EC0BEA20>)]
    [StructuredTool(name='mcp_howtocook_getAllRecipes', description="Powered by 'mcp_howtocook_getAllRecipes'.\n获取所有菜谱", args_schema={'type': 'object', 'properties': {'no_param': {'type': 'string', 'description': '无参数'}}, 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}, response_format='content_and_artifact', coroutine=<function convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x000002D3EC0BC5E0>), StructuredTool(name='mcp_howtocook_getRecipesByCategory', description="Powered by 'mcp_howtocook_getRecipesByCategory'.\n根据分类查询菜谱，可选分类有: 水产, 早餐, 调味料, 甜品, 饮品, 荤菜, 半成品, 汤羹, 主食, 素菜", args_schema={'type': 'object', 'properties': {'category': {'type': 'string', 'enum': ['水产', '早餐', '调味料', '甜品', '饮品', '荤菜', '半成品', '汤羹', '主食', '素菜'], 'description': '菜谱分类名称，如水产、早餐、荤菜、主食等'}}, 'required': ['category'], 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}, response_format='content_and_artifact', coroutine=<function convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x000002D3EC0BD300>), StructuredTool(name='mcp_howtocook_recommendMeals', description="Powered by 'mcp_howtocook_recommendMeals'.\n根据用户的忌口、过敏原、人数智能推荐菜谱，创建一周的膳食计划以及大致的购物清单", args_schema={'type': 'object', 'properties': {'allergies': {'type': 'array', 'items': {'type': 'string'}, 'description': '过敏原列表，如["大蒜", "虾"]'}, 'avoidItems': {'type': 'array', 'items': {'type': 'string'}, 'description': '忌口食材列表，如["葱", "姜"]'}, 'peopleCount': {'type': 'integer', 'minimum': 1, 'maximum': 10, 'description': '用餐人数，1-10之间的整数'}}, 'required': ['peopleCount'], 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}, response_format='content_and_artifact', coroutine=<function convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x000002D3EC0BDD00>), StructuredTool(name='mcp_howtocook_whatToEat', description="Powered by 'mcp_howtocook_whatToEat'.\n不知道吃什么？根据人数直接推荐适合的菜品组合", args_schema={'type': 'object', 'properties': {'peopleCount': {'type': 'integer', 'minimum': 1, 'maximum': 10, 'description': '用餐人数，1-10之间的整数，会根据人数推荐合适数量的菜品'}}, 'required': ['peopleCount'], 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}, response_format='content_and_artifact', coroutine=<function convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x000002D3EC0BEA20>)]
    


    
![png](5mcp%E9%9B%86%E6%88%90_files/5mcp%E9%9B%86%E6%88%90_1_1.png)
    



```python
#与您的聊天机器人互动


config = {"configurable": {"thread_id": "1"}}


user_input = "一个人，晚上吃什么呢，推荐几个菜单给我."

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
    
```

    ================================[1m Human Message [0m=================================
    
    一个人，晚上吃什么呢，推荐几个菜单给我.
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      mcp_howtocook_whatToEat (call_0_2441726e-79ee-4dd7-8650-cfe92d3425ef)
     Call ID: call_0_2441726e-79ee-4dd7-8650-cfe92d3425ef
      Args:
        peopleCount: 1
    =================================[1m Tool Message [0m=================================
    Name: mcp_howtocook_whatToEat
    
    Error: NotImplementedError('StructuredTool does not support sync invocation.')
     Please fix your mistakes.
    ==================================[1m Ai Message [0m==================================
    
    我似乎遇到了一些技术问题，无法直接为您推荐菜单。不过，我可以手动为您推荐一些适合一个人晚餐的菜品：
    
    1. **番茄炒蛋** - 简单易做，营养丰富。
    2. **青椒土豆丝** - 清爽可口，适合搭配米饭。
    3. **蒜蓉西兰花** - 健康低脂，富含维生素。
    4. **煎鸡胸肉** - 高蛋白，适合健身人士。
    5. **紫菜蛋花汤** - 快速又暖胃。
    
    如果您有其他需求或偏好，可以告诉我，我会进一步调整推荐！
    


```python
# 先创建llm
import os
from langchain.chat_models import init_chat_model

API_KEY = "sk-123"

BASE_URL = "https://api.deepseek.com"

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL


llm = init_chat_model("openai:deepseek-chat")



from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

client = MultiServerMCPClient(
    {
        "bing-cn-mcp-server": {
            # make sure you start your weather server on port 8000
            "url": "https://mcp.api-inference.modelscope.cn/sse/3a7bdd8299e141",
            "transport": "sse",
        }
    }
)
tools = await client.get_tools()
print(tools)



agent = create_react_agent(llm, tools)

# bing_response = await agent.ainvoke({"messages": "小米发布的芯片叫什么?"})

bing_response=agent.stream({"messages": "小米发布的芯片叫什么?"}, stream_mode="updates")
for chunk in bing_response:
    print(chunk)

```

    [StructuredTool(name='bing_search', description='使用必应搜索指定的关键词，并返回搜索结果列表，包括标题、链接、摘要和ID', args_schema={'type': 'object', 'properties': {'query': {'type': 'string', 'description': '搜索关键词'}, 'num_results': {'type': 'number', 'default': 5, 'description': '返回的结果数量，默认为5'}}, 'required': ['query'], 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}, response_format='content_and_artifact', coroutine=<function convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x000002D3EE827880>), StructuredTool(name='fetch_webpage', description='根据提供的ID获取对应网页的内容', args_schema={'type': 'object', 'properties': {'result_id': {'type': 'string', 'description': '从bing_search返回的结果ID'}}, 'required': ['result_id'], 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}, response_format='content_and_artifact', coroutine=<function convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x000002D3EE826700>)]
    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_0_ac63dab9-2991-4e1b-b62b-6dae8a6fd2eb', 'function': {'arguments': '{"query":"小米 发布 芯片","num_results":5}', 'name': 'bing_search'}, 'type': 'function', 'index': 0}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 29, 'prompt_tokens': 311, 'total_tokens': 340, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 55}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_8802369eaa_prod0425fp8', 'id': '704db5df-dbdd-4067-9d07-c13de02741ab', 'service_tier': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--e4e110e6-bfb5-4cee-b0a6-d3df403214d9-0', tool_calls=[{'name': 'bing_search', 'args': {'query': '小米 发布 芯片', 'num_results': 5}, 'id': 'call_0_ac63dab9-2991-4e1b-b62b-6dae8a6fd2eb', 'type': 'tool_call'}], usage_metadata={'input_tokens': 311, 'output_tokens': 29, 'total_tokens': 340, 'input_token_details': {'cache_read': 256}, 'output_token_details': {}})]}}
    {'tools': {'messages': [ToolMessage(content="Error: NotImplementedError('StructuredTool does not support sync invocation.')\n Please fix your mistakes.", name='bing_search', id='92bebc95-8bb0-4247-8b5c-d3c4a569ddfa', tool_call_id='call_0_ac63dab9-2991-4e1b-b62b-6dae8a6fd2eb', status='error')]}}
    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_0_4d96acab-b12f-4abc-a9ff-493379933ae2', 'function': {'arguments': '{"query":"小米 发布 芯片"}', 'name': 'bing_search'}, 'type': 'function', 'index': 0}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 385, 'total_tokens': 407, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 320}, 'prompt_cache_hit_tokens': 320, 'prompt_cache_miss_tokens': 65}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_8802369eaa_prod0425fp8', 'id': '0e79e673-26df-4cde-9aea-d116b964254d', 'service_tier': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--4921cf02-1c05-4d87-a2d8-bf0406585dec-0', tool_calls=[{'name': 'bing_search', 'args': {'query': '小米 发布 芯片'}, 'id': 'call_0_4d96acab-b12f-4abc-a9ff-493379933ae2', 'type': 'tool_call'}], usage_metadata={'input_tokens': 385, 'output_tokens': 22, 'total_tokens': 407, 'input_token_details': {'cache_read': 320}, 'output_token_details': {}})]}}
    {'tools': {'messages': [ToolMessage(content="Error: NotImplementedError('StructuredTool does not support sync invocation.')\n Please fix your mistakes.", name='bing_search', id='d8a47124-9382-4232-9265-942abc4f3e99', tool_call_id='call_0_4d96acab-b12f-4abc-a9ff-493379933ae2', status='error')]}}
    {'agent': {'messages': [AIMessage(content='我无法直接搜索相关信息。不过，小米发布的芯片通常指的是其自研的“澎湃芯片”（Surge Chip）。例如：\n\n1. **澎湃S1**：小米在2017年发布的手机处理器芯片。\n2. **澎湃P1**：2021年发布的快充芯片，用于提升充电效率。\n3. **澎湃C1**：2021年发布的影像处理芯片，用于提升拍照效果。\n\n如果你需要更详细的信息，可以告诉我，我会尽力帮助你！', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 100, 'prompt_tokens': 452, 'total_tokens': 552, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 384}, 'prompt_cache_hit_tokens': 384, 'prompt_cache_miss_tokens': 68}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_8802369eaa_prod0425fp8', 'id': '4b3d5283-d447-4e7b-98a1-45e342a53938', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--e79a5148-7620-4b2d-b705-8d9fa67718f6-0', usage_metadata={'input_tokens': 452, 'output_tokens': 100, 'total_tokens': 552, 'input_token_details': {'cache_read': 384}, 'output_token_details': {}})]}}
    


```python

# New usage: 0.1.0以上的版本

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


client = MultiServerMCPClient(
    {
        "bing-cn-mcp-server": {
            "url": "https://mcp.api-inference.modelscope.cn/sse/3a7bdd8299e141",
            "transport": "sse",
        }
    }
)

tools = await client.get_tools()
print(tools)

```

    [StructuredTool(name='bing_search', description='使用必应搜索指定的关键词，并返回搜索结果列表，包括标题、链接、摘要和ID', args_schema={'type': 'object', 'properties': {'query': {'type': 'string', 'description': '搜索关键词'}, 'num_results': {'type': 'number', 'default': 5, 'description': '返回的结果数量，默认为5'}}, 'required': ['query'], 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}, response_format='content_and_artifact', coroutine=<function convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x000002D3EE83A840>), StructuredTool(name='fetch_webpage', description='根据提供的ID获取对应网页的内容', args_schema={'type': 'object', 'properties': {'result_id': {'type': 'string', 'description': '从bing_search返回的结果ID'}}, 'required': ['result_id'], 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}, response_format='content_and_artifact', coroutine=<function convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x000002D3EE83BE20>)]
    
