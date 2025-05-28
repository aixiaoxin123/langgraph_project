参考文档教程：

https://langchain-ai.github.io/langgraph/tutorials/get-started/4-human-in-the-loop/


```python
# https://python.langchain.com/docs/integrations/tools/tavily_search/

# 获取密钥的地址： https://app.tavily.com/sign-in




import os
TAVILY_API_KEY="tvly-dev-123"
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

```


```python
from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=5)
tools = [tool]
tool.invoke("What's a 'node' in LangGraph?")

```




    {'query': "What's a 'node' in LangGraph?",
     'follow_up_questions': None,
     'answer': None,
     'images': [],
     'results': [{'title': 'LangGraph Basics: Understanding State, Schema, Nodes, and Edges',
       'url': 'https://medium.com/@vivekvjnk/langgraph-basics-understanding-state-schema-nodes-and-edges-77f2fd17cae5',
       'content': 'LangGraph Basics: Understanding State, Schema, Nodes, and Edges | by Story_Teller | Medium LangGraph Basics: Understanding State, Schema, Nodes, and Edges LangGraph Basics: Understanding State, Schema, Nodes, and Edges These predefined structures in the messaging app are synonymous with the schema of the state in LangGraph. Just as a messaging app ensures all interactions (messages) follow a consistent format, the schema in LangGraph ensures the state passed along edges is structured and interpretable. This static schema allows nodes to rely on a consistent state format, ensuring seamless communication along edges throughout the graph. In this article, we explored the foundational concepts of graph-based systems, drawing parallels to familiar messaging applications to illustrate how edges, nodes, and state transitions function seamlessly in dynamic workflows.',
       'score': 0.98525,
       'raw_content': None},
      {'title': 'LangGraph Tutorial for Beginners - Analytics Vidhya',
       'url': 'https://www.analyticsvidhya.com/blog/2025/05/langgraph-tutorial-for-beginners/',
       'content': 'Now that we have built the simplest graph in the above section, in this section,\xa0I will show you how to use LangGraph to build a support chatbot, starting with basic functionality and progressively adding features like web search, memory, and human-in-loop. A. LangGraph offers:– State management which keeps track of data as the agent performs tasks.– Multi-agent support which allows multiple agents to work together within a graph.– Persistence with checkpointers as it saves the state at each step which enable error recovery and debudding.– Human-in-the-loop which helps in pausing the workflow for human review and approval. It allows us to build applications that use the power of LLMs, such as chatbots and AI assistants, while managing complex workflows and state across multiple agents.',
       'score': 0.98138,
       'raw_content': None},
      {'title': 'LangGraph Tutorial: What Is LangGraph and How to Use It?',
       'url': 'https://www.datacamp.com/tutorial/langgraph-tutorial',
       'content': 'LangGraph is a library within the LangChain ecosystem that provides a framework for defining, coordinating, and executing multiple LLM agents (or chains) in a structured and efficient manner. By managing the flow of data and the sequence of operations, LangGraph allows developers to focus on the high-level logic of their applications rather than the intricacies of agent coordination. Whether you need a chatbot that can handle various types of user requests or a multi-agent system that performs complex tasks, LangGraph provides the tools to build exactly what you need. LangGraph significantly simplifies the development of complex LLM applications by providing a structured framework for managing state and coordinating agent interactions.',
       'score': 0.97647,
       'raw_content': None},
      {'title': 'LangGraph - LangChain',
       'url': 'https://www.langchain.com/langgraph',
       'content': "[LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/)[LangSmith](https://docs.smith.langchain.com/)[LangChain](https://python.langchain.com/docs/introduction/) ![](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/667b080e4b3ca12dc5d5d439_Langgraph%20UI-2.webp) [Read a conceptual guide](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#human-in-the-loop) ![](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/68152f21a3b30d65d6c71bb3_Customizable-Agent-Architectures_v3%20(1).gif) [See different agent architectures](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/) [Learn about agent memory](https://langchain-ai.github.io/langgraph/concepts/memory/) ![](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/67878de387cf10f90c7ad65f_LangGraph---Memory-HQ.gif) [![](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/678e35d6c553c4fb20f9b753_Frame%2099644.webp)![](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/6787ae0bce5c99dd808545ce_card%202.webp)](https://academy.langchain.com/courses/intro-to-langgraph) Design agent-driven user experiences with LangGraph Platform's APIs. Quickly deploy and scale your application with infrastructure built for agents. The next chapter in building complex production-ready features with LLMs is agentic, and with LangGraph and LangSmith, LangChain delivers an out-of-the-box solution to iterate quickly, debug immediately, and scale effortlessly.” The next chapter in building complex production-ready features with LLMs is agentic, and with LangGraph and LangSmith, LangChain delivers an out-of-the-box solution to iterate quickly, debug immediately, and scale effortlessly.” LangGraph Platform is a service for deploying and scaling LangGraph applications, with an opinionated API for building agent UXs, plus an integrated developer studio.",
       'score': 0.97456,
       'raw_content': None},
      {'title': 'Nodes and Edges | langchain-ai/langgraph-101 | DeepWiki',
       'url': 'https://deepwiki.com/langchain-ai/langgraph-101/2.2-nodes-and-edges',
       'content': 'Nodes and Edges | langchain-ai/langgraph-101 | DeepWiki Nodes and Edges Nodes and Edges What are Nodes and Edges? In LangGraph, a graph is composed of nodes connected by edges to form a directed workflow. Nodes are the workhorses of LangGraph - they are Python functions that receive the current graph state as input, perform operations, and return updates to that state. Edges define the flow of execution between nodes in a LangGraph. graph_builder.add_edge("retrieve_documents", "generate_response") Conditional edges use a function to determine the next node based on the current state. Building a Graph with Nodes and Edges graph_builder.add_node("retrieve_documents", retrieve_documents) graph_builder.add_edge("retrieve_documents", "generate_response") When designing nodes and edges in LangGraph: Nodes and Edges What are Nodes and Edges? Building a Graph with Nodes and Edges',
       'score': 0.96957,
       'raw_content': None}],
     'response_time': 0.96}




```python
import os
from langchain.chat_models import init_chat_model

API_KEY = "sk-123"

BASE_URL = "https://api.deepseek.com"

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL


llm = init_chat_model("openai:deepseek-chat")
```


```python
#我们现在可以将它合并到 ：StateGraph

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Modification: tell the LLM which tools it can call
# highlight-next-line
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
```




    <langgraph.graph.state.StateGraph at 0x193a56a3560>




```python
from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.types import Command, interrupt

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]



tool = TavilySearch(max_results=2)

tools = [tool,human_assistance] # 添加human_assistance工具

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
```


```python
# 添加内存功能
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```


```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```


    
![png](4%E6%B7%BB%E5%8A%A0human-in-the-loop_files/4%E6%B7%BB%E5%8A%A0human-in-the-loop_7_0.png)
    



```python
#与您的聊天机器人互动


config = {"configurable": {"thread_id": "2"}}


user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
    
```

    ================================[1m Human Message [0m=================================
    
    I need some expert guidance for building an AI agent. Could you request assistance for me?
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      human_assistance (call_0_b8540b0e-65d5-4ea6-99d4-33e14342b9fc)
     Call ID: call_0_b8540b0e-65d5-4ea6-99d4-33e14342b9fc
      Args:
        query: I need expert guidance for building an AI agent.
    


```python
snapshot = graph.get_state(config)
snapshot.next
```




    ('tools',)




```python
# 恢复执行

human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)

human_command = Command(resume={"data": human_response})

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      human_assistance (call_0_b8540b0e-65d5-4ea6-99d4-33e14342b9fc)
     Call ID: call_0_b8540b0e-65d5-4ea6-99d4-33e14342b9fc
      Args:
        query: I need expert guidance for building an AI agent.
    =================================[1m Tool Message [0m=================================
    Name: human_assistance
    
    We, the experts are here to help! We'd recommend you check out LangGraph to build your agent. It's much more reliable and extensible than simple autonomous agents.
    ==================================[1m Ai Message [0m==================================
    
    It looks like the experts have provided a helpful recommendation! They suggest using **LangGraph** to build your AI agent, as it is more reliable and extensible compared to simple autonomous agents. 
    
    If you'd like further details or assistance with LangGraph or any other aspect of building your AI agent, feel free to ask!
    
