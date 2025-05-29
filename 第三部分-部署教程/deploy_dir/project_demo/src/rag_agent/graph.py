# 1 读取本地txt，作为rag的参考文献，本文主要读取倚天屠龙记小说；

from langchain_community.document_loaders import  TextLoader

txt_path=r"D:\github_workspace\agent_dir\aixiaoxin_project\langgraph_project\第三部分-部署教程\deploy_dir\project_demo\src\rag_agent\data\金庸-倚天屠龙记txt精校版.txt"
docs = [TextLoader(txt_path,encoding='gb18030').load()]
print(len(docs))
docs[0][0].page_content.strip()[:1000]


# 2. 将获取的文档拆分成更小的块，以便索引到我们的 vectorstore 中：

from langchain_text_splitters import RecursiveCharacterTextSplitter

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1024, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)
print(len(doc_splits))

# 取样100条数据，方便测试，实际使用时，可以全部使用
doc_splits=doc_splits[0:300]


# 2. 创建检索工具¶
"""
现在我们有了拆分的文档，我们可以将它们索引到用于语义搜索的向量存储中。

2.1. 使用内存中向量存储和 ollama的本地向量模型 嵌入：
"""


from langchain_ollama import OllamaEmbeddings

local_embeddings = OllamaEmbeddings(
    model="bge-m3:latest",
    base_url="127.0.0.1:11434",
)

print(local_embeddings)






# 2.2. 使用 LangChain 的预构建创建一个 retriever 工具：create_retriever_tool

# 使用内存向量库
from langchain_core.vectorstores import InMemoryVectorStore

from langchain.tools.retriever import create_retriever_tool


vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=local_embeddings
)
retriever = vectorstore.as_retriever()



retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_novel_info",
    "Search and return information about novel_info.",
)


# 3.1 创建llm


from langchain_community.chat_models import ChatOpenAI
import os

API_KEY = "sk-123"

BASE_URL = "https://api.deepseek.com"

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL

llm = ChatOpenAI(model="deepseek-chat",temperature=0)



### 3.2 Retrieval Grader
# 检索 评分者
"""您是一名评分员，评估检索文档与用户问题的相关性。\ n
下面是检索到的文档：  {document} \n\n
下面是用户的问题：{question} \n
如果文档包含与用户问题相关的关键字，则将其评分为相关。\ n
这并不需要严格的测试。目标是过滤掉错误的检索。\ n
给出一个二元分数“是”或“否”，以表明该文档是否与问题相关。\ n
将二进制得分提供为一个只有一个键` score `的JSON，并且没有预表或解释。
"""


from langchain.prompts import PromptTemplate
# from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser



prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()
# question = "张翠山是谁"
# k 表示top-k的数量，找到最相关的k个文档

# docs = retriever.get_relevant_documents(question,k=10)
# print(len(docs))
# doc_txt = docs[1].page_content
# print(doc_txt)
# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))


### 3.3 Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt = hub.pull("rlm/rag-prompt")
print("prompt:{}".format(prompt))

# LLM


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt | llm | StrOutputParser()

# # Run
# generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)

# 流式输出




### 3.4 Hallucination Grader 幻觉评分员

"""“你是一名评分者，评估一个答案是否基于一组事实。\ n
以下是事实：
\n ------- \n
{文档}
\n ------- \n
下面是答案：{generation}
给出一个“是”或“不是”的二元分数，以表明答案是否有一系列事实支持。\ n
将二进制得分提供为一个只有` score `键的JSON，没有序言或解释。"""



# Prompt
prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm | JsonOutputParser()
# hallucination_grader.invoke({"documents": docs, "generation": generation})


### 3.5 Answer Grader  答案评分员
"""你正在评估一个答案是否对解决一个问题有用。\ n
下面是答案：
\n ------- \n
{一代}
\n ------- \n
问题是这样的：{question}
给出一个二元分数“是”或“否”，以表明该答案对解决问题是否有用。\ n
将二进制得分提供为一个只有` score `键的JSON，没有序言或解释。"""


# Prompt
prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()
# answer_grader.invoke({"question": question, "generation": generation})


### 3.6 Question Re-writer  问题重写器

"""
你是一个问题重写器，它将输入的问题转换为优化过的更好的版本
用于vectorstore检索。看看最初的问题，然后提出一个改进的问题。\ n
这是最初的问题： {question}。改进的无序言问题：

"""
# LLM

# Prompt
re_write_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
    input_variables=["generation", "question"],
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
# print("question:{}".format(question))

# question_rewriter.invoke({"question": question})



## 4.1  Graph state  构建agent的图

from typing import List

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

### Nodes

# 检索文件
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.get_relevant_documents(question,k=5)
    return {"documents": documents, "question": question}

# 生成答案
def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

# 检索的文档评分
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    print("一共检索了{}个段落".format(len(documents)))
          
    # Score each doc
    filtered_docs = []
    curr_index=0
    for d in documents:
        curr_index+=1
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        print("---DOCUMENT SCORE:---")

        grade = score["score"]
        if grade == "yes":
            print("第{curr_index}个段落---GRADE: DOCUMENT RELEVANT---".format(curr_index=curr_index))
            filtered_docs.append(d)
        else:
            print("第{curr_index}个段落---GRADE: DOCUMENT NOT RELEVANT---".format(curr_index=curr_index))
            continue
        

    print("最后筛选后与问题相关的段落有{}个".format(len(filtered_docs)))

    return {"documents": filtered_docs, "question": question}


# 重写问题
def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    print("原始问题：{}".format(question))
    print("重写后的问题：{}".format(better_question))
    return {"documents": documents, "question": better_question}


### Edges

# 确定是生成答案还是重新生成问题。
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


# 确定生成是否基于文档并回答问题。
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    


## 4.2 创建流程图，增加节点和边

from langgraph.graph import END, StateGraph, START

graph = StateGraph(GraphState)

# Define the nodes
graph.add_node("retrieve", retrieve)  # retrieve
graph.add_node("grade_documents", grade_documents)  # grade documents
graph.add_node("generate", generate)  # generate
graph.add_node("transform_query", transform_query)  # transform_query

# Build graph
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
graph.add_edge("transform_query", "retrieve")
graph.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = graph.compile()


