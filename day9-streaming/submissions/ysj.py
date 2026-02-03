"""
.stream() : 동기
.astream() : 비동기

| 모드        | 의미                                         | 출력 예시
| ---------- | ------------------------------------------- | ------------------------------- |
| `updates`  | 각 스텝(step)에서의 상태 변경만 스트리밍함      | {'nodeA': {'field': 'value'}}
| `values`   | 전체 상태를 계속 추적해야 할 때                 | 전체 상태 객체 {...}
| `messages` | 실시간 채팅처럼 토큰 단위 출력이 필요할 때         | (토큰조각, metadata)
| `custom`   | 노드 내부에서 writer()로 직접 만든 임의 스트림 출력   | {"progress":"50%"}
| `debug`    | 디버깅용 상세 정보 (노드 id, 시간, 입력/출력 등)     | 상세 실행 로그

"""
from dotenv import load_dotenv
load_dotenv()

# =========================================================
# 1. 기본 스트리밍 (updates / values / custom / messages / debug)
# =========================================================

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer  

class State(TypedDict):
    topic: str
    joke: str

def refine_topic(state: State):
    return {"topic": state["topic"] + " 그리고 고양이"}

def generate_joke(state: State):
    msg = llm.invoke(f"{state['topic']}에 대한 농담 하나 만들어줘")
    # for custom stream
    writer = get_stream_writer()  
    writer(f"[1/2] {state['topic']} 조회 시작")
    writer(f"[2/2] {state['topic']} 조회 완료")
    return {"joke": f"이것은 {state['topic']}에 대한 농담입니다: \n{msg.content}"}

graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile()
)

# Basic usage (기본은 updates 모드)
print("\n#######Basic usage#######")
for chunk in graph.stream(  
    {"topic": "ice cream"}
):
    print(chunk)

# Multiple modes
print("\n#######Multiple modes#######")
for mode, chunk in graph.stream({"topic": "ice cream"}, stream_mode=["updates", "custom", "values", "messages", "debug"]):
    print(f"\n{mode}: {chunk}")


# =========================================================
# 2. messages 스트림 + metadata 필터링 (누가 / 어디서 말했는지)
# =========================================================
print(f"\n\n# -------------------------\n# metadata를 이용한 필터링\n# -------------------------")

import asyncio
from typing import TypedDict
from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model

# 1. LLM 호출마다 tags를 달아 "누가 말했는지" 구분
joke_model = init_chat_model(model="gemini-flash-latest", tags=["joke"])
poem_model = init_chat_model(model="gemini-flash-latest", tags=["poem"])

class State(TypedDict):
    topic: str
    joke: str
    poem: str

# 2. 노드 2개 (node 기준 필터용)
async def write_joke(state: State, config):
    res = await joke_model.ainvoke(
        [{"role": "user", "content": f"{state['topic']}에 대한 농담 하나 만들어줘"}],
        config,
    )
    return {"joke": res.content}

async def write_poem(state: State, config):
    res = await poem_model.ainvoke(
        [{"role": "user", "content": f"{state['topic']}에 대한 짧은 시 하나 만들어줘"}],
        config,
    )
    return {"poem": res.content}

# 3. 그래프 (병렬 실행)
graph = (
    StateGraph(State)
    .add_node("write_joke", write_joke)
    .add_node("write_poem", write_poem)
    .add_edge(START, "write_joke")
    .add_edge(START, "write_poem")
    .compile()
)


print("\n######## messages 스트림 + metadata 필터링 ########")
# 4. 스트리밍 출력
async def main():
    async for msg, metadata in graph.astream(
        {"topic": "고양이"},
        stream_mode="messages",
    ):
        if not msg.content:
            continue

        # Filter by LLM invocation (tags) : 어떤 LLM 호출에서
        if metadata.get("tags") == ["joke"]:
            print(msg.content, end="", flush=True)

        # # Filter by node : 그래프의 어느 노드에서
        # if metadata.get("langgraph_node") == "write_poem":
        #     print(msg.content, end="|", flush=True)

asyncio.run(main())


# =========================================================
# 3. LangChain을 안 쓰는 LLM이라도, 토큰 스트리밍을 custom 스트림으로 LangGraph에 끼워 넣을 수 있음
# =========================================================
import operator
import json

from typing import TypedDict
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START

from openai import AsyncOpenAI

openai_client = AsyncOpenAI()
model_name = "gemini-flash-latest"


async def stream_tokens(model_name: str, messages: list[dict]):
    response = await openai_client.chat.completions.create(
        messages=messages, model=model_name, stream=True
    )
    role = None
    async for chunk in response: # 비동기 반복문으로 토큰 조각을 하나씩 받음
        delta = chunk.choices[0].delta

        if delta.role is not None:
            role = delta.role

        if delta.content: # 토큰 내용이 있다면 yield로 밖으로 보냄
            yield {"role": role, "content": delta.content}


# this is our tool
async def get_items(place: str) -> str:
    """Use this tool to list items one might find in a place you're asked about."""
    writer = get_stream_writer()
    response = ""
    async for msg_chunk in stream_tokens(
        model_name,
        [{"role": "user", "content": f"{place}에서 볼 수 있는 물건 3가지를 설명과 함께 알려줘"}],
    ):
        response += msg_chunk["content"]
        writer(msg_chunk)

    return response


class State(TypedDict):
    messages: Annotated[list[dict], operator.add]


# this is the tool-calling graph node
async def call_tool(state: State):
    ai_message = state["messages"][-1]
    tool_call = ai_message["tool_calls"][-1]

    function_name = tool_call["function"]["name"]
    if function_name != "get_items":
        raise ValueError(f"Tool {function_name} not supported")

    function_arguments = tool_call["function"]["arguments"]
    arguments = json.loads(function_arguments)

    function_response = await get_items(**arguments)
    tool_message = {
        "tool_call_id": tool_call["id"],
        "role": "tool",
        "name": function_name,
        "content": function_response,
    }
    return {"messages": [tool_message]}


graph = (
    StateGraph(State)
    .add_node(call_tool)
    .add_edge(START, "call_tool")
    .compile()
)

print("\n\n######## custom 스트림 (LangChain 없이 토큰 스트리밍) ########")
inputs = {
    "messages": [
        {
            "content": None,
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "arguments": '{"place":"bedroom"}',
                        "name": "get_items",
                    },
                    "type": "function",
                }
            ],
        }
    ]
}

import asyncio
async def main():
    async for chunk in graph.astream(
        inputs,
        stream_mode="custom",
    ):
        print(chunk["content"], end="|", flush=True)

asyncio.run(main())

# =========================================================
# 4. Subgraph 스트리밍
# =========================================================
print("\n\n######## Subgraph 스트리밍 ########")
from langgraph.graph import START, StateGraph
from typing import TypedDict

# Define subgraph
class SubgraphState(TypedDict):
    foo: str  # note that this key is shared with the parent graph state
    bar: str

def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

# Define parent graph
class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

for chunk in graph.stream(
    {"foo": "foo"},
    stream_mode="updates",
    # Set subgraphs=True to stream outputs from subgraphs
    subgraphs=True,  
):
    print(chunk)