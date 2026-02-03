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
import os
from dotenv import load_dotenv
load_dotenv()

# =========================================================
# 기본 설정 
# =========================================================
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI

# DB 설정
MongoDB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MongoDB_URI)
DB = client["brickers"]
collection = DB["ldraw_parts"]

# 모델 설정
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# =========================================================
# 1. 기본 스트리밍 (updates / values / custom / messages / debug)
# =========================================================

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer

class State(TypedDict):
    topic: str
    joke: str

def refine_topic(state: State):
    return {"topic": state["topic"] + " 그리고 고양이"}

def generate_joke(state: State):
    # custom stream
    writer = get_stream_writer()
    writer(f"[1/2] {state['topic']} 조회 시작")
    msg = llm.invoke(f"{state['topic']}에 대한 농담 하나 만들어줘")
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
joke_model = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai", tags=["joke"])
poem_model = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai", tags=["poem"])

class State2(TypedDict):
    topic: str
    joke: str
    poem: str

# 2. 노드 2개 (node 기준 필터용)
async def write_joke(state: State2, config):
    res = await joke_model.ainvoke(
        [{"role": "user", "content": f"{state['topic']}에 대한 농담 하나 만들어줘"}],
        config,
    )
    return {"joke": res.content}

async def write_poem(state: State2, config):
    res = await poem_model.ainvoke(
        [{"role": "user", "content": f"{state['topic']}에 대한 짧은 시 하나 만들어줘"}],
        config,
    )
    return {"poem": res.content}

# 3. 그래프 (병렬 실행)
graph2 = (
    StateGraph(State2)
    .add_node("write_joke", write_joke)
    .add_node("write_poem", write_poem)
    .add_edge(START, "write_joke")
    .add_edge(START, "write_poem")
    .compile()
)


print("\n######## messages 스트림 + metadata 필터링 ########")
# 4. 스트리밍 출력
async def main():
    async for msg, metadata in graph2.astream(
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
# 3. MongoDB 검색 + custom 스트림 (토큰 단위 스트리밍)
# =========================================================
print("\n\n######## MongoDB 검색 + custom 스트림 ########")

import operator
from typing import Annotated
from langgraph.graph import StateGraph, START, END

class SearchState(TypedDict):
    query: str
    results: str
    response: str

def search_parts(state: SearchState):
    """MongoDB에서 부품 검색"""
    writer = get_stream_writer()
    writer(f"[검색 시작] {state['query']}")

    results = list(collection.find(
        {
            "$or": [
                {"name": {"$regex": state["query"], "$options": "i"}},
                {"keywords": {"$regex": state["query"], "$options": "i"}},
                {"partId": {"$regex": state["query"], "$options": "i"}}
            ]
        },
        {"_id": 0, "name": 1, "partId": 1, "keywords": 1, "category": 1}
    ).limit(3))

    writer(f"[검색 완료] {len(results)}개 결과")
    return {"results": str(results) if results else "(부품 정보 없음)"}

def generate_response(state: SearchState):
    """검색 결과 기반 응답 생성"""
    writer = get_stream_writer()
    writer("[응답 생성 시작]")

    prompt = f"""
    사용자 질문: {state['query']}
    검색된 부품 정보: {state['results']}

    위 정보를 바탕으로 친절하게 답변해줘.
    """

    msg = llm.invoke(prompt)
    writer("[응답 생성 완료]")
    return {"response": msg.content}

search_graph = (
    StateGraph(SearchState)
    .add_node("search_parts", search_parts)
    .add_node("generate_response", generate_response)
    .add_edge(START, "search_parts")
    .add_edge("search_parts", "generate_response")
    .add_edge("generate_response", END)
    .compile()
)

# custom 스트림으로 진행 상황 확인 
for mode, chunk in search_graph.stream(
    {"query": "Castle"},  # keywords에 있는 값으로 검색
    stream_mode=["updates", "custom"]
):
    if mode == "custom":
        print(f"진행: {chunk}")
    else:
        print(f"결과: {chunk}")


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
graph3 = builder.compile()

for chunk in graph3.stream(
    {"foo": "foo"},
    stream_mode="updates",
    # Set subgraphs=True to stream outputs from subgraphs
    subgraphs=True,
):
    print(chunk)
