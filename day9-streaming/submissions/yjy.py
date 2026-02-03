"""
LangGraph Streaming Demo (Single File)

- stream()  : 동기
- astream() : 비동기

stream_mode:
- updates  : 상태 변경 diff
- values   : 전체 상태 스냅샷
- messages : 토큰 단위 출력 + metadata
- custom   : 개발자 정의 이벤트 스트림
- debug    : 내부 실행 정보
"""

from dotenv import load_dotenv
load_dotenv()

# =========================================================
# 공통 LLM 설정 (Gemini)
# =========================================================
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

# =========================================================
# 1. 기본 스트리밍 (updates / values / custom / messages / debug)
# =========================================================
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer


class EventState(TypedDict):
    event: str
    report: str


def enrich_event(state: EventState):
    return {"event": state["event"] + " (야간, 지하주차장)"}


def generate_report(state: EventState):
    writer = get_stream_writer()
    writer("[1/2] 이벤트 분석 시작")

    msg = llm.invoke(
        f"{state['event']} 상황을 관제 리포트 형식으로 한 문단 요약해줘"
    )

    writer("[2/2] 이벤트 분석 완료")
    return {"report": f"[이벤트 요약]\n{msg.content}"}


graph = (
    StateGraph(EventState)
    .add_node("enrich_event", enrich_event)
    .add_node("generate_report", generate_report)
    .add_edge(START, "enrich_event")
    .add_edge("enrich_event", "generate_report")
    .add_edge("generate_report", END)
    .compile()
)

print("\n######## Basic streaming (default: updates) ########")
for chunk in graph.stream({"event": "침입 의심 인원 감지"}):
    print(chunk)

print("\n######## Multiple stream modes ########")
for mode, chunk in graph.stream(
        {"event": "침입 의심 인원 감지"},
        stream_mode=["updates", "custom", "values", "messages", "debug"],
):
    print(f"\n[{mode}] {chunk}")

# =========================================================
# 2. messages 스트림 + metadata 필터링
# =========================================================
import asyncio

summary_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
).with_config({"tags": ["incident_summary"]})

guide_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
).with_config({"tags": ["response_guide"]})


class ParallelState(TypedDict):
    event: str
    summary: str
    guide: str


async def write_summary(state: ParallelState, config):
    res = await summary_model.ainvoke(
        [{"role": "user", "content": f"{state['event']} 상황을 요약해줘"}],
        config,
    )
    return {"summary": res.content}


async def write_guide(state: ParallelState, config):
    res = await guide_model.ainvoke(
        [{"role": "user", "content": f"{state['event']} 발생 시 대응 가이드를 작성해줘"}],
        config,
    )
    return {"guide": res.content}


parallel_graph = (
    StateGraph(ParallelState)
    .add_node("write_summary", write_summary)
    .add_node("write_guide", write_guide)
    .add_edge(START, "write_summary")
    .add_edge(START, "write_guide")
    .compile()
)


print("\n######## messages stream + metadata filtering ########")


async def run_messages_stream():
    async for msg, metadata in parallel_graph.astream(
            {"event": "지하주차장 쓰러진 사람 감지"},
            stream_mode="messages",
    ):
        if not msg.content:
            continue

        if metadata.get("tags") == ["incident_summary"]:
            print(msg.content, end="", flush=True)


asyncio.run(run_messages_stream())

# =========================================================
# 3. LangChain 없이 custom 스트림으로 토큰 전달
# [수정] OpenAI 의존성 제거 -> Gemini(LangChain) 활용으로 변경
# =========================================================
import operator
import json
from typing_extensions import Annotated

# OpenAI 관련 코드 제거
# from openai import AsyncOpenAI
# openai_client = AsyncOpenAI()
# MODEL_NAME = "gpt-4o-mini"

# Gemini 모델 재사용
custom_stream_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

async def stream_tokens(messages: list[dict]):
    # LangChain의 astream 메서드를 사용하여 토큰 스트리밍
    # messages 포맷 변환 (dict -> str or BaseMessage)
    prompt = messages[0]["content"] # 간단하게 content만 추출
    
    async for chunk in custom_stream_llm.astream(prompt):
        if chunk.content:
            yield {"role": "assistant", "content": chunk.content}


async def analyze_scene(place: str) -> str:
    writer = get_stream_writer()
    result = ""

    async for token in stream_tokens(
            [{"role": "user", "content": f"{place}에서 관찰 가능한 객체 3가지를 설명해줘"}]
    ):
        result += token["content"]
        writer(token)

    return result


class ToolState(TypedDict):
    messages: Annotated[list[dict], operator.add]


async def call_tool(state: ToolState):
    # Tool call 파싱 로직은 유지하되, 실제 호출은 위에서 정의한 analyze_scene 사용
    tool_call = state["messages"][-1]["tool_calls"][-1]
    args = json.loads(tool_call["function"]["arguments"])

    response = await analyze_scene(**args)

    return {
        "messages": [
            {
                "role": "tool",
                "name": "analyze_scene",
                "tool_call_id": tool_call["id"],
                "content": response,
            }
        ]
    }


tool_graph = (
    StateGraph(ToolState)
    .add_node("call_tool", call_tool)
    .add_edge(START, "call_tool")
    .compile()
)

print("\n\n######## custom stream (external token source) ########")

tool_input = {
    "messages": [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "analyze_scene",
                        "arguments": '{"place":"지하주차장"}',
                    },
                }
            ],
        }
    ]
}


async def run_custom_stream():
    async for chunk in tool_graph.astream(tool_input, stream_mode="custom"):
        # chunk는 writer(token)에서 보낸 token 딕셔너리
        print(chunk["content"], end="|", flush=True)


asyncio.run(run_custom_stream())

# =========================================================
# 4. Subgraph 스트리밍
# =========================================================
from langgraph.graph import StateGraph


class SubgraphState(TypedDict):
    event_id: str
    status: str


def detect_event(state: SubgraphState):
    return {"status": "[DETECTED]"}


def confirm_event(state: SubgraphState):
    return {"event_id": state["event_id"] + state["status"]}


subgraph = (
    StateGraph(SubgraphState)
    .add_node("detect_event", detect_event)
    .add_node("confirm_event", confirm_event)
    .add_edge(START, "detect_event")
    .add_edge("detect_event", "confirm_event")
    .compile()
)


class ParentState(TypedDict):
    event_id: str


def init_event(state: ParentState):
    return {"event_id": "EVT-001-" + state["event_id"]}


parent_graph = (
    StateGraph(ParentState)
    .add_node("init_event", init_event)
    .add_node("subgraph", subgraph)
    .add_edge(START, "init_event")
    .add_edge("init_event", "subgraph")
    .compile()
)

print("\n\n######## Subgraph streaming ########")
for chunk in parent_graph.stream(
        {"event_id": "PARKING"},
        stream_mode="updates",
        subgraphs=True,
):
    print(chunk)
