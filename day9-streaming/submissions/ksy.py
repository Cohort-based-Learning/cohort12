from dotenv import load_dotenv
load_dotenv()

# =========================================================
# 1. 기본 스트리밍 (updates / values / custom / messages / debug)
# =========================================================

from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer

# 1. 상태 정의
class ResearchState(TypedDict):
    query: str
    raw_data: str
    summary: str

# LLM 설정 (2026년 최신 모델 기준)
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

# 2. 노드 정의
def search_engine(state: ResearchState):
    writer = get_stream_writer()
    writer("🌐 웹 데이터 검색을 시작합니다...")
    
    # 실제 검색 대신 시뮬레이션
    search_result = f"'{state['query']}'에 대한 최신 트렌드는 AI 에이전트의 자율성 강화입니다."
    
    writer("✅ 검색 완료! 요약 단계로 넘어갑니다.")
    return {"raw_data": search_result}

def summarize_info(state: ResearchState):
    writer = get_stream_writer()
    writer("✍️ 요약문 생성 중...")
    
    response = llm.invoke(f"다음 데이터를 한 문장으로 요약해줘: {state['raw_data']}")
    
    return {"summary": response.content}

# 3. 그래프 구성
workflow = StateGraph(ResearchState)

workflow.add_node("search", search_engine)
workflow.add_node("summarize", summarize_info)

workflow.add_edge(START, "search")
workflow.add_edge("search", "summarize")
workflow.add_edge("summarize", END)

app = workflow.compile()

# 4. 멀티 모드 스트리밍 실행
print("=== 실시간 리서치 프로세스 시작 ===\n")

inputs = {"query": "2026년 AI 트렌드"}

# 주로 실무에서는 'custom'과 'updates'를 섞어서 많이 씁니다.
# stream_mode=["custom", "updates"]: 
#   1. "custom": 노드 내부에서 writer()를 통해 보낸 실시간 진행 상황이나 중간 데이터를 수신
#   2. "updates": 노드 실행이 완료된 후 반환(return)된 상태 업데이트 값을 수신
for mode, chunk in app.stream(inputs, stream_mode=["custom", "updates"]):
    if mode == "custom":
        # writer("문자열")로 보낸 내용이 chunk에 들어옵니다.
        print(f"[진행 상황] {chunk}")
    elif mode == "updates":
        # 노드가 return한 딕셔너리(상태 변경분)가 chunk에 들어옵니다.
        for node_name, output in chunk.items():
            print(f"[{node_name}] 단계 완료: {output}")

# =========================================================
# 2. messages 스트림 + metadata 필터링 (누가 / 어디서 말했는지)
# =========================================================

"""
[요약 설명]
위 코드는 '메타데이터(Tag)'를 활용해 멀티 LLM 스트리밍을 제어하는 패턴입니다.
1. 모델 초기화 시(init_chat_model) `tags=["insight"]` 등을 설정해둡니다.
2. `stream_mode="messages"`로 실행하면, 토큰과 함께 메타데이터가 넘어옵니다.
3. `if "insight" in metadata.get("tags", []):` 조건문으로 태그를 확인하여,
   병렬로 실행되는 여러 모델 중 원하는 모델의 답변만 골라서 출력할 수 있습니다.
"""

import asyncio
from typing import TypedDict
from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model

# 1. 서로 다른 역할을 가진 모델 설정
# 하나는 'technical' 태그를, 하나는 'insight' 태그를 가집니다.
analysis_model = init_chat_model(model="claude-haiku-4-5-20251001", model_provider="anthropic", tags=["technical"])
insight_model = init_chat_model(model="claude-haiku-4-5-20251001", model_provider="anthropic", tags=["insight"])

class ReportState(TypedDict):
    topic: str
    data_summary: str
    key_insights: str

# 2. 노드 정의
async def analyze_data(state: ReportState, config):
    # 데이터 수집 및 분석 시뮬레이션
    res = await analysis_model.ainvoke(
        [{"role": "user", "content": f"{state['topic']}에 대한 수치적 통계 정보를 요약해줘"}],
        config
    )
    return {"data_summary": res.content}

async def generate_insights(state: ReportState, config):
    # 핵심 인사이트 추출 시뮬레이션
    # insight_model은 초기화 시 tags=["insight"]가 설정되어 있습니다.
    # 이 태그 덕분에 스트리밍 시 메타데이터 필터링(metadata.get("tags"))을 통해
    # 특정 모델의 출력만 선별해서 사용자에게 보여줄 수 있습니다.
    res = await insight_model.ainvoke(
        [{"role": "user", "content": f"{state['topic']}의 향후 전망과 비즈니스 인사이트를 말해줘"}],
        config
    )
    return {"key_insights": res.content}

# 3. 그래프 구성 (병렬 처리)
builder = StateGraph(ReportState)
builder.add_node("analyze", analyze_data)
builder.add_node("insight", generate_insights)

builder.add_edge(START, "analyze")
builder.add_edge(START, "insight")

report_app = builder.compile()

# 4. 스트리밍 실행 (메타데이터 필터링)
async def run_report_stream():
    print(f"--- '전기차 시장' 리포트 생성 중 (인사이트 스트리밍 중) ---\n")
    
    inputs = {"topic": "2026년 전기차 시장 전망"}
    
    async for msg, metadata in report_app.astream(
        inputs, 
        stream_mode="messages"
    ):
        # 메시지 내용이 없거나 빈 문자열이면 스킵
        if not msg.content:
            continue
            
        # [필터링] 'insight' 태그가 달린 모델의 답변만 실시간 스트리밍
        if "insight" in metadata.get("tags", []):
            print(msg.content, end="", flush=True)
            
    print("\n\n--- 리포트 생성 완료 ---")

asyncio.run(run_report_stream())

# =========================================================
# 3. LangChain을 안 쓰는 LLM이라도, 토큰 스트리밍을 custom 스트림으로 LangGraph에 끼워 넣을 수 있음
# =========================================================

"""
[이 코드를 사용하는 이유]
1. 최신 기능 활용: LangChain에서 아직 지원하지 않는 각 LLM 모델의 최신 API 기능을 직접 제어하고 싶을 때 사용합니다.
2. 완전한 제어권: 토큰뿐만 아니라 중간 연산 과정, 로그, 커스텀 시각화 데이터 등 '내가 원하는 포맷'을 직접 정의해서 스트리밍 채널에 태워 보낼 수 있습니다.
3. 유연한 통합: LangChain의 추상화 계층 없이 순수 파이썬 로직으로 처리하면서도, LangGraph의 상태 관리(State)와 제어 흐름(Graph) 안에서 조화롭게 동작하도록 설계할 수 있습니다.
"""
import asyncio
import json
import operator
from typing import TypedDict, Annotated
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START
from langgraph.config import get_stream_writer
from anthropic import AsyncAnthropic

# 1. 초기 설정
client = AsyncAnthropic()
MODEL = "claude-haiku-4-5-20251001"

class State(TypedDict):
    # 메시지 이력을 누적하기 위한 설정
    messages: Annotated[list[dict], operator.add]

# 2. 로우 레벨 토큰 스트리머
async def raw_llm_stream(prompt: str):
    response = await client.messages.create(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL,
        max_tokens=1000,
        stream=True
    )
    async for chunk in response:
        if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
            yield chunk.delta.text

# 3. Custom 스트리밍을 사용하는 도구 (코드 리뷰 도구)
async def review_code_tool(code_snippet: str) -> str:
    """코드를 분석하고 개선점을 실시간으로 출력합니다."""
    writer = get_stream_writer()
    full_response = ""
    
    prompt = f"다음 코드의 버그나 개선점을 3가지만 지적해줘:\n{code_snippet}"
    
    # 도구 내부에서 토큰별로 스트리밍 발생
    async for token in raw_llm_stream(prompt):
        full_response += token
        # custom 스트림으로 토큰 전송
        writer({"type": "token", "content": token})
    
    return full_response

# 4. 그래프 노드 (도구 실행 로직)
async def tool_node(state: State):
    last_msg = state["messages"][-1]
    t_call = last_msg["tool_calls"][0]
    
    args = json.loads(t_call["function"]["arguments"])
    
    # 도구 실행 (내부에서 writer를 통해 스트리밍 쏨)
    result = await review_code_tool(args["code_snippet"])
    
    return {
        "messages": [{
            "tool_call_id": t_call["id"],
            "role": "tool",
            "name": "review_code_tool",
            "content": result
        }]
    }

# 5. 그래프 빌드
builder = StateGraph(State)
builder.add_node("call_tool", tool_node)
builder.add_edge(START, "call_tool")
review_app = builder.compile()

# 6. 실행 및 custom 스트림 수신
async def run_review():
    print("--- 실시간 코드 리뷰 스트리밍 시작 ---")
    
    initial_input = {
        "messages": [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "review_123",
                "function": {
                    "name": "review_code_tool",
                    "arguments": '{"code_snippet": "def add(a, b): return a+b"}'
                },
                "type": "function"
            }]
        }]
    }

    async for chunk in review_app.astream(initial_input, stream_mode="custom"):
        # chunk는 위에서 writer()에 넣은 데이터 포맷 그대로 나옵/니다.
        if chunk.get("type") == "token":
            print(chunk["content"], end="", flush=True)

    print("\n--- 리뷰 완료 ---")

if __name__ == "__main__":
    asyncio.run(run_review())

# =========================================================
# 4. Subgraph 스트리밍
# =========================================================
from langgraph.graph import START, END, StateGraph
from typing import TypedDict

# --- 1. 하위 그래프 (카피 제작 팀) ---
class CopywritingState(TypedDict):
    base_idea: str  # 부모로부터 받을 키
    draft: str
    is_approved: bool

def write_draft_node(state: CopywritingState):
    print("  [자식] 초안 작성 중...")
    return {"draft": f"멋진 광고 문구: {state['base_idea']}!"}

def review_copy_node(state: CopywritingState):
    print("  [자식] 검수 중...")
    return {"is_approved": True}

sub_builder = StateGraph(CopywritingState)
sub_builder.add_node("writer", write_draft_node)
sub_builder.add_node("reviewer", review_copy_node)
sub_builder.add_edge(START, "writer")
sub_builder.add_edge("writer", "reviewer")
sub_builder.add_edge("reviewer", END)
copy_subgraph = sub_builder.compile()


# --- 2. 부모 그래프 (마케팅 전략 팀) ---
class MarketingState(TypedDict):
    base_idea: str
    final_report: str

def planning_node(state: MarketingState):
    print("[부모] 전략 기획 중...")
    return {"base_idea": "친환경 텀블러"}

def final_step_node(state: MarketingState):
    print("[부모] 최종 보고서 정리 중...")
    return {"final_report": "캠페인 준비 완료"}

parent_builder = StateGraph(MarketingState)
parent_builder.add_node("planner", planning_node)
# 하위 그래프를 'creative_team'이라는 이름의 노드로 추가
parent_builder.add_node("creative_team", copy_subgraph)
parent_builder.add_node("reporter", final_step_node)

parent_builder.add_edge(START, "planner")
parent_builder.add_edge("planner", "creative_team")
parent_builder.add_edge("creative_team", "reporter")
parent_builder.add_edge("reporter", END)

marketing_app = parent_builder.compile()

# --- 3. 실행 및 서브그래프 스트리밍 관찰 ---
print("### 마케팅 프로세스 시작 ###\n")

# subgraphs=True 옵션으로 내부 진행상황을 투명하게 확인
for path, chunk in marketing_app.stream(
    {"base_idea": "기본 아이디어"}, 
    stream_mode="updates", 
    subgraphs=True
):
    # path는 현재 실행 중인 노드의 위치를 튜플로 보여줍니다. (예: ('creative_team', 'writer'))
    if not path:
        node_name = "Root"
    else:
        node_name = path[-1]
    print(f"경로: {path} | 노드: {node_name} | 업데이트: {chunk}")

"""
[서브그래프(Subgraph) 요약 정리]
1. 정의: 독립된 그래프를 다른 부모 그래프의 '하나의 노드'로 포함시킨 구조입니다.
2. 특징:
   - 상태 분리: 자식 그래프만의 전용 장부(State)를 사용하여 부모의 복잡도를 낮춥니다.
   - 모듈화: 특정 기능(예: 카피 제작)을 독립된 부품처럼 만들어 여러 곳에서 재사용 가능합니다.
   - 관찰 가능성: stream(subgraphs=True) 옵션을 통해 블랙박스 같던 내부 실행 과정을 투명하게 모니터링할 수 있습니다.
3. 경로(Path): 튜플 형태의 path를 통해 현재 작업이 부모 노드인지 혹은 특정 세부 업무(자식) 내부인지 계층적으로 파악합니다.
"""

