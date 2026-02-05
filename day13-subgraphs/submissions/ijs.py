"""
- 멀티에이전트 시스템 구축 시
- 동일한 로직을 여러 그래프에서 재사용하고 싶을 때
- 팀 간에 그래프 특정 부분을 분리해 개발하고 싶을 때 (단, 입출력 스키마 준수는 필요)

*subgraph와 parent graph간의 통신 방식 정의 필요
1. invoke a graph from a node
2. add a graph as a node
"""

#------------------------------------------
#Invoke a graph from a node
"""
Parent graph
  └ parent_1
      └ child graph
          └ child_1

*서브그래프를 노드 함수 안에서 직접 호출
*부모/자식 state는 분리
*그래서 입출력 변환을 호출자가 직접 해야 함
"""
#------------------------------------------

print(f"\n### invoke a graph from a node ###\n")
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

# Define subgraph
class SubgraphState(TypedDict):
    # note: 부모 그래프와 공유되지 않는 state
    bar: str
    baz: str

def subgraph_node_1(state: SubgraphState):
    return {"baz": "baz"}

def subgraph_node_2(state: SubgraphState):
    return {"bar": state["bar"] + state["baz"]}

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

def node_2(state: ParentState):
    # 상태를 서브그래프 state로 변환
    response = subgraph.invoke({"bar": state["foo"]})
    # 응답을 부모 state로 변환
    return {"foo": response["bar"]}


builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :") #사이트 들어가서 코드 붙여넣기
print(graph.get_graph(xray=True).draw_mermaid())

for chunk in graph.stream({"foo": "foo"}, subgraphs=True):
    print(chunk)


#------------------------------------------
#Add a graph as a node
"""
point: shared messages key

Parent graph (state: foo)
  ├ node_1
  └ node_2 (subgraph)
       ├ subgraph_node_1
       └ subgraph_node_2

*컴파일된 graph 자체를 부모 그래프의 노드로 추가
*부모 state 중 공유 키는 그대로, 서브그래프의 private 키는 서브그래프 내부에서만 사용
*결과로 공유 키 업데이트만 부모로 전달
"""
#------------------------------------------

print(f"\n### add a graph as a node ###\n")

from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

# Define subgraph
class SubgraphState(TypedDict):
    foo: str  # shared with parent graph state
    baz: str  # private to SubgraphState

def subgraph_node_1(state: SubgraphState):
    return {"baz": "baz"}

def subgraph_node_2(state: SubgraphState):
    # 이 노드는 서브그래프 내에서만 사용 가능한 상태 키('bar')를 사용하고 있으며
    # 공유 상태 키('foo')에 대한 업데이트를 전송하고 있습니다.
    return {"foo": state["foo"] + state["baz"]}

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

print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :") #사이트 들어가서 코드 붙여넣기
print(graph.get_graph(xray=True).draw_mermaid())

for chunk in graph.stream({"foo": "foo"}, subgraphs=True):
    print(chunk)


# ------------------------------------------
# View subgraph state: only in interrupt
# ------------------------------------------

print(f"\n### view subgraph state: only in interrupt ###\n")

from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

class State(TypedDict):
    foo: str

# Subgraph

def subgraph_node_1(state: State):
    value = interrupt("Provide value:")
    return {"foo": state["foo"] + value}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")

subgraph = subgraph_builder.compile()

# Parent graph

builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}

graph.invoke({"foo": ""}, config)
parent_state = graph.get_state(config)
print(f"[parent_state]: {parent_state}\n")

# interrupt 동안에만 subgraphs=True로 서브그래프 내부 snapshot을 펼쳐서 볼 수 있음
state_with_subgraphs = graph.get_state(config, subgraphs=True)
if state_with_subgraphs.tasks:
    subgraph_state = state_with_subgraphs.tasks[0].state
    print(f"[subgraph_state]: {subgraph_state}\n")
else:
    print("[subgraph_state]: tasks 없음\n")

# resume the subgraph
graph.invoke(Command(resume="bar"), config)

# 실행이 끝나면 또 못 봄
final_state = graph.get_state(config, subgraphs=True)
if final_state.tasks:
    subgraph_state = final_state.tasks[0].state
    print(f"\n[interrupt 후]: {subgraph_state}")
else:
    print(f"\n[interrupt 후]: 실행 완료 - 서브그래프 상태 접근 불가 (tasks 비어있음)")


# ============================================================
# 🎥 CCTV AI 관제 시스템 - 서브그래프 패턴 응용
# ============================================================
"""
서브그래프 3가지 패턴을 활용한 모듈식 CCTV AI 관제 시스템

구조:
┌─────────────────────────────────────────────────────────┐
│  Parent Graph (CCTVControlState)                        │
│  ├─ initialize (프레임 초기화)                           │
│  ├─ vlm_detection (객체 탐지) ← Add as Node 패턴         │
│  ├─ llm_analysis (상황 분석) ← Invoke from Node 패턴     │
│  ├─ human_approval (고위험 승인) ← Interrupt 패턴        │
│  └─ generate_report (보고서 생성)                        │
└─────────────────────────────────────────────────────────┘
"""

print(f"\n{'='*60}")
print("🎥 CCTV AI 서브그래프 시스템")
print("="*60)

from typing import Literal, Optional, List
from typing_extensions import TypedDict, NotRequired
from langgraph.graph.state import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from datetime import datetime
import uuid

# ============================================================
# 1️⃣ State 정의
# ============================================================

# VLM 탐지 서브그래프용 State (private 키 포함)
class VLMDetectionState(TypedDict):
    frame_id: str                    # 공유: 프레임 식별자
    detected_objects: List[str]      # 공유: 탐지된 객체 목록
    threat_level: str                # 공유: 위협 수준
    raw_confidence: NotRequired[float]  # private: 내부 신뢰도 점수

# LLM 분석 서브그래프용 State (완전 분리)
class LLMAnalysisState(TypedDict):
    context_input: str       # 입력: 상황 컨텍스트
    manual_reference: str    # 내부: 대응 매뉴얼
    analysis_result: str     # 출력: 분석 결과

# 부모 그래프 State
class CCTVControlState(TypedDict):
    frame_id: str
    camera_location: str
    timestamp: str
    detected_objects: List[str]
    threat_level: str
    situation_summary: str
    recommended_action: str
    approval_status: str
    final_report: str


# ============================================================
# 2️⃣ VLM 탐지 서브그래프 (Add as Node 패턴)
# ============================================================
"""
*컴파일된 graph 자체를 부모 그래프의 노드로 추가
*공유 키(frame_id, detected_objects, threat_level)는 자동으로 부모와 동기화
*private 키(raw_confidence)는 서브그래프 내부에서만 사용
"""

def vlm_detect_objects(state: VLMDetectionState):
    """VLM이 CCTV 프레임에서 객체 탐지"""
    print(f"🔍 [VLM] 프레임 {state['frame_id']} 객체 탐지 중...")

    # 시뮬레이션: 실제로는 VLM API 호출
    detected = ["person", "person", "fallen_person"]
    confidence = 0.87

    return {
        "detected_objects": detected,
        "raw_confidence": confidence  # private 키
    }

def vlm_classify_threat(state: VLMDetectionState):
    """탐지된 객체를 기반으로 위협 수준 분류"""
    objects = state.get("detected_objects", [])
    confidence = state.get("raw_confidence", 0.5)

    # 위협 수준 결정 로직
    if "fire" in objects or "weapon" in objects:
        threat = "critical"
    elif "fallen_person" in objects or "violence" in objects:
        threat = "high"
    elif "intrusion" in objects:
        threat = "medium"
    else:
        threat = "low"

    print(f"⚠️  [VLM] 탐지: {objects} → 위협수준: {threat} (신뢰도: {confidence:.2f})")
    return {"threat_level": threat}

# VLM 서브그래프 빌드
vlm_builder = StateGraph(VLMDetectionState)
vlm_builder.add_node("vlm_detect", vlm_detect_objects)
vlm_builder.add_node("vlm_classify", vlm_classify_threat)
vlm_builder.add_edge(START, "vlm_detect")
vlm_builder.add_edge("vlm_detect", "vlm_classify")
vlm_subgraph = vlm_builder.compile()


# ============================================================
# 3️⃣ LLM 분석 서브그래프 (Invoke from Node 패턴)
# ============================================================
"""
*서브그래프를 노드 함수 안에서 직접 호출
*부모/자식 state는 완전 분리
*입출력 변환을 호출자가 직접 해야 함
"""

def llm_load_manual(state: LLMAnalysisState):
    """대응 매뉴얼 로드"""
    manuals = {
        "fallen_person": "1)방송 알림 2)119 신고 3)경비 출동",
        "violence": "1)경비 출동 2)112 신고 3)녹화 보존",
        "fire": "1)화재 경보 2)119 신고 3)대피 안내",
        "default": "1)상황 모니터링 2)이상 시 보고"
    }

    context = state["context_input"]
    matched = manuals["default"]
    for key in manuals:
        if key in context.lower():
            matched = manuals[key]
            break

    return {"manual_reference": matched}

def llm_generate_analysis(state: LLMAnalysisState):
    """LLM이 상황 분석"""
    print(f"🧠 [LLM] 상황 분석 중...")
    analysis = f"[분석] {state['context_input']} | 권장조치: {state['manual_reference']}"
    return {"analysis_result": analysis}

# LLM 서브그래프 빌드
llm_builder = StateGraph(LLMAnalysisState)
llm_builder.add_node("load_manual", llm_load_manual)
llm_builder.add_node("generate_analysis", llm_generate_analysis)
llm_builder.add_edge(START, "load_manual")
llm_builder.add_edge("load_manual", "generate_analysis")
llm_subgraph = llm_builder.compile()


# ============================================================
# 4️⃣ 부모 그래프 노드 정의
# ============================================================

def cctv_initialize(state: CCTVControlState):
    """프레임 초기화"""
    print(f"\n📹 [CCTV] 새 프레임 수신 - {state['camera_location']} @ {state['timestamp']}")
    return {
        "frame_id": state.get("frame_id", str(uuid.uuid4())[:8]),
        "approval_status": "pending"
    }

def cctv_llm_analysis(state: CCTVControlState):
    """LLM 서브그래프를 직접 호출 (Invoke from Node 패턴)"""

    # 부모 상태 → 서브그래프 상태로 변환
    context = f"위치:{state['camera_location']}, 탐지:{state['detected_objects']}, 위협:{state['threat_level']}"

    # 서브그래프 호출
    response = llm_subgraph.invoke({"context_input": context})

    # 서브그래프 응답 → 부모 상태로 변환
    return {
        "situation_summary": response["analysis_result"],
        "recommended_action": response["manual_reference"]
    }

def cctv_route_approval(state: CCTVControlState):
    """위협 수준에 따라 승인 필요 여부 라우팅"""
    if state["threat_level"] in ["high", "critical"]:
        return "human_approval"
    return "generate_report"

def cctv_human_approval(state: CCTVControlState):
    """고위험 상황 시 담당자 승인 대기 (Interrupt 패턴)"""
    print(f"\n🛑 [INTERRUPT] 고위험 상황! 담당자 승인 필요")
    print(f"   위협 수준: {state['threat_level']}")
    print(f"   권장 조치: {state['recommended_action']}")

    # interrupt 발생 - 담당자 승인 대기
    decision = interrupt({
        "type": "approval_request",
        "message": f"⚠️ {state['threat_level'].upper()} 위험 상황!",
        "location": state["camera_location"],
        "detected": state["detected_objects"],
        "options": ["approved", "rejected"]
    })

    print(f"✅ [승인] 담당자 결정: {decision}")
    return {"approval_status": decision}

def cctv_generate_report(state: CCTVControlState):
    """최종 보고서 생성"""
    report = f"""
╔════════════════════════════════════════════╗
║     AEGIS CCTV AI 관제 보고서              ║
╠════════════════════════════════════════════╣
║ 프레임: {state['frame_id']}
║ 위치: {state['camera_location']}
║ 시간: {state['timestamp']}
╠────────────────────────────────────────────╣
║ 탐지: {', '.join(state['detected_objects'])}
║ 위협: {state['threat_level'].upper()}
║ 승인: {state['approval_status']}
╠────────────────────────────────────────────╣
║ {state['situation_summary'][:40]}
╚════════════════════════════════════════════╝"""
    print(f"\n📋 [REPORT] 보고서 생성 완료")
    return {"final_report": report}


# ============================================================
# 5️⃣ 부모 그래프 조립
# ============================================================

cctv_builder = StateGraph(CCTVControlState)

# 노드 추가
cctv_builder.add_node("initialize", cctv_initialize)
cctv_builder.add_node("vlm_detection", vlm_subgraph)  # ★ Add as Node 패턴
cctv_builder.add_node("llm_analysis", cctv_llm_analysis)  # ★ Invoke from Node 패턴
cctv_builder.add_node("human_approval", cctv_human_approval)  # ★ Interrupt 패턴
cctv_builder.add_node("generate_report", cctv_generate_report)

# 엣지 연결
cctv_builder.add_edge(START, "initialize")
cctv_builder.add_edge("initialize", "vlm_detection")
cctv_builder.add_edge("vlm_detection", "llm_analysis")
cctv_builder.add_conditional_edges(
    "llm_analysis",
    cctv_route_approval,
    {"human_approval": "human_approval", "generate_report": "generate_report"}
)
cctv_builder.add_edge("human_approval", "generate_report")
cctv_builder.add_edge("generate_report", END)

# 체크포인터로 컴파일 (interrupt 지원)
cctv_checkpointer = InMemorySaver()
cctv_graph = cctv_builder.compile(checkpointer=cctv_checkpointer)


# ============================================================
# 6️⃣ 실행
# ============================================================

print("\n📊 [Graph Structure] Mermaid (https://mermaid.live/):")
print(cctv_graph.get_graph(xray=True).draw_mermaid())

# 테스트 입력
cctv_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

cctv_input = {
    "camera_location": "주차장 B구역 3층",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "frame_id": "FRAME_001",
    "detected_objects": [],
    "threat_level": "low",
}

print("\n" + "-"*50)
print("▶️  1단계: 실행 (고위험 시 interrupt 발생)")
print("-"*50)

for event in cctv_graph.stream(cctv_input, cctv_config, subgraphs=True):
    if isinstance(event, tuple):
        ns, ev = event
        print(f"  [서브그래프:{ns}] {list(ev.keys())}")
    else:
        print(f"  [이벤트] {list(event.keys())}")

# 상태 확인
cctv_snapshot = cctv_graph.get_state(cctv_config)

if cctv_snapshot.next:
    print("\n" + "-"*50)
    print("🛑 Interrupt 발생! 서브그래프 상태 확인 가능")
    print("-"*50)

    full_state = cctv_graph.get_state(cctv_config, subgraphs=True)
    print(f"다음 노드: {cctv_snapshot.next}")

    if full_state.tasks and full_state.tasks[0].interrupts:
        interrupt_info = full_state.tasks[0].interrupts[0].value
        print(f"Interrupt 정보: {interrupt_info}")

    print("\n" + "-"*50)
    print("▶️  2단계: 담당자 승인 후 재개 (approved)")
    print("-"*50)

    for event in cctv_graph.stream(Command(resume="approved"), cctv_config):
        print(f"  [이벤트] {list(event.keys())}")

# 최종 결과
final = cctv_graph.get_state(cctv_config)
print("\n" + "="*50)
print("✅ 최종 보고서")
print("="*50)
print(final.values.get("final_report", "보고서 없음"))

