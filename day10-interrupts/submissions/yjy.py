# pip install langgraph langgraph-checkpoint-sqlite langchain-google-genai pydantic

from typing import TypedDict, Literal, Optional
from datetime import datetime
import json

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------
# 1. LLM & 구조화된 출력 설정 (안전장치)
# -----------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1 # 분석용이므로 낮게 설정
)

# LLM이 뱉어야 할 정확한 JSON 스키마
class AnalysisOutput(BaseModel):
    summary: str = Field(description="사건의 3줄 요약")
    risk_score: float = Field(description="0.0 ~ 1.0 사이의 위험도 점수")

# 구조화된 출력을 강제하는 LLM 객체 생성
structured_llm = llm.with_structured_output(AnalysisOutput)

# -----------------------------------------
# 2. State 정의 (유지)
# -----------------------------------------
class EventState(TypedDict):
    vlm_label: Literal["normal", "abnormal"]
    event_type: Optional[str]
    camera_location: str
    occurred_at: str
    summary: Optional[str]
    risk_score: Optional[float]
    status: Literal["pending", "approved", "rejected"]
    action_manual: Optional[str]
    similar_cases: Optional[str]
    report: Optional[str]

# -----------------------------------------
# 3. Node 정의
# -----------------------------------------

def event_validation_node(state: EventState):
    """이벤트가 Normal인지 Abnormal인지 1차 분기"""
    print(f"\n🔍 [검증] VLM 판단 결과: {state['vlm_label']}")
    if state["vlm_label"] == "normal":
        return Command(goto="end_normal")
    return Command(goto="abnormal_type_validation")

def abnormal_type_validation_node(state: EventState):
    """
    🛑 Interrupt 1: VLM이 감지한 유형이 맞는지 사람이 확정
    """
    # 여기서 멈추고 UI에 질문을 던짐
    confirmed_type = interrupt({
        "type": "validation",
        "msg": f"현재 감지된 유형 '{state['event_type']}'이(가) 맞습니까?",
        "candidates": ["fall", "violence", "intrusion", "fire"]
    })

    # Resume 시 들어온 데이터로 업데이트
    print(f"✅ [확정] 담당자가 유형을 '{confirmed_type}'(으)로 확정했습니다.")
    return {
        "event_type": confirmed_type
    }

def llm_analysis_node(state: EventState):
    """LLM이 사건을 정밀 분석 (구조화된 출력 사용)"""
    print("🧠 [AI] Gemini가 사건을 정밀 분석 중...")

    prompt = f"""
    상황: {state['event_type']}
    위치: {state['camera_location']}
    시간: {state['occurred_at']}
    
    위 CCTV 관제 상황에 대해 보안 보고서용 요약과 위험도를 평가해줘.
    """

    # .invoke() 하면 Pydantic 객체가 반환됨 (파싱 에러 없음)
    result: AnalysisOutput = structured_llm.invoke(prompt)

    return {
        "summary": result.summary,
        "risk_score": result.risk_score
    }

def approval_node(state: EventState):
    """
    🛑 Interrupt 2: 분석 결과를 보고 담당자가 최종 진행 승인
    """
    # 여기서 또 멈춤
    decision = interrupt({
        "type": "approval",
        "msg": "분석 결과를 승인하고 RAG를 진행하시겠습니까?",
        "summary": state["summary"],
        "risk_score": state["risk_score"]
    })

    if decision:
        print("✅ [승인] 담당자가 후속 조치를 승인했습니다.")
        return Command(goto="action_planning")
    else:
        print("❌ [반려] 담당자가 이벤트를 종료시켰습니다.")
        return Command(goto="reject_event")

def action_planning_node(state: EventState):
    """RAG 등 후속 조치 (Mock)"""
    print("📚 [RAG] 매뉴얼 및 유사 사례 검색 중...")
    return {
        "action_manual": f"[{state['event_type']}] 표준 대응 절차 2.0",
        "similar_cases": "2024년 12월 유사 사건(ID:992) 참조"
    }

def report_node(state: EventState):
    """최종 보고서 생성"""
    report = f"""
    [🚨 AEGIS 보안 리포트]
    --------------------------------
    유형: {state['event_type']} (위험도 {state['risk_score']})
    요약: {state['summary']}
    조치: {state['action_manual']}
    상태: 승인됨 (Approved)
    """
    print("📄 [완료] 최종 보고서가 생성되었습니다.")
    return {"report": report, "status": "approved"}

def reject_node(state: EventState):
    return {"status": "rejected"}

def end_normal_node(state: EventState):
    return {"status": "approved"}

# -----------------------------------------
# 4. Graph 빌드
# -----------------------------------------
builder = StateGraph(EventState)

builder.add_node("event_validation", event_validation_node)
builder.add_node("abnormal_type_validation", abnormal_type_validation_node)
builder.add_node("llm_analysis", llm_analysis_node)
builder.add_node("approval", approval_node)
builder.add_node("action_planning", action_planning_node)
builder.add_node("report", report_node)
builder.add_node("reject_event", reject_node)
builder.add_node("end_normal", end_normal_node)

builder.add_edge(START, "event_validation")
builder.add_edge("abnormal_type_validation", "llm_analysis")
builder.add_edge("llm_analysis", "approval")
builder.add_edge("action_planning", "report")
builder.add_edge("report", END)
builder.add_edge("reject_event", END)
builder.add_edge("end_normal", END)

# -----------------------------------------
# 5. 실행 시나리오 (핵심 수정 부분)
# -----------------------------------------
# DB 연결 (메모리 대신 파일 DB 사용 권장)
with SqliteSaver.from_conn_string("aegis_v2.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

    # 스레드 ID (이 ID가 같으면 대화가 유지됨)
    config = {"configurable": {"thread_id": "event_case_999"}}

    print("\n🎬 [Scenario] 폭력 의심 상황 발생!")
    initial_input = {
        "vlm_label": "abnormal",
        "event_type": "unknown", # 초기엔 모름 -> 사람이 확정해줘야 함
        "camera_location": "로비 A구역",
        "occurred_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # --- 1단계: 최초 실행 ---
    # abnormal_type_validation에서 멈출 것임
    print("\n▶️ 1단계 실행 중...")
    for event in graph.stream(initial_input, config):
        pass

    # 상태 확인
    snapshot = graph.get_state(config)
    if snapshot.next:
        interrupt_info = snapshot.tasks[0].interrupts[0].value
        print(f"\n🛑 [Interrupt 1 발생] {interrupt_info['msg']}")

        # --- 2단계: 유형 확정 (Resume) ---
        # 사용자가 "violence"라고 입력했다고 가정
        print("⌨️  사용자 입력: 'violence'")

        # Command(resume="값")을 통해 값을 전달하며 재개
        print("\n▶️ 2단계 실행 중 (유형 확정)...")
        for event in graph.stream(Command(resume="violence"), config):
            pass

    # 상태 확인 (다시 멈췄는지)
    snapshot = graph.get_state(config)
    if snapshot.next:
        interrupt_info = snapshot.tasks[0].interrupts[0].value
        print(f"\n🛑 [Interrupt 2 발생] {interrupt_info['msg']}")
        print(f"   (AI 분석 결과: 위험도 {interrupt_info['risk_score']})")

        # --- 3단계: 최종 승인 (Resume) ---
        # 사용자가 "승인(True)" 했다고 가정
        print("⌨️  사용자 입력: 승인(Yes)")

        print("\n▶️ 3단계 실행 중 (최종 리포트 생성)...")
        for event in graph.stream(Command(resume=True), config):
            pass

    # 최종 결과 확인
    final_snapshot = graph.get_state(config)
    print(f"\n🎉 최종 결과 Report:\n{final_snapshot.values['report']}")