# 환경 변수 로드를 위한 dotenv 설정
from dotenv import load_dotenv
load_dotenv()

import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage, BaseMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

# 1. 환경 설정 및 모델 초기화 (Claude Haiku 사용)
model = init_chat_model("claude-haiku-4-5-20251001")

# 2. 상태 정의 (메시지 히스토리 + 요약본)
class FinanceState(MessagesState):
    summary: str

# ---------------------------------------------------------
# 3. 서브그래프 정의 (투자 승인 프로세스 - Interrupt 활용)
# ---------------------------------------------------------
class SubState(TypedDict):
    decision: str

def investment_approval_node(state: SubState):
    # 사용자의 최종 승인을 기다리는 인터럽트
    answer = interrupt({"question": "정말로 해당 포트폴리오로 투자를 실행하시겠습니까? (yes/no)"})
    return {"decision": f"사용자 승인 결과: {answer}"}

sub_builder = StateGraph(SubState)
sub_builder.add_node("approval_node", investment_approval_node)
sub_builder.add_edge(START, "approval_node")
investment_subgraph = sub_builder.compile() # 필요시 별도 checkpointer 가능

# ---------------------------------------------------------
# 4. 메인 그래프 노드 정의 (Trim, Summarize, Delete 포함)
# ---------------------------------------------------------

# [Node 1] 모델 호출 노드 (Trim 전략 적용)
def financial_advisor(state: FinanceState):
    messages = state["messages"]
    
    # 요약본이 있다면 컨텍스트에 추가
    if state.get("summary"):
        summary_msg = HumanMessage(content=f"[이전 상담 요약]: {state['summary']}")
        messages = [summary_msg] + messages

    # Trim 전략: 최근 메시지 위주로 최대 128토큰까지만 유지 (효율 극대화)
    trimmed_messages = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=128,
        start_on="human",
        end_on=("human", "tool"),
    )
    
    response = model.invoke(trimmed_messages)
    return {"messages": [response]}

# [Node 2] 메모리 관리 노드 (Summarize + Delete 전략 적용)
def manage_memory(state: FinanceState):
    messages = state["messages"]
    
    # 메시지가 6개를 넘으면 요약하고 오래된 메시지 삭제
    if len(messages) > 6:
        print("\n--- 🛠️ 메모리 최적화 수행 중 (요약 및 정리) ---")
        
        # 요약 생성
        existing_summary = state.get("summary", "")
        summary_prompt = (
            f"기존 요약: {existing_summary}\n"
            "추가된 대화 내용을 바탕으로 사용자의 자산 상황과 투자 성향을 한 문장으로 업데이트해줘."
        )
        summary_response = model.invoke(messages + [HumanMessage(content=summary_prompt)])
        
        # Delete 전략: 최신 2개 메시지 제외하고 모두 삭제 (RemoveMessage)
        delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
        
        return {
            "summary": summary_response.content,
            "messages": delete_messages
        }
    return {}

# [조건부 라우팅 함수] 투자 실행 요청 감지 시 서브그래프로 분기
def should_trigger_investment(state: FinanceState) -> str:
    """마지막 메시지에서 투자 실행 키워드 감지"""
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        # HumanMessage인 경우만 확인
        if hasattr(last_msg, 'content'):
            content = last_msg.content.lower() if isinstance(last_msg.content, str) else str(last_msg.content).lower()
            # 투자 실행 관련 키워드 감지
            investment_keywords = ["투자 실행", "포트폴리오로 투자", "실행해줘", "투자해줘"]
            if any(keyword in content for keyword in investment_keywords):
                return "investment_process"  # 서브그래프로 분기
    return "advisor"  # 기본 advisor 노드로

# 5. 그래프 결합
builder = StateGraph(FinanceState)

builder.add_node("advisor", financial_advisor)
builder.add_node("memory_manager", manage_memory)
builder.add_node("investment_process", investment_subgraph)  # 서브그래프 추가

# 조건부 라우팅: START에서 투자 실행 여부에 따라 분기
# path_map으로 가능한 경로를 명시해야 Mermaid 시각화에서 edge가 표시됨
builder.add_conditional_edges(
    START, 
    should_trigger_investment,
    path_map={
        "advisor": "advisor",              # 일반 대화 → advisor 노드
        "investment_process": "investment_process"  # 투자 실행 → 서브그래프
    }
)
builder.add_edge("advisor", "memory_manager")
builder.add_edge("memory_manager", END)
builder.add_edge("investment_process", END)  # 서브그래프 완료 후 종료

# 체크포인터 설정 (단기 메모리)
checkpointer = InMemorySaver()
app = builder.compile(checkpointer=checkpointer)

# ---------------------------------------------------------
# 6. 워크플로우 시각화 (Mermaid 그래프)
# ---------------------------------------------------------
print("=== 워크플로우 시각화 ===")
print("아래 Mermaid 코드를 https://mermaid.live/ 에 붙여넣으면 그래프를 볼 수 있습니다:\n")

# 커스텀 Mermaid 그래프 (한글 설명 + 이모지 포함)
custom_mermaid = """
flowchart TD
    %% 메인 그래프 정의
    subgraph MainGraph["📋 메인 그래프"]
        START((START))
        START -->|조건부 라우팅| COND{should_trigger_investment}
        
        %% 분기
        COND -->|일반 대화| ADVISOR["💼 advisor"]
        COND -->|투자 실행| INVEST["📈 investment_process"]
        
        %% 일반 대화 흐름
        ADVISOR --> MEMORY["🧠 memory_manager"]
        MEMORY --> END1((invoke 완료))
        
        %% 투자 실행 흐름
        INVEST --> END2((🚫 세션 종료))
        
        %% 일반 대화 Loop (뒤로 돌아가기)
        END1 -.->|대화 계속| START
    end
    
    %% 서브그래프 정의 (MainGraph 아래가 아닌 옆에 배치 유도)
    subgraph SubGraph["📈 서브그래프"]
        S_START((START)) --> APPROVAL["✋ approval_node"] --> S_END((END))
    end
    
    %% 연결 및 배치
    INVEST -.->|포함| SubGraph
"""
print(custom_mermaid)

# ---------------------------------------------------------
# 7. 실행 시나리오 테스트
# ---------------------------------------------------------
config = {"configurable": {"thread_id": "user_123"}}

print("=== 1. 일반 대화 (메모리 누적) ===")

print("\n[입력 1]: 안녕, 나 자산 관리 좀 도와줘. 현재 자산은 1억 정도 있어.")
app.invoke({"messages": [HumanMessage(content="안녕, 나 자산 관리 좀 도와줘. 현재 자산은 1억 정도 있어.")]}, config)

print("\n[입력 2]: 위험한 투자보다는 안정적인 배당주를 선호해.")
app.invoke({"messages": [HumanMessage(content="위험한 투자보다는 안정적인 배당주를 선호해.")]}, config)

print("\n[입력 3]: 연 5% 정도 수익률을 목표로 하고 싶어.")
app.invoke({"messages": [HumanMessage(content="연 5% 정도 수익률을 목표로 하고 싶어.")]}, config)

print("\n=== 2. 메모리 최적화 확인 (요약 및 삭제 발생 지점) ===")
# 여러 번 질문하여 메시지 개수를 늘림
print("\n[입력 4]: 내 자산 상황 다시 알려주고, 추천 섹터 하나만 말해줘.")
res = app.invoke({"messages": [HumanMessage(content="내 자산 상황 다시 알려주고, 추천 섹터 하나만 말해줘.")]}, config)
print(f"\n현재 요약본: {res.get('summary')}")
print(f"남은 메시지 개수: {len(res['messages'])}")

print("\n=== 3. 서브그래프 및 Interrupt 테스트 ===")
# 투자 실행 요청 - 인터럽트가 발생함
investment_config = {"configurable": {"thread_id": "user_investment"}}
print("\n[입력 5]: 결정했어, 추천해준 포트폴리오로 투자 실행해줘.")
result = app.invoke({"messages": [HumanMessage(content="결정했어, 추천해준 포트폴리오로 투자 실행해줘.")]}, investment_config)

# 인터럽트 결과 확인 (example.py 방식)
if "__interrupt__" in result:
    print(f"\n[인터럽트 발생]: {result['__interrupt__']}")
    print("[시스템 노티스]: 투자 실행 전 사용자의 최종 승인이 필요합니다.")
    
    # 사용자가 승인(resume)을 보냄
    print("\n[입력 6 - Resume]: yes")
    final_state = app.invoke(Command(resume="yes"), investment_config)
    print("\n최종 결과:")
    if final_state.get("messages"):
        final_state["messages"][-1].pretty_print()
    else:
        print(f"승인 결과: {final_state}")
else:
    print("인터럽트 없이 완료됨")
    final_state = result

# ---------------------------------------------------------
# 7. 서브그래프 상태 히스토리 조회 (example.py 참고)
# ---------------------------------------------------------
print("\n=== 4. 상태 히스토리 조회 ===")

# 전체 히스토리 조회
history = list(app.get_state_history(investment_config))
print(f"전체 히스토리 스냅샷 개수: {len(history)}")

# 서브그래프 네임스페이스 추출
subgraph_namespaces = set()
for snap in history:
    for task in snap.tasks or []:
        state = task.state
        if state and "checkpoint_ns" in state.get("configurable", {}):
            subgraph_namespaces.add(
                state["configurable"]["checkpoint_ns"]
            )

# 서브그래프별 메모리 상태 출력
for ns in subgraph_namespaces:
    sub_config = {
        "configurable": {
            "thread_id": "user_investment",
            "checkpoint_ns": ns
        }
    }
    print(f"\n--- 서브그래프 메모리: {ns} ---")
    sub_history = list(app.get_state_history(sub_config))
    print(f"서브그래프 히스토리 개수: {len(sub_history)}")

print("\n=== 모든 테스트 완료 ===")

