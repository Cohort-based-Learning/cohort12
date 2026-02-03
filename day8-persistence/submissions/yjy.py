# ============================================================
# [Day 8] LangGraph 통합 예제: AI 명탐정 셜록
#
# 주제: 살인 사건 수사
# 1. Memory Store (수사 수첩):
#    - 중요한 단서는 '수첩'에 영구 저장됩니다.
#    - 다른 형사(새로운 Thread)가 와서 물어봐도 대답할 수 있습니다. (전역 기억)
#
# 2. Checkpoints (취조 기록):
#    - 용의자와의 대화 흐름을 저장합니다.
#    - "잠깐, 아까 그 말 취소할게"라며 과거로 돌아가 다시 심문할 수 있습니다. (타임 트래블)
# ============================================================

import os
import uuid
from dotenv import load_dotenv
from typing import Annotated, List
from typing_extensions import TypedDict
from operator import add

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

# 환경 변수 로드
load_dotenv()

# ------------------------------------------------------------
# 1. 모델 및 저장소 설정 (Gemini Free Tier)
# ------------------------------------------------------------

# LLM: 가볍고 빠른 Flash 모델
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# [장기 기억] 수사 수첩
# [수정] 임베딩 모델 없이 기본 저장소로 사용 (API 에러 방지)
memory_store = InMemoryStore()

# [단기 기억] 대화 상태 저장소
checkpointer = InMemorySaver()


# ------------------------------------------------------------
# 2. 그래프 상태(State) 및 노드 정의
# ------------------------------------------------------------

class DetectiveState(TypedDict):
    # 대화 내역 (누적됨)
    messages: Annotated[List[BaseMessage], add]

# 노드 1: 추리 및 기록 (The Brain)
def detective_node(state: DetectiveState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"].get("user_id", "default_user")
    namespace = (user_id, "case_file_001") # 사건 번호 001에 대한 수첩

    # 사용자의 마지막 말
    last_msg = state["messages"][-1].content

    # 1. [기억 검색] 수첩에서 단서 찾아오기
    # [수정] query(유사도 검색) 대신 네임스페이스의 모든 기억을 가져옵니다.
    memories = store.search(namespace, limit=10)
    
    memory_context = ""
    if memories:
        # 가져온 메모리 객체에서 값(value) 추출
        found_clues = [m.value['content'] for m in memories]
        memory_context = "\n".join(found_clues)
        print(f"   📖 [수첩 확인] 기록된 단서들: {found_clues}")
    else:
        print("   📖 [수첩 확인] 기록된 단서 없음.")

    # 2. [LLM 추리] 답변 생성
    system_prompt = f"""
    당신은 명탐정 셜록입니다. 
    사용자(동료 형사 또는 증인)와 대화하며 사건을 수사하세요.
    
    [수사 수첩(장기 기억)]
    {memory_context}
    
    지시사항:
    1. 사용자가 새로운 단서(범인 특징, 장소 등)를 말하면 "단서가 기록되었습니다"라고 말하세요.
    2. 수첩의 내용을 바탕으로 추리하여 답변하세요.
    3. 거만하지만 천재적인 말투를 사용하세요.
    """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"] # 대화 히스토리 전체 전달
    ])

    # 3. [기억 저장] 만약 중요한 단서라면 수첩에 기록
    if "단서" in last_msg or "범인" in last_msg or "증거" in last_msg or "스카프" in last_msg or "넥타이" in last_msg:
        print(f"   ✍️ [수첩 기록] 새로운 정보를 적습니다...")
        store.put(
            namespace,
            str(uuid.uuid4()),
            {"content": last_msg}
        )

    return {"messages": [response]}


# ------------------------------------------------------------
# 3. 그래프 빌드
# ------------------------------------------------------------
workflow = StateGraph(DetectiveState)
workflow.add_node("detective", detective_node)
workflow.add_edge(START, "detective")
workflow.add_edge("detective", END)

# 체크포인터(대화용)와 스토어(수첩용)를 모두 장착
app = workflow.compile(checkpointer=checkpointer, store=memory_store)


# ============================================================
# 4. 실행 시나리오 (Simulation)
# ============================================================

# --- [Scene 1] 왓슨과의 대화 (단서 수집) ---
config_watson = {
    "configurable": {
        "thread_id": "watson_session",
        "user_id": "scotland_yard" # 수첩 공유를 위한 ID
    }
}

print("\n🕵️ [Scene 1] 왓슨 박사가 현장 정보를 보고합니다.")
print("-" * 50)

input_1 = {"messages": [HumanMessage(content="셜록, 단서가 나왔어. 범인은 '왼쪽 다리를 전다'고 해.")]}
for update in app.stream(input_1, config_watson, stream_mode="updates"):
    print(f"셜록: {update['detective']['messages'][0].content}")

input_2 = {"messages": [HumanMessage(content="그리고 현장에서 '빨간색 스카프'가 발견됐어.")]}
for update in app.stream(input_2, config_watson, stream_mode="updates"):
    print(f"셜록: {update['detective']['messages'][0].content}")


# --- [Scene 2] 레스트레이드 경감과의 대화 (기억 공유 확인) ---
print("\n👮 [Scene 2] 레스트레이드 경감이 수사 상황을 묻습니다. (다른 쓰레드)")
print("-" * 50)

config_lestrade = {
    "configurable": {
        "thread_id": "lestrade_session",
        "user_id": "scotland_yard" # 왓슨과 같은 수첩을 공유
    }
}

input_3 = {"messages": [HumanMessage(content="어이 셜록, 범인의 인상착의에 대해 알아낸 게 있나?")]}
for update in app.stream(input_3, config_lestrade, stream_mode="updates"):
    print(f"셜록(To 경감): {update['detective']['messages'][0].content}")


# --- [Scene 3] 타임 트래블 (Checkpoints 활용) ---
print("\n⏳ [Scene 3] 타임 트래블: 왓슨과의 대화 중 '빨간 스카프' 얘기 전으로 되감기")
print("-" * 50)

history = list(app.get_state_history(config_watson))
target_checkpoint = history[1].config
print(f"돌아갈 시점 ID: {target_checkpoint['configurable']['checkpoint_id']}")

past_state = app.get_state(target_checkpoint)
print(f"과거 시점의 대화 내용: {[m.content for m in past_state.values['messages']]}")

print("\n▶️ [재개] 과거 시점에서 다시 대화합니다. (빨간 스카프 대신 다른 정보 입력)")
config_forked = target_checkpoint

input_fork = {"messages": [HumanMessage(content="아 정정할게. 스카프가 아니라 '파란색 넥타이'였어.")]}
for update in app.stream(input_fork, config_forked, stream_mode="updates"):
    print(f"셜록(과거 수정): {update['detective']['messages'][0].content}")
