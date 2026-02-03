"""
[Windows 인코딩 문제 해결]
실행 시 아래 환경변수 설정 필요:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 conda run -n lang python bgm.py

- PYTHONUTF8=1: Python 전체를 UTF-8 모드로 강제
- PYTHONIOENCODING=utf-8: 표준 입출력(stdin/stdout/stderr)을 UTF-8로 설정
- Windows 기본 인코딩(CP949)이 LLM 응답의 일부 유니코드 문자를 처리 못해서 발생하는 문제 방지

실행 이력을 저장해두고, 과거로 돌아가 분석·재실험 가능
- Checkpoint: 특정 시점 상태의 스냅샷
- thread_id: 스레드 ID
- Fork: 기존 결과를 덮지 않고 새로운 분기 생성

[AEGIS AI 관제 솔루션]
VLM(상황분류) → LLM Agent(상황맥락 파악) → LLM Agent(대응방안 결정)
상황 분류 5가지: 폭행, 실신, 기물파손, 절도, 무단투기
"""

from dotenv import load_dotenv
load_dotenv()

import uuid

from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    situation_type: NotRequired[str]  # VLM이 분류한 상황 유형 (5가지 중 하나)
    context: NotRequired[str]         # Agent가 파악한 상황 맥락
    response: NotRequired[str]        # Agent가 결정한 대응 방안


model = init_chat_model(
    "claude-haiku-4-5-20251001",
    temperature=0,
)


def classify_situation(state: State):
    """VLM이 CCTV 영상 프레임을 분석하여 이상 상황을 5가지로 분류"""
    msg = model.invoke(
        "당신은 AEGIS AI 관제 솔루션의 VLM입니다. "
        "CCTV 영상에서 감지된 이상 상황을 다음 5가지 중 하나로 분류해주세요: "
        "[1.폭행, 2.실신, 3.기물파손, 4.절도, 5.무단투기]. "
        "지금 야간 주차장에서 두 사람이 언쟁 후 한 명이 주먹을 휘두르는 장면이 감지되었습니다. "
        "상황 유형만 간단히 답변해주세요."
    )
    return {"situation_type": msg.content}


def analyze_context(state: State):
    """LLM Agent가 상황 맥락을 능동적으로 파악"""
    msg = model.invoke(
        f"당신은 AEGIS AI 관제 솔루션의 LLM Agent입니다. "
        f"VLM이 분류한 상황: [{state['situation_type']}]\n"
        f"추가 영상 프레임을 확인한 결과, 야간 주차장 B구역에서 발생한 상황입니다. "
        f"관련자는 2명이며, 한 명은 바닥에 쓰러져 있습니다. "
        f"이 정보를 바탕으로 상황 맥락을 텍스트로 정리해주세요. 간결하게 작성해주세요."
    )
    return {"context": msg.content}


def generate_response(state: State):
    """LLM Agent가 매뉴얼과 상황을 종합하여 대응방안 결정"""
    msg = model.invoke(
        f"당신은 AEGIS AI 관제 솔루션의 LLM Agent입니다. "
        f"상황 맥락: [{state['context']}]\n"
        f"[대응 매뉴얼] 폭행 상황 시: 1)경비실 즉시 출동 요청, 2)경찰 신고(112), 3)피해자 응급상황 시 119 신고, 4)CCTV 녹화 보존\n"
        f"위 매뉴얼과 상황 맥락을 종합하여 구체적인 대응방안을 결정해주세요. 간결하게 작성해주세요."
    )
    return {"response": msg.content}


# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("classify_situation", classify_situation)
workflow.add_node("analyze_context", analyze_context)
workflow.add_node("generate_response", generate_response)

# Add edges to connect nodes
workflow.add_edge(START, "classify_situation")
workflow.add_edge("classify_situation", "analyze_context")
workflow.add_edge("analyze_context", "generate_response")
workflow.add_edge("generate_response", END)

# Compile
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
graph

config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
    }
}
state = graph.invoke({}, config)

print("\n\n ## 초기 실행 결과 (AEGIS 관제) ##")
print(f"**상황 분류: {' '.join(state['situation_type'].split())[:30]}")
print(f"**상황 맥락: {' '.join(state['context'].split())[:30]}")
print(f"**대응 방안: {' '.join(state['response'].split())[:30]}")

# The states are returned in reverse chronological order.
states = list(graph.get_state_history(config))

print("\n\n ## 체크포인트 히스토리 ##")
for idx, state in enumerate(states):
    print(f"\n {idx}. 다음 실행 노드: {state.next}")
    print(f"체크포인트 ID: {state.config['configurable']['checkpoint_id']}")
    # print(f"\nraw: {state}")

# 마지막에서 두 번째 상태 선택 (상황분류는 되었지만 맥락 파악 전 시점)
selected_state = states[2]
print("\n\n ## 선택된 과거 시점 (Time Travel) ##", {selected_state.config['configurable']['checkpoint_id']})
print(f"다음 실행 노드: {selected_state.next}")
print(f"해당 시점의 상태값:\n{selected_state.values}")


# 상태 업데이트 (상황 분류를 '기물파손'으로 변경 - 다른 시나리오로 분기)
new_config = graph.update_state(selected_state.config, values={"situation_type": "3.기물파손 - 야간 주차장 A구역에서 차량 파손 행위 감지"})
print(f"\n새로운 설정: {new_config}")
print(f"\n\n ## 변경된 설정의 히스토리 ##")
print(f"{list(graph.get_state_history(new_config))}")


result = graph.invoke(None, new_config)  # None 새 입력 없음, 상태만 이어서 실행
print("\n\n ## 변경된 상황으로 재실행한 최종 결과 ##")
print(f"**상황 분류: {' '.join(result['situation_type'].split())[:30]}")
print(f"**상황 맥락: {' '.join(result['context'].split())[:30]}")
print(f"**대응 방안: {' '.join(result['response'].split())[:30]}")
print(f"{list(graph.get_state_history(config))}")  # thread_id로 전체 이력을 확인

"""
(원래 실행 - 폭행 시나리오)
start → (상황분류: 폭행) → (맥락파악: 야간 주차장 폭행) → (대응: 경비+112+119) → END

                       └─ (update_state로 분기 - 기물파손 시나리오)
                          (상황분류: 기물파손) → (맥락파악: 차량 파손) → (대응: 경비+112+CCTV보존) → END

"""
