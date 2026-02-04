"""
실행 이력을 저장해두고, 과거로 돌아가 분석·재실험 가능
- Checkpoint: 특정 시점 상태의 스냅샷
- thread_id: 스레드 ID
- Fork: 기존 결과를 덮지 않고 새로운 분기 생성
"""

from dotenv import load_dotenv

load_dotenv()

import uuid

from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]
    evaluation: NotRequired[str]


model = init_chat_model(
    "gpt-4o-mini",
    temperature=0,
)


def generate_topic(state: State):
    """농담 주제를 생성하는 LLM 호출"""
    msg = model.invoke(
        "개발자를 주제로 재미있는 농담 주제를 하나 짧게 알려줘 without emojis"
    )
    return {"topic": msg.content}


def write_joke(state: State):
    """주제를 기반으로 농담을 작성하는 LLM 호출"""
    msg = model.invoke(f"{state['topic']}에 대한 짧은 농담을 작성해줘 without emojis")
    return {"joke": msg.content}


def evaluate_joke(state: State):
    """농담의 재미를 평가하는 LLM 호출"""
    msg = model.invoke(f"{state['joke']}에 대한 재미를 평가해줘 without emojis")
    return {"evaluation": msg.content}


# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate_topic", generate_topic)
workflow.add_node("write_joke", write_joke)
workflow.add_node("evaluate_joke", evaluate_joke)

# Add edges to connect nodes
workflow.add_edge(START, "generate_topic")
workflow.add_edge("generate_topic", "write_joke")
workflow.add_edge("write_joke", "evaluate_joke")
workflow.add_edge("evaluate_joke", END)

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

print("\n\n ## 초기 실행 결과 ##")
print(f"**생성된 주제: {" ".join(state['topic'].split())[:10]}")
print(f"**작성된 농담: {" ".join(state['joke'].split())[:10]}")
print(f"**농담 평가: {" ".join(state['evaluation'].split())[:10]}")

# The states are returned in reverse chronological order.
states = list(graph.get_state_history(config))

print("\n\n ## 체크포인트 히스토리 ##")
for idx, state in enumerate(states):
    print(f"\n {idx}. 다음 실행 노드: {state.next}")
    print(f"체크포인트 ID: {state.config['configurable']['checkpoint_id']}")
    # print(f"\nraw: {state}")

# 마지막에서 두 번째 상태 선택 (주제는 생성되었지만 농담은 아직 작성되지 않은 시점)
selected_state = states[2]
print(
    "\n\n ## 선택된 과거 시점 (Time Travel) ##",
    {selected_state.config["configurable"]["checkpoint_id"]},
)
print(f"다음 실행 노드: {selected_state.next}")
print(f"해당 시점의 상태값:\n{selected_state.values}")


# 상태 업데이트 (주제를 닭으로 변경)
new_config = graph.update_state(
    selected_state.config, values={"topic": "닭"}
)  # 새로운 설정을 적용
print(f"\n새로운 설정: {new_config}")
print(f"\n\n ## 변경된 설정의 히스토리 ##")  # 최신 체크포인트 하나임을 확인
print(f"{list(graph.get_state_history(new_config))}")


result = graph.invoke(None, new_config)  # None 새 입력 없음, 상태만 이어서 실행
print("\n\n ## 변경된 주제로 재실행한 최종 결과 ##")
print(f"**생성된 주제: {" ".join(result['topic'].split())[:10]}")
print(f"**작성된 농담: {" ".join(result['joke'].split())[:10]}")
print(f"**농담 평가: {" ".join(result['evaluation'].split())[:10]}")
print(f"{list(graph.get_state_history(config))}")  # thread_id로 전체 이력을 확인

"""
(원래 실행)
start → (topic 생성: 개발자 개그) → (joke 생성: 개발자 개그 농담) → (평가: 개발자 개그 농담) → END

                       └─ (update_state로 분기)
                          (topic=닭, joke 없음) → (joke 생성: 닭 농담) → (평가: 닭 농담) → END

"""
