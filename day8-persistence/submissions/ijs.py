"""
체크포인터를 붙여 컴파일하면, 그래프 상태가 매 super-step마다 자동 저장(체크포인트) 된다.
저장된 체크포인트는 thread(스레드) 에 쌓이며, 실행이 끝난 뒤에도 상태 조회/재개/분기가 가능해진다.
그 결과로 HITL(중간에 사람 개입), 대화/세션 메모리, 과거 시점 재생(타임 트래블), 실패 지점부터 복구(장애 내성) 같은 기능이 열린다.
Agent Server를 쓰면 이런 저장/관리(체크포인팅)를 서버가 자동으로 해줘서 직접 설정할 필요가 없다


thread
thread는 그래프 실행 상태를 묶는 고유 식별자(ID)
thread_id는 이 그래프 실행의 기억이 저장되는 대화방/세션 키
"""

#----------------------------------------
# 🏋️ 기가차드 AI 상담사 (GigaChad Counselor)
# - 냉철하고 직설적인 조언
# - 과거 고민 이력 추적
# - 반복 고민 시 더 강한 피드백
# - 모든 모델을 gemini-2.5-flash로 통일
#----------------------------------------

#----------------------------------------
#Checkpoints (스레드(thread)의 특정 시점 상태를 체크포인트(checkpoint))
"""
config: 해당 체크포인트와 연관된 설정(config)
metadata: 체크포인트에 대한 메타데이터
values: 해당 시점의 state 값
next: 다음에 실행될 노드
tasks: 다음에 실행될 작업 정보
"""
#----------------------------------------

from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}

workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# Checkpointer 설정
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 초기 실행
config: RunnableConfig = {"configurable": {"thread_id": "1"}}
print("--- [초기 실행 시작] ---")
result=graph.invoke({"foo": "", "bar": []}, config)
print("초기 실행 완료\n")
print(result) # {'foo': 'b', 'bar': ['a', 'b']}


##1. Get state (상태 조회)##
config = {"configurable": {"thread_id": "1"}} #{"thread_id": "1", "checkpoint_id": } 특정 체크포인트 조회 가능
latest_state = graph.get_state(config)
print("\n## 최신 상태 조회 ##")
print(latest_state)

##2. Get state history (상태 히스토리 조회)##
print("\n## 상태 히스토리 조회 ##")
config = {"configurable": {"thread_id": "1"}}
history = list(graph.get_state_history(config))
print(history) # 내림차순, checkpoint_id는 매번 변경 됨

##3. Replay (리플레이)##
config = {"configurable": {"thread_id": "1", "checkpoint_id": history[1].config['configurable']['checkpoint_id']}}
graph.invoke(None, config=config)
print("\n## 리플레이 결과 ##")
print(graph.get_state(config))


##4. Update state (상태 수정)##
"""
config
thread_id: 업데이트할 스레드
checkpoint_id(선택): 해당 체크포인트에서 상태를 포크(fork)
"""

"""
values
업데이트할 state 값
노드가 state를 업데이트하는 것과 동일하게 처리
reducer 규칙에 따라 처리됨
"""

config = {"configurable": {"thread_id": "1"}}
graph.update_state(config, {"foo": "2", "bar": ["b"]})
print("\n## Update State 결과 ##")
print(graph.get_state(config))

"""
as_node
update_state를 호출할 때 선택적으로 as_node를 지정, 선택 없으면 최신으로
"""
# 특정 노드가 업데이트한 것처럼 속여서 다음 실행 흐름을 제어할 수 있음
print("\n## 4-1. Update State with as_node ##")
graph.update_state(config, {"foo": "forked_foo", "bar": ["hello world"]}, as_node="node_a")
forked_state = graph.get_state(config)
print(f"as_node='node_a' 업데이트 후 다음에 실행될 노드(next): {forked_state.next}")

print("\n## 재개 실행 후 상태 ##")
graph.invoke(None, config={"configurable": {"thread_id": "1"}})
print(graph.get_state({"configurable": {"thread_id": "1"}}))

#----------------------------------------
#Memory store
"""
Store는 “스레드를 넘어 공유되는 사용자/전역 기억”을 담당
"""
#----------------------------------------

##1. Basic Usage##
import uuid
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()

user_id = "1"
namespace_for_memory = (user_id, "memories") #1번 사용자 memories 폴더
print("\n## 1. Basic Usage ##")
print(f"네임스페이스: {namespace_for_memory}")

memory_id = str(uuid.uuid4())
memory = {"음식선호도" : "피자 좋아"}
in_memory_store.put(namespace_for_memory, memory_id, memory) #저장 put

memories = in_memory_store.search(namespace_for_memory) #조회 search
print(f"메모리에 담긴 데이터: {memories[-1].dict()}")


##2. Semantic Search (임베딩 없이 사용)##
from dotenv import load_dotenv
import os

# .env 파일 로드 (상위 디렉토리)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# 임베딩 없이 InMemoryStore 사용 (API 할당량 이슈 회피)
store = InMemoryStore()

print("\n## 2. Semantic Search ##")

#(1) 아직 메모리 없을 때 검색
memories = store.search(
    namespace_for_memory,
    limit=3  # Return top 3 matches
)
print("초기 검색 결과:", [m.value for m in memories])

# 특정 필드만 임베딩
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {
        "음식선호도": "나는 이탈리아 음식을 좋아해",
        "context": "저녁에 관한 계획 논의하기"
    },
    index=["음식선호도"] # Only embed "음식선호도" field
)

# 임베딩 없이 저장 (조회는 가능, 시맨틱 검색은 불가)
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {
        "취미": "스킨스쿠버 다이빙이 취미야",
        "context": "저녁에 관한 계획 논의하기"
    },
	index=False
)

store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {
        "음식선호도": "나는 고수가 싫어",
        "context": "점심에 관한 계획 논의하기"
    }
)

#(2) 메모리 추가 후 검색
memories = store.search(
    namespace_for_memory,
    limit=3  # Return top 3 matches
)
print("최종 검색 결과:", [(m.value, m.score) for m in memories])


##3. Using in LangGraph##
print("\n## 3. LangGraph에서 Memory Store 사용하기 ##")

import uuid
import operator
from typing_extensions import TypedDict, Annotated

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.9,  # 기가차드 특유의 강렬한 답변을 위해 높게 설정
)

# thread_id 기준 대화 상태 저장
checkpointer = InMemorySaver()

# user_id 기준 장기 기억 저장
store = InMemoryStore()

class State(TypedDict):
    # messages는 노드 실행 결과가 누적됨
    messages: Annotated[list[dict], operator.add]

# 고민 저장 노드
def update_memory(state: State, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "concerns")  # 고민 저장소

    # 가장 최근 사용자 고민
    last_user_message = state["messages"][-1]["content"]

    memory_id = str(uuid.uuid4())
    store.put(
        namespace,
        memory_id,
        {"concern": last_user_message},
    )

    return {}

# 🏋️ 기가차드 상담 노드
def call_model(state: State, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "concerns")

    user_text = state["messages"][-1]["content"]

    # 과거 비슷한 고민 검색 (임베딩 없이)
    past_concerns = store.search(
        namespace,
        limit=5,
    )

    # 반복 패턴 분석
    repeat_count = len(past_concerns)
    past_text = "\n".join([f"- {c.value['concern']}" for c in past_concerns]) if past_concerns else "(첫 상담)"

    # 🏋️ 기가차드 페르소나 시스템 프롬프트
    system = SystemMessage(
        content=(
            "너는 '기가차드' 스타일의 AI 상담사야.\n\n"
            "## 캐릭터 특징:\n"
            "- 냉철하고 직설적이며, 감정에 휘둘리지 않음\n"
            "- 현실적이고 실용적인 조언\n"
            "- 변명이나 자기 합리화를 용납하지 않음\n"
            "- '해야 한다'가 아니라 '할 수 있다'는 자신감 주입\n"
            "- 간결하고 임팩트 있는 문장\n"
            "- 가끔 💪🔥😎 이모지 사용\n"
            "- 사용자를 처음 부를 때 Oh, My sweet (사용자 이름) 삣삐! 이라 상냥하게 부르기\n"
            "- 사용자가 잘못되거나 부정적인 생각을 가질 때 You Puking Stop, (사용자 이름) 삣삐! 라고 다그치기\n\n"
            "## 상담 원칙:\n"
            "1. 문제의 본질을 꿰뚫어 봄\n"
            "2. 실행 가능한 구체적 행동 제시\n"
            "3. 자책보다는 앞으로 나아갈 방향 제시\n"
            "4. 필요시 따끔한 일침\n\n"
            f"## 사용자 정보:\n"
            f"- 반복 상담 횟수: {repeat_count}회\n"
            f"- 과거 고민들:\n{past_text}\n\n"
            "## 응답 전략:\n"
            "- 1회: 공감 + 명확한 해결책\n"
            "- 2-3회: 패턴 지적 + 강한 동기부여\n"
            "- 4회 이상: 직설적 피드백 + 최후통첩식 조언\n\n"
            "반드시 한국어로 답변하고, 300자 이내로 간결하게."
        )
    )
    human = HumanMessage(content=user_text)

    ai_msg = llm.invoke([system, human])

    return {
        "messages": [
            {
                "role": "assistant",
                "content": ai_msg.content,
            }
        ]
    }

# 그래프 구성
graph = StateGraph(State)

graph.add_node("update_memory", update_memory)
graph.add_node("call_model", call_model)

graph.add_edge(START, "update_memory")
graph.add_edge("update_memory", "call_model")
graph.add_edge("call_model", END)

graph = graph.compile(
    checkpointer=checkpointer,
    store=store,
)


# 🏋️ 기가차드 상담사 실행 예시
config = {
    "configurable": {
        "thread_id": "1",
        "user_id": "user_1",
    }
}

print("\n" + "="*60)
print("🏋️ 기가차드 AI 상담사 시작")
print("="*60)

# 고민 시나리오
concerns = [
    "자꾸 미루게 돼요. 어떻게 하죠?",
    "운동을 시작하고 싶은데 의지가 안 생겨요",
    "또 미루게 됐어요... 저 안 변하나봐요",
]

for i, concern in enumerate(concerns, 1):
    print(f"\n--- [{i}번째 상담] ---")
    print(f"😢 사용자: {concern}")
    print("-" * 40)

    for update in graph.stream(
        {"messages": [{"role": "user", "content": concern}]},
        config,
        stream_mode="updates",
    ):
        if "call_model" in update:
            response = update["call_model"]["messages"][-1]["content"]
            print(f"🏋️ 기가차드: {response}")

print("\n" + "="*60)
print("상담 종료")
print("="*60)


"""Apendix

- langgraph.json 설정은 “배포/서버 환경에서 store 인덱스 켜기”
"store": {
  "index": {
    "embed": "openai:text-embeddings-3-small",
    "dims": 1536,
    "fields": ["$"]
  }

# 시스템적으로 어떻게 보관·보호·운영할까
 - langgraph-checkpoint vs langgraph-checkpoint-sqlite vs langgraph-checkpoint-postgres: '어디에 저장하냐' 차이
 - Serializer: 'state를 저장 가능하게 변환'하는 규칙
 - pickle_fallback: 변환 실패 시 pickle로 구제
 - EncryptedSerializer: 저장되는 내용 자체를 암호화
 """