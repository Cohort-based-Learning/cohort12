"""
체크포인터를 붙여 컴파일하면, 그래프 상태가 매 super-step마다 자동 저장(체크포인트) 된다.
저장된 체크포인트는 thread(스레드) 에 쌓이며, 실행이 끝난 뒤에도 상태 조회/재개/분기가 가능해진다.
그 결과로 HITL(중간에 사람 개입), 대화/세션 메모리, 과거 시점 재생(타임 트래블), 실패 지점부터 복구(장애 내성) 같은 기능이 열린다.
Agent Server를 쓰면 이런 저장/관리(체크포인팅)를 서버가 자동으로 해줘서 직접 설정할 필요가 없다


thread
thread는 그래프 실행 상태를 묶는 고유 식별자(ID)
thread_id는 이 그래프 실행의 기억이 저장되는 대화방/세션 키
"""
# 영역		           주요 메서드		             역할
# Checkpointer	    get_state		        thread 최신 state 읽기
#                   get_state_history	    checkpoint 이력 읽기
#                   update_state		    state 수정 / fork
# Store		        put			            메모리 저장
#                   search			        메모리 검색

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

# [설명] 상태(State) 정의
# 그래프 내에서 관리될 데이터 구조입니다.
# foo: 단순 문자열로, 노드가 실행될 때마다 새로운 값으로 덮어쓰기 됩니다.
# bar: 리스트 형태로, Annotated와 add 연산자가 있어 값이 덮어써지지 않고 계속 추가(append)됩니다.
class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

# [설명] 노드 정의
# node_a: foo를 'a'로 바꾸고, bar 리스트에 'a'를 추가합니다.
def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

# node_b: foo를 'b'로 바꾸고, bar 리스트에 'b'를 추가합니다.
def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}

# [설명] 워크플로우(그래프) 구성
# 시작 -> node_a -> node_b -> 끝 순서로 실행되도록 연결합니다.
workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# Checkpointer 설정
# InMemorySaver는 DB가 아닌 메모리(RAM)에 체크포인트를 저장합니다. (실습용)
checkpointer = InMemorySaver()

# [설명] 그래프 컴파일
# checkpointer를 인자로 전달하여, 실행될 때마다 상태가 저장되도록 설정합니다.
graph = workflow.compile(checkpointer=checkpointer)

# 초기 실행
# thread_id: "1"을 지정하여 이 실행 흐름의 고유 ID를 부여합니다.
config: RunnableConfig = {"configurable": {"thread_id": "1"}}
print("--- [초기 실행 시작] ---")

# [설명] invoke 실행 흐름
# 1. 입력: {"foo": "", "bar": []}
# 2. node_a 실행: foo="a", bar=["a"]
# 3. node_b 실행: foo="b", bar=["a", "b"] (bar는 add 연산자 때문에 누적됨)
# 실행 후 모든 단계의 상태가 'thread_id=1'에 저장됩니다.
result=graph.invoke({"foo": "", "bar": []}, config)
print("초기 실행 완료\n")
print(result) # {'foo': 'b', 'bar': ['a', 'b']}


##1. Get state (상태 조회)##
# [설명] 현재 상태 조회
# thread_id="1"인 스레드의 가장 최신 상태(State)를 가져옵니다.
config = {"configurable": {"thread_id": "1"}} #{"thread_id": "1", "checkpoint_id": } 특정 체크포인트 조회 가능
latest_state = graph.get_state(config)
print("\n## 최신 상태 조회 ##")
print(latest_state)

##2. Get state history (상태 히스토리 조회)##
# [설명] 과거 기록 조회
# 해당 스레드에서 발생한 모든 상태 변경 이력을 역순(최신순)으로 가져옵니다.
print("\n## 상태 히스토리 조회 ##")
config = {"configurable": {"thread_id": "1"}}
history = list(graph.get_state_history(config))
print(history) # 내림차순, checkpoint_id는 매번 변경 됨

##3. Replay (리플레이)##
# [설명] 타임 트래블 (과거로 되돌아가기)
# history[1]은 'node_a' 실행 직후의 시점입니다.
# 그 시점의 checkpoint_id를 config에 넣고 invoke(None)을 호출하면,
# 그 시점부터 다시 실행을 재개합니다. (즉, node_b가 다시 실행됨)
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

# [설명] 상태 강제 수정
# 현재 상태에서 foo를 "2"로, bar에 "b"를 추가하도록 상태를 직접 수정합니다.
config = {"configurable": {"thread_id": "1"}}
graph.update_state(config, {"foo": "2", "bar": ["b"]})
print("\n## Update State 결과 ##")
print(graph.get_state(config))

"""
as_node
update_state를 호출할 때 선택적으로 as_node를 지정, 선택 없으면 최신으로
"""
# 특정 노드가 업데이트한 것처럼 속여서 다음 실행 흐름을 제어할 수 있음
# [설명] 분기(Fork) 만들기
# 마치 "node_a"가 실행된 직후인 것처럼 위장하여 상태를 변경합니다.
# 이렇게 하면 다음 실행 순서는 node_a의 다음인 'node_b'가 됩니다.
print("\n## 4-1. Update State with as_node ##")
graph.update_state(config, {"foo": "forked_foo", "bar": ["hello world"]}, as_node="node_a")
forked_state = graph.get_state(config)
print(f"as_node='node_a' 업데이트 후 다음에 실행될 노드(next): {forked_state.next}")

print("\n## 재개 실행 후 상태 ##")
# 변경된 상태(foo="forked_foo")에서 실행을 재개합니다. node_b가 실행되어 foo는 "b"가 됩니다.
# 하지만 bar에는 "hello world"가 남아있게 됩니다.
graph.invoke(None, config={"configurable": {"thread_id": "1"}})
print(graph.get_state({"configurable": {"thread_id": "1"}}))

#----------------------------------------
#Memory store
"""
Store는 “스레드를 넘어 공유되는 사용자/전역 기억”을 담당
"""
#----------------------------------------

##1. Basic Usage##
# [설명] 메모리 스토어
# 체크포인트와 달리, 스레드가 없어도 정보를 영구적으로 저장하는 공간입니다.
import uuid
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()

user_id = "1"
# namespace: 정보를 저장할 경로(폴더)라고 생각하면 됩니다.
namespace_for_memory = (user_id, "memories") #1번 사용자 memories 폴더
print("\n## 1. Basic Usage ##")
print(f"네임스페이스: {namespace_for_memory}")

memory_id = str(uuid.uuid4())
memory = {"음식선호도" : "피자 좋아"}
in_memory_store.put(namespace_for_memory, memory_id, memory) #저장 put

memories = in_memory_store.search(namespace_for_memory) #조회 search
print(f"메모리에 담긴 데이터: {memories[-1].dict()}")


##2. Semantic Search##
from dotenv import load_dotenv
load_dotenv()

from langchain.embeddings import init_embeddings

# [설명] 시맨틱 검색(의미 기반 검색) 설정
# index 설정을 통해 저장되는 텍스트를 벡터로 변환(임베딩)하여 저장합니다.
# 이렇게 하면 "피자"라고 정확히 검색하지 않고 "좋아하는 음식"이라고 물어도 찾을 수 있습니다.
store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),  # Embedding provider
        "dims": 1536,                              # Embedding dimensions
        "fields": ["음식선호도", "$"] #유저에 대한 모든 문맥을 기억하려면 "$" / 특정 필드만 임베딩할 수 도 있음
    }
)

print("\n## 2. Semantic Search ##")

#(1) 아직 메모리 없을 때 검색
memories = store.search(
    namespace_for_memory,
    query="유저가 좋아한다고 말한 음식은?",
    limit=3  # Return top 3 matches
)
print("초기 시맨틱 검색 결과:", [m.value for m in memories])

# 특정 필드만 임베딩
# [설명] 데이터 저장
# 사용자의 대화 맥락, 취미, 음식 선호도 등을 저장합니다.
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
# "좋아한다고 말한 음식"과 의미상 가장 가까운 "이탈리아 음식 좋아해"가 검색됩니다.
memories = store.search(
    namespace_for_memory,
    query="유저가 좋아한다고 말한 음식은?",
    limit=3  # Return top 3 matches
)
print("최종 시맨틱 검색 결과:", [(m.value, m.score) for m in memories])


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
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    "gpt-4o-mini",
    temperature=0.2,
)

# thread_id 기준 대화 상태 저장
checkpointer = InMemorySaver()

# user_id 기준 장기 기억 저장
store = InMemoryStore()

class State(TypedDict):
    # messages는 노드 실행 결과가 누적됨
    messages: Annotated[list[dict], operator.add]

# Memory 저장 노드
# [설명] update_memory 함수
# 사용자의 마지막 발화를 가져와서 Memory Store에 저장하는 역할을 합니다.
def update_memory(state: State, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")

    # 가장 최근 사용자 발화
    last_user_message = state["messages"][-1]["content"]

    memory_id = str(uuid.uuid4())
    store.put(
        namespace,
        memory_id,
        {"memory": last_user_message},
    )

    return {}

# Model 호출 노드 (memory 검색)
# [설명] call_model 함수
# 1. 사용자의 현재 질문과 관련된 기억을 Store에서 검색합니다.
# 2. 검색된 기억을 System Prompt에 포함시켜 LLM에게 전달합니다.
# 3. LLM은 이 기억을 참고하여 사용자에게 맞춤형 답변을 합니다.
def call_model(state: State, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")

    user_text = state["messages"][-1]["content"]

    # 관련 메모리 검색
    memories = store.search(
        namespace,
        query=user_text,
        limit=3,
    )

    memory_text = "\n".join(m.value["memory"] for m in memories) if memories else "(저장된 기억 없음)"

    # 메모리를 “컨텍스트”로 넣고, OpenAI가 답하도록 함
    system = SystemMessage(
        content=(
            "너는 사용자와 대화하는 비서야.\n"
            "아래 [기억]은 이 사용자에 대해 저장된 정보야. 답변에 참고해.\n"
            "만약 기억이 현재 질문과 관련 없으면, 억지로 끼워 맞추지 말고 무시해.\n\n"
            f"[기억]\n{memory_text}"
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

# [설명] 그래프 컴파일 (중요)
# checkpointer: 대화 맥락(History) 유지용
# store: 장기 기억(Memory) 저장용
# 이 두 가지를 모두 연결해야 봇이 '기억'을 가질 수 있습니다.
graph = graph.compile(
    checkpointer=checkpointer,
    store=store,
)


# 결과 보기
config = {
    "configurable": {
        "thread_id": "1",
        "user_id": "user_1",
    }
}

print(f"{config['configurable']} 유저는 피자를 좋아한다고 말함\n")

# [설명] 첫 번째 실행
# 유저가 "피자 좋아해"라고 말하면, update_memory 노드가 이를 기억 저장소에 저장합니다.
for update in graph.stream(
    {"messages": [{"role": "user", "content": "나는 피자를 좋아해"}]},
    config,
    stream_mode="updates",
):
    print(update)

config2 = {"configurable": {"thread_id": "2", "user_id": "user_1"}}

# [설명] 두 번째 실행 (다른 스레드)
# thread_id가 '2'로 바뀌었으므로, 이전 대화 맥락(단기 기억)은 사라집니다.
# 하지만 user_id가 'user_1'로 같으므로, Store에 저장된 "피자 좋아해"라는 기억(장기 기억)은 검색됩니다.
for update in graph.stream(
    {"messages": [{"role": "user", "content": "내 기억 뭐야?"}]},
    config2,
    stream_mode="updates",
):
    print(update)

config3 = {"configurable": {"thread_id": "2", "user_id": "user_2"}} #다른 유저에는 기억이 없는 것 확인

# [설명] 세 번째 실행 (다른 유저)
# user_id가 'user_2'입니다. user_1의 기억 공간(namespace)과는 다르므로 기억이 검색되지 않습니다.
for update in graph.stream(
    {"messages": [{"role": "user", "content": "내 기억 뭐야?"}]},
    config3,
    stream_mode="updates",
):
    print(update)


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