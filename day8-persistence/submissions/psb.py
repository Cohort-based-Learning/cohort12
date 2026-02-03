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
#Checkpoints (스레드(thread)의 특정 시점 상태를 체크포인트(checkpoint))
"""
config: 해당 체크포인트와 연관된 설정(config)
metadata: 체크포인트에 대한 메타데이터
values: 해당 시점의 state 값
next: 다음에 실행될 노드
tasks: 다음에 실행될 작업 정보
"""
#----------------------------------------
import os
import uuid
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# DB 설정
MongoDB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MongoDB_URI)
DB = client["brickers"]
collection = DB["ldraw_parts"]

# 모델 설정
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)


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
print(result)


##1. Get state (상태 조회)##
config = {"configurable": {"thread_id": "1"}}
latest_state = graph.get_state(config)
print("\n## 최신 상태 조회 ##")
print(latest_state)

##2. Get state history (상태 히스토리 조회)##
print("\n## 상태 히스토리 조회 ##")
config = {"configurable": {"thread_id": "1"}}
history = list(graph.get_state_history(config))
print(history)

##3. Replay (리플레이)##
config = {"configurable": {"thread_id": "1", "checkpoint_id": history[1].config['configurable']['checkpoint_id']}}
graph.invoke(None, config=config)
print("\n## 리플레이 결과 ##")
print(graph.get_state(config))


##4. Update state (상태 수정)##
config = {"configurable": {"thread_id": "1"}}
graph.update_state(config, {"foo": "2", "bar": ["b"]})
print("\n## Update State 결과 ##")
print(graph.get_state(config))

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
Store는 "스레드를 넘어 공유되는 사용자/전역 기억"을 담당
"""
#----------------------------------------

##1. Basic Usage##
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()

user_id = "1"
namespace_for_memory = (user_id, "memories")
print("\n## 1. Basic Usage ##")
print(f"네임스페이스: {namespace_for_memory}")

memory_id = str(uuid.uuid4())
memory = {"음식선호도" : "피자 좋아"}
in_memory_store.put(namespace_for_memory, memory_id, memory)

memories = in_memory_store.search(namespace_for_memory)
print(f"메모리에 담긴 데이터: {memories[-1].dict()}")


##2. Semantic Search##
from langchain_google_genai import GoogleGenerativeAIEmbeddings

store = InMemoryStore(
    index={
        "embed": GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
        "dims": 768,
        "fields": ["음식선호도", "$"]
    }
)

print("\n## 2. Semantic Search ##")

memories = store.search(
    namespace_for_memory,
    query="유저가 좋아한다고 말한 음식은?",
    limit=3
)
print("초기 시맨틱 검색 결과:", [m.value for m in memories])

store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {
        "음식선호도": "나는 이탈리아 음식을 좋아해",
        "context": "저녁에 관한 계획 논의하기"
    },
    index=["음식선호도"]
)

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

memories = store.search(
    namespace_for_memory,
    query="유저가 좋아한다고 말한 음식은?",
    limit=3
)
print("최종 시맨틱 검색 결과:", [(m.value, m.score) for m in memories])


##3. Using in LangGraph (MongoDB 연동)##
print("\n## 3. LangGraph에서 Memory Store + MongoDB 사용하기 ##")

import operator
from langgraph.store.base import BaseStore
from langchain.messages import SystemMessage, HumanMessage

# thread_id 기준 대화 상태 저장
checkpointer = InMemorySaver()

# user_id 기준 장기 기억 저장
store = InMemoryStore()

class ChatState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    part_info: str  # MongoDB에서 가져온 부품 정보

# MongoDB 검색 노드
def search_db(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    user_text = state["messages"][-1]["content"]
    
    # MongoDB에서 부품 검색
    results = list(collection.find(
        {
            "$or": [
                {"name": {"$regex": user_text, "$options": "i"}},
                {"keywords": {"$regex": user_text, "$options": "i"}},
                {"partId": {"$regex": user_text, "$options": "i"}}
            ]
        },
        {"_id": 0, "name": 1, "partId": 1, "keywords": 1, "category": 1}
    ).limit(3))
    
    part_info = str(results) if results else "(부품 정보 없음)"
    print(f"[DB 검색] {part_info[:80]}...")
    
    return {"part_info": part_info}

# Memory 저장 노드
def update_memory(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")

    last_user_message = state["messages"][-1]["content"]

    memory_id = str(uuid.uuid4())
    store.put(
        namespace,
        memory_id,
        {"memory": last_user_message},
    )

    return {}

# Model 호출 노드
def call_model(state: ChatState, config: RunnableConfig, *, store: BaseStore):
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
    part_info = state.get("part_info", "(부품 정보 없음)")

    system = SystemMessage(
        content=(
            "너는 브릭/레고 부품 전문 상담사야.\n"
            "아래 [기억]은 이 사용자에 대해 저장된 정보야. 답변에 참고해.\n"
            "아래 [부품 정보]는 MongoDB에서 검색한 결과야. 관련 있으면 활용해.\n\n"
            f"[기억]\n{memory_text}\n\n"
            f"[부품 정보]\n{part_info}"
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
graph_builder = StateGraph(ChatState)

graph_builder.add_node("search_db", search_db)
graph_builder.add_node("update_memory", update_memory)
graph_builder.add_node("call_model", call_model)

graph_builder.add_edge(START, "search_db")
graph_builder.add_edge("search_db", "update_memory")
graph_builder.add_edge("update_memory", "call_model")
graph_builder.add_edge("call_model", END)

final_graph = graph_builder.compile(
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

print(f"\n{config['configurable']} 유저가 Brick 부품 질문\n")

for update in final_graph.stream(
    {"messages": [{"role": "user", "content": "Brick 2x4 부품 알려줘"}]},
    config,
    stream_mode="updates",
):
    print(update)

config2 = {"configurable": {"thread_id": "2", "user_id": "user_1"}}

print(f"\n같은 유저, 다른 스레드에서 기억 확인\n")

for update in final_graph.stream(
    {"messages": [{"role": "user", "content": "내가 뭐 물어봤었어?"}]},
    config2,
    stream_mode="updates",
):
    print(update)

config3 = {"configurable": {"thread_id": "3", "user_id": "user_2"}}

print(f"\n다른 유저는 기억 없음 확인\n")

for update in final_graph.stream(
    {"messages": [{"role": "user", "content": "내 기억 뭐야?"}]},
    config3,
    stream_mode="updates",
):
    print(update)


"""Apendix

- langgraph.json 설정은 "배포/서버 환경에서 store 인덱스 켜기"
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