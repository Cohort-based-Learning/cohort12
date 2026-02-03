import os
import uuid
import operator
from typing import Annotated, TypedDict

from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 라이브러리 임포트
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

# 1. 시맨틱 검색을 위한 로컬 임베딩 모델 설정
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# [수정] 임베딩 호환성 래퍼 (단일 쿼리 vs 문서 리스트)
def safe_embed(input_data):
    if isinstance(input_data, list):
        return embeddings.embed_documents(input_data)
    else:
        return embeddings.embed_query(input_data)

# 2. 상태(State) 정의
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# 3. 노드 정의
def update_memory_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"].get("user_id", "default_user")
    namespace = (user_id, "memories")
    
    # [수정 1] 마지막 메시지에서 안전하게 텍스트 추출 (방어 로직)
    last_msg = state["messages"][-1]
    if isinstance(last_msg, dict):
        last_message_text = last_msg.get("content", "")
    else:
        last_message_text = getattr(last_msg, "content", "")
    
    # [수정] content가 None일 경우 빈 문자열로 변환 (NoneType 에러 방지)
    if last_message_text is None:
        last_message_text = ""
    
    if any(keyword in last_message_text for keyword in ["좋아", "싫어", "취미", "못해"]):
        memory_id = str(uuid.uuid4())
        store.put(namespace, memory_id, {"memory": last_message_text})
        print(f"\n[System] 장기 기억 저장 완료: {last_message_text}")
    
    return {}

def assistant_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"].get("user_id", "default_user")
    
    # [수정 2] 마지막 메시지에서 안전하게 텍스트 추출
    last_msg = state["messages"][-1]
    val = last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", "")
    user_input = val if val is not None else ""
    
    # 시맨틱 검색
    memories = store.search((user_id, "memories"), query=user_input, limit=2)
    memory_context = "\n".join([f"- {m.value['memory']}" for m in memories])
    
    # 모델 설정 (기존 유지)
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    
    system_prompt = f"""너는 사용자의 과거를 기억하는 유능한 비서야. 
아래 [기억] 정보를 참고해서 답변해줘.

[기억]
{memory_context if memory_context else "아직 저장된 기억이 없음"}
"""
    
    # [수정 3] 메시지 객체 변환 로직 최적화 및 안전성 확보
    formatted_messages = [SystemMessage(content=system_prompt)]
    for msg in state["messages"]:
        # 딕셔너리와 객체 모두 대응
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        role = msg.get("role", "") if isinstance(msg, dict) else ("user" if isinstance(msg, HumanMessage) else "assistant")
        
        if role == "user":
            formatted_messages.append(HumanMessage(content=content))
        else:
            formatted_messages.append(AIMessage(content=content))
            
    response = llm.invoke(formatted_messages)
    # AIMessage 객체 리스트 반환 (operator.add와 호환)
    return {"messages": [response]}

# 4. 그래프 구성 (기존 유지)
builder = StateGraph(ChatState)
builder.add_node("memory_node", update_memory_node)
builder.add_node("assistant", assistant_node)
builder.add_edge(START, "memory_node")
builder.add_edge("memory_node", "assistant")
builder.add_edge("assistant", END)

# 5. 설정 (기존 유지)
checkpointer = InMemorySaver()
store = InMemoryStore(
    index={
        "embed": safe_embed,  # [수정] 래퍼 함수 사용 
        "dims": 1024,
        "fields": ["memory"]
    }
)

app = builder.compile(checkpointer=checkpointer, store=store)

# ---------------------------------------------------------
# 🤖 2. 실행 및 결과 확인
# ---------------------------------------------------------
def run_and_display(user_input: str, thread_id: str, user_id: str):
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    print(f"\n[입력] User({user_id}): {user_input}")
    
    # [수정 4] 입력을 HumanMessage 객체로 전달하여 Type 호환성 확보
    initial_input = {"messages": [HumanMessage(content=user_input)]}
    
    for update in app.stream(initial_input, config, stream_mode="updates"):
        for node, value in update.items():
            if value and "messages" in value:
                # 결과 출력 시 .content 사용
                content = value["messages"][-1].content
                print(f"  └─ [{node}] 응답: {content}")

# --- 테스트 시나리오 시작 ---

# 1. 정보 주입 (Kim 유저)
run_and_display("안녕! 나는 민트초코를 아주 싫어해.", "thread_1", "user_kim")

# 2. 다른 세션에서 의미 기반 검색 확인 (Kim 유저)
# "싫어하는 것" -> "민트초코"를 의미적으로 찾아내야 함
run_and_display("내가 아주 극혐하는 음식이 뭐였지?", "thread_2", "user_kim")

# 3. 다른 유저의 독립성 확인 (Lee 유저)
run_and_display("내가 싫어하는 음식이 뭐야?", "thread_3", "user_lee")