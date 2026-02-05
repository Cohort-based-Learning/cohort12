import os
from dotenv import load_dotenv
load_dotenv()

# =========================================================
# 기본 설정
# =========================================================
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.mongodb import MongoDBSaver

# MongoDB 연결
MongoDB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MongoDB_URI)
DB = client["brickers"]
collection = DB["ldraw_parts"]

# MongoDB Checkpointer (장기 메모리 역할 - 세션 꺼도 유지됨)
checkpointer = MongoDBSaver(client, db_name="brickers")

# AI 모델
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

"""
단기 메모리(Short-term): 같은 thread 안에서만 유지되는 기억
- 대화 히스토리, 현재 세션의 맥락

장기 메모리(Long-term): thread가 달라도 유지되는 기억
- 유저 프로필, 선호도, 설정 (MongoDB에 저장)

메모리 관리 전략:
1. Trim - 토큰 수 제한해서 자르기
2. Delete - 오래된 메시지 삭제
3. Summarize - 요약해서 저장

"""


# =========================================================
# 1번 예제: Trim - 메모리 잘라서 관리하기
# (책가방에 책 10권 못 넣으니까 5권만 넣는 거랑 같음)
# =========================================================
print("\n" + "="*50)
print("1. Trim 예제 - 토큰 제한으로 메모리 관리")
print("="*50)

from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.graph import StateGraph, START, MessagesState

def call_model_with_trim(state: MessagesState):
    """
    메모리를 128 토큰 이하로 유지하면서 대화하기
    오래된 대화는 자동으로 잘려나감
    """
    # 메시지 자르기 - 최대 128 토큰만 유지
    trimmed = trim_messages(
        state["messages"],
        strategy="last",                    # 최근 메시지 유지
        token_counter=count_tokens_approximately,  # 빠른 토큰 계산
        max_tokens=128,                     # 최대 128 토큰
        start_on="human",                   # HumanMessage부터 시작
        end_on=("human", "tool"),           # human이나 tool로 끝남
    )

    print(f"  [Trim] 원본 {len(state['messages'])}개 → 자른 후 {len(trimmed)}개")

    response = llm.invoke(trimmed)
    return {"messages": [response]}

# 그래프 만들기
builder = StateGraph(MessagesState)
builder.add_node("chat", call_model_with_trim)
builder.add_edge(START, "chat")

trim_graph = builder.compile(checkpointer=checkpointer)

# 실행
config = {"configurable": {"thread_id": "trim-demo-001"}}

print("\n[1] 이름 알려주기")
r1 = trim_graph.invoke({"messages": "안녕! 내 이름은 브릭몬이야."}, config)
print(f"AI: {r1['messages'][-1].content[:100]}...")

print("\n[2] 시 요청하기")
r2 = trim_graph.invoke({"messages": "레고에 대한 짧은 시를 써줘."}, config)
print(f"AI: {r2['messages'][-1].content[:100]}...")

print("\n[3] 변형 요청하기")
r3 = trim_graph.invoke({"messages": "이번엔 브릭으로 해줘."}, config)
print(f"AI: {r3['messages'][-1].content[:100]}...")

print("\n[4] 이름 물어보기 (Trim 때문에 까먹을 수 있음!)")
r4 = trim_graph.invoke({"messages": "내 이름이 뭐였어?"}, config)
print(f"AI: {r4['messages'][-1].content}")


# =========================================================
# 2번 예제: Delete - 오래된 메시지 삭제하기
# (카톡 용량 꽉 차서 오래된 거 지우는 거랑 같음)
# =========================================================
print("\n" + "="*50)
print("2. Delete 예제 - 오래된 메시지 삭제")
print("="*50)

from langchain.messages import RemoveMessage

def chat_node(state: MessagesState):
    """일반 대화 노드"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def delete_old_messages(state: MessagesState):
    """
    메시지가 4개 넘으면 가장 오래된 2개 삭제
    채팅방 용량 관리하는 거랑 같음
    """
    messages = state["messages"]

    if len(messages) > 4:
        print(f"  [Delete] 메시지 {len(messages)}개 → 오래된 2개 삭제!")
        # 가장 오래된 2개 메시지 삭제
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}

    print(f"  [Delete] 메시지 {len(messages)}개 - 삭제 안 함")
    return None

# 그래프: 대화 → 정리
builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_node("cleanup", delete_old_messages)
builder.add_edge(START, "chat")
builder.add_edge("chat", "cleanup")

delete_graph = builder.compile(checkpointer=checkpointer)

# 실행
config = {"configurable": {"thread_id": "delete-demo-001"}}

print("\n[1] 첫 대화")
for event in delete_graph.stream(
    {"messages": [{"role": "user", "content": "안녕! 난 레고 수집가야."}]},
    config, stream_mode="values"
):
    if event["messages"]:
        print(f"  메시지 수: {len(event['messages'])}")

print("\n[2] 두 번째 대화")
for event in delete_graph.stream(
    {"messages": [{"role": "user", "content": "가장 좋아하는 레고 시리즈가 뭐야?"}]},
    config, stream_mode="values"
):
    if event["messages"]:
        print(f"  메시지 수: {len(event['messages'])}")

print("\n[3] 세 번째 대화 (이제 삭제 시작!)")
for event in delete_graph.stream(
    {"messages": [{"role": "user", "content": "테크닉 시리즈는 어때?"}]},
    config, stream_mode="values"
):
    if event["messages"]:
        print(f"  메시지 수: {len(event['messages'])}")


# =========================================================
# 3번 예제: Summarize - 대화 요약해서 저장하기
# (수업 끝나고 필기 정리하는 거랑 같음)
# =========================================================
print("\n" + "="*50)
print("3. Summarize 예제 - 대화 요약으로 메모리 관리")
print("="*50)

from typing import TypedDict
from langchain.messages import HumanMessage, RemoveMessage

class SummaryState(MessagesState):
    summary: str  # 지금까지 대화 요약

def summarize_conversation(state: SummaryState):
    """
    대화 내용을 요약하고, 최근 2개만 남기기
    수업 필기 정리하듯이!
    """
    summary = state.get("summary", "")

    # 요약 프롬프트 만들기
    if summary:
        summary_prompt = (
            f"지금까지 대화 요약:\n{summary}\n\n"
            "위 요약에 새로운 대화 내용을 추가해서 확장해줘:"
        )
    else:
        summary_prompt = "위 대화를 한국어로 간단히 요약해줘:"

    # AI한테 요약 시키기
    messages = state["messages"] + [HumanMessage(content=summary_prompt)]
    response = llm.invoke(messages)

    # 최근 2개 메시지만 남기고 나머지 삭제
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    print(f"  [Summarize] 요약 완료! 메시지 {len(state['messages'])}개 → 2개 유지")

    return {
        "summary": response.content,
        "messages": delete_messages,
    }

def chat_with_summary(state: SummaryState):
    """요약을 참고해서 대화하기"""
    messages = state["messages"]

    # 요약이 있으면 맨 앞에 추가
    if state.get("summary"):
        messages = [
            HumanMessage(content=f"(이전 대화 요약)\n{state['summary']}")
        ] + messages
        print(f"  [Chat] 요약 참고해서 대화 중...")

    response = llm.invoke(messages)
    return {"messages": [response]}

# 그래프: 대화 → 요약
builder = StateGraph(SummaryState)
builder.add_node("chat", chat_with_summary)
builder.add_node("summarize", summarize_conversation)
builder.add_edge(START, "chat")
builder.add_edge("chat", "summarize")

summary_graph = builder.compile(checkpointer=checkpointer)

# 실행
config = {"configurable": {"thread_id": "summary-demo-001"}}

print("\n[1] 이름 알려주기")
summary_graph.invoke({"messages": "안녕! 내 이름은 브릭마스터야."}, config)

print("\n[2] 취미 말하기")
summary_graph.invoke({"messages": "나는 레고 테크닉을 좋아해."}, config)

print("\n[3] 컬렉션 자랑하기")
summary_graph.invoke({"messages": "람보르기니 시안이 내 최애 세트야."}, config)

print("\n[4] 이름 물어보기 (요약에 있어서 기억함!)")
final = summary_graph.invoke({"messages": "내 이름이랑 좋아하는 거 기억해?"}, config)

print(f"\nAI 답변: {final['messages'][-1].content}")
print(f"\n저장된 요약:\n{final.get('summary', '없음')}")


# =========================================================
# 4번 예제: 부품 추천 봇 (메모리 활용)
# MongoDB에서 부품 찾고, 대화 기록 요약해서 관리
# =========================================================
print("\n" + "="*50)
print("4. 부품 추천 봇 - 실전 메모리 활용")
print("="*50)

class PartBotState(MessagesState):
    summary: str           # 대화 요약
    favorite_parts: list   # 좋아하는 부품 목록 (장기 기억)

def search_parts(query: str) -> list:
    """MongoDB에서 부품 검색"""
    results = list(collection.find(
        {"$or": [
            {"name": {"$regex": query, "$options": "i"}},
            {"keywords": {"$regex": query, "$options": "i"}},
        ]},
        {"_id": 0, "name": 1, "partId": 1, "category": 1}
    ).limit(3))
    return results

def part_bot_chat(state: PartBotState):
    """부품 추천 봇 - 요약과 선호도 활용"""
    messages = state["messages"]
    user_msg = messages[-1].content if messages else ""

    # 시스템 컨텍스트 구성
    context_parts = []

    if state.get("summary"):
        context_parts.append(f"[이전 대화 요약]\n{state['summary']}")

    if state.get("favorite_parts"):
        fav_list = ", ".join(state["favorite_parts"])
        context_parts.append(f"[사용자가 좋아하는 부품]\n{fav_list}")

    # 부품 관련 질문이면 검색
    if any(kw in user_msg for kw in ["부품", "브릭", "레고", "찾아", "추천"]):
        parts = search_parts(user_msg)
        if parts:
            parts_info = "\n".join([f"- {p['name']} (ID: {p['partId']})" for p in parts])
            context_parts.append(f"[검색된 부품]\n{parts_info}")

    # 컨텍스트 + 메시지
    if context_parts:
        context_msg = HumanMessage(content="\n\n".join(context_parts))
        messages = [context_msg] + messages

    response = llm.invoke(messages)
    return {"messages": [response]}

def part_bot_summarize(state: PartBotState):
    """대화 요약 + 선호도 추출"""
    summary = state.get("summary", "")

    # 요약 생성
    if len(state["messages"]) > 4:
        summary_prompt = f"지금까지 요약: {summary}\n\n위 대화를 간단히 요약해줘:"
        messages = state["messages"] + [HumanMessage(content=summary_prompt)]
        response = llm.invoke(messages)
        new_summary = response.content

        # 오래된 메시지 삭제
        delete_msgs = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

        print(f"  [PartBot] 요약 완료!")
        return {"summary": new_summary, "messages": delete_msgs}

    return None

# 그래프 구성
builder = StateGraph(PartBotState)
builder.add_node("chat", part_bot_chat)
builder.add_node("summarize", part_bot_summarize)
builder.add_edge(START, "chat")
builder.add_edge("chat", "summarize")

part_bot = builder.compile(checkpointer=checkpointer)

# 실행
config = {"configurable": {"thread_id": "partbot-demo-001"}}

print("\n[대화 시작]")
r = part_bot.invoke({
    "messages": "안녕! 나는 브릭을 좋아해. brick 부품 추천해줘!",
    "favorite_parts": ["3001 Brick 2x4", "3003 Brick 2x2"]
}, config)
print(f"AI: {r['messages'][-1].content[:200]}...")

print("\n[이어서 대화]")
r = part_bot.invoke({"messages": "plate 부품도 보여줘"}, config)
print(f"AI: {r['messages'][-1].content[:200]}...")


print("\n" + "="*50)
print("실습 완료!")
print("="*50)
print("""
정리:
1. Trim - 토큰 제한으로 자르기 (빠름, 정보 손실)
2. Delete - 오래된 메시지 삭제 (단순, 정보 손실)
3. Summarize - 요약 저장 (정보 보존, 느림)

실무 팁:
- 단순 챗봇 → Trim이나 Delete
- 중요한 정보 유지 필요 → Summarize
- 사용자 선호도 등 → MongoDB에 별도 저장 (장기 메모리)
""")
