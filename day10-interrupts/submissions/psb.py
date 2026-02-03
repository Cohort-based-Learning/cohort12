#pip install langgraph-checkpoint-mongodb

import os
from dotenv import load_dotenv
load_dotenv()  # .env 파일에서 비밀번호 같은 거 불러오기

# =========================================================
# 기본 설정 (맨날 쓰는 것들 세팅)
# =========================================================
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.mongodb import MongoDBSaver  # ★ MongoDB 전용 세이브 저장소!

# DB 설정 - MongoDB라는 창고에 연결하는 거야
MongoDB_URI = os.getenv("MONGODB_URI")  # 창고 주소 (비밀이라 .env에 숨겨둠)
client = MongoClient(MongoDB_URI)       # 창고 문 열기
DB = client["brickers"]                 # "brickers"라는 방으로 들어가기
collection = DB["ldraw_parts"]          # 그 방에 있는 "ldraw_parts" 서랍 열기

# ★ MongoDB Checkpointer 설정 ★
# SQLite는 내 컴퓨터에만 저장되는데, 이건 클라우드 MongoDB에 저장됨!
# 장점: 컴퓨터 꺼도 안 날아감, 다른 컴퓨터에서도 이어서 할 수 있음
checkpointer = MongoDBSaver(client, db_name="brickers")

# AI 모델 설정 - 구글의 Gemini AI 불러오기
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

"""
★★★ interrupt 핵심 개념 ★★★

interrupt() = 게임 일시정지 버튼
- 컴퓨터가 "잠깐! 사람한테 물어볼게" 하고 멈추는 거

Command(resume=대답) = 다시 시작 버튼
- 사람이 대답하면 그 대답 들고 다시 진행

result["__interrupt__"] = 뭘 물어봤는지 담긴 곳
- "뭐라고 물어봤더라?" 확인할 때 씀
"""

#-----------------------------------------
# 1번 예제: 부품 주문 승인/거부
# (엄마한테 "과자 사도 돼?" 물어보는 거랑 같음)
#-----------------------------------------
print(f"\n#1. Approve or reject - 부품 주문 승인/거부")
from typing import Literal, Optional, TypedDict

from langgraph.graph import StateGraph, START, END     # 그래프 만드는 도구
from langgraph.types import Command, interrupt         # 일시정지 & 다시시작 도구


# 상태(State) = 게임 세이브 파일에 저장되는 정보들
class OrderState(TypedDict):
    part_id: str      # 부품 번호
    part_name: str    # 부품 이름
    quantity: int     # 몇 개 살 건지
    status: Optional[Literal["pending", "approved", "rejected"]]  # 주문 상태


def search_part_node(state: OrderState):
    """
    1단계: 창고(MongoDB)에서 부품 찾기
    마트가서 물건 찾는 거랑 같음
    """
    # DB에서 부품 번호로 검색
    part = collection.find_one(
        {"partId": state["part_id"]},  # 이 번호로 찾아줘
        {"_id": 0, "name": 1, "partId": 1, "category": 1}  # 이것만 가져와
    )
    # 찾았으면 이름 반환, 못 찾았으면 "없음" 반환
    if part:
        return {"part_name": part.get("name", "Unknown")}
    return {"part_name": "부품을 찾을 수 없음"}


def approval_node(state: OrderState) -> Command[Literal["proceed", "cancel"]]:
    """
    2단계: 사람한테 "이거 사도 돼?" 물어보기
    ★ 여기서 일시정지(interrupt) 됨! ★
    """
    # interrupt = 일시정지! 사람한테 물어보고 기다림
    decision = interrupt({
        "question": "이 부품을 주문하시겠습니까?",  # 질문
        "part_id": state["part_id"],               # 부품 번호
        "part_name": state["part_name"],           # 부품 이름
        "quantity": state["quantity"],             # 수량
    })

    # 사람이 True 하면 proceed(진행)로, False 하면 cancel(취소)로
    # Command(goto=...) = "다음에 여기로 가!" 라는 명령
    return Command(goto="proceed" if decision else "cancel")


def proceed_node(state: OrderState):
    """3-A단계: 승인됐을 때 - 주문 진행!"""
    print(f"✅ 주문 승인: {state['part_name']} x {state['quantity']}개")
    return {"status": "approved"}


def cancel_node(state: OrderState):
    """3-B단계: 거부됐을 때 - 주문 취소!"""
    print(f"❌ 주문 거부: {state['part_name']}")
    return {"status": "rejected"}


# ===== 그래프 조립 (레고 조립하듯이) =====
builder = StateGraph(OrderState)  # 빈 그래프 만들기

# 노드 추가 = 역(정거장) 만들기
builder.add_node("search_part", search_part_node)  # 1번 역: 부품 찾기
builder.add_node("approval", approval_node)         # 2번 역: 승인 요청
builder.add_node("proceed", proceed_node)           # 3-A번 역: 주문 진행
builder.add_node("cancel", cancel_node)             # 3-B번 역: 주문 취소

# 엣지 추가 = 역이랑 역 연결하는 철로
builder.add_edge(START, "search_part")      # 출발 → 1번 역
builder.add_edge("search_part", "approval") # 1번 → 2번 역
builder.add_edge("proceed", END)            # 3-A번 → 종점
builder.add_edge("cancel", END)             # 3-B번 → 종점

# 그래프 완성! (위에서 만든 MongoDB checkpointer 사용)
graph = builder.compile(checkpointer=checkpointer)

# ===== 실행! =====
# config = 세이브 슬롯 번호
config = {"configurable": {"thread_id": "order-001"}}

# 첫 실행 - 부품 찾고, 승인 요청에서 멈춤
initial = graph.invoke(
    {"part_id": "3001", "quantity": 10, "status": "pending"},  # 3001번 부품 10개 주문
    config=config,
)
# 뭘 물어봤는지 출력
print(f"interrupt() 호출 결과: {initial['__interrupt__']}")

# 사람이 대답함 (True = 승인, False = 거부)
ans = input("주문 승인? (True/False): ").strip().lower() == "true"

# 대답 들고 다시 시작!
resumed = graph.invoke(Command(resume=ans), config=config)
print(f"최종 상태: {resumed['status']}")


#-----------------------------------------
# 2번 예제: AI가 쓴 글 검사받기
# (숙제하고 선생님한테 검사받는 거랑 같음)
#-----------------------------------------
print(f"\n#2. Review and edit state - AI 생성 부품 설명 리뷰")
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class DescriptionState(TypedDict):
    part_id: str               # 부품 번호
    part_info: str             # 부품 정보 (DB에서 가져온 거)
    generated_description: str  # AI가 쓴 설명


def fetch_part_node(state: DescriptionState):
    """1단계: DB에서 부품 정보 가져오기"""
    part = collection.find_one(
        {"partId": state["part_id"]},
        {"_id": 0, "name": 1, "partId": 1, "keywords": 1, "category": 1}
    )
    return {"part_info": str(part) if part else "부품 정보 없음"}


def generate_description_node(state: DescriptionState):
    """
    2단계: AI한테 설명 써달라고 시키기
    "야 이 부품 설명 좀 써줘" 하는 거
    """
    prompt = f"다음 레고 부품 정보를 바탕으로 한국어로 짧은 설명(2-3문장)을 작성해줘:\n{state['part_info']}"
    response = llm.invoke(prompt)  # AI야 써줘!
    return {"generated_description": response.content}  # AI가 쓴 글


def review_node(state: DescriptionState):
    """
    3단계: 사람한테 검사받기
    ★ 여기서 일시정지! ★
    "선생님 이거 맞아요?" 하고 기다리는 거
    """
    updated = interrupt({
        "instruction": "AI가 생성한 설명을 확인하고 수정해주세요",
        "part_info": state["part_info"],                       # 원본 정보
        "generated_description": state["generated_description"], # AI가 쓴 거
    })
    # 사람이 수정한 걸로 바꿈
    return {"generated_description": updated}


# 그래프 조립
builder = StateGraph(DescriptionState)
builder.add_node("fetch_part", fetch_part_node)              # 1번: 정보 가져오기
builder.add_node("generate_description", generate_description_node)  # 2번: AI가 글쓰기
builder.add_node("review", review_node)                      # 3번: 검사받기

builder.add_edge(START, "fetch_part")                  # 출발 → 1번
builder.add_edge("fetch_part", "generate_description") # 1번 → 2번
builder.add_edge("generate_description", "review")     # 2번 → 3번
builder.add_edge("review", END)                        # 3번 → 끝

graph = builder.compile(checkpointer=checkpointer)  # MongoDB checkpointer 사용

# 실행
config = {"configurable": {"thread_id": "desc-001"}}
initial = graph.invoke({"part_id": "3001"}, config=config)
print(f"interrupt() 호출 결과: {initial['__interrupt__']}")

# 사람이 수정 (안 치면 원본 그대로)
edited = input("수정된 설명 입력 (엔터시 원본 유지): ").strip()
if not edited:
    # 엔터만 쳤으면 AI가 쓴 거 그대로 씀
    edited = initial['__interrupt__'][0].value['generated_description']

# 수정한 거 들고 다시 시작
final_state = graph.invoke(Command(resume=edited), config=config)
print(f"최종 설명: {final_state['generated_description']}")


#-----------------------------------------
# 3번 예제: 도구(Tool) 안에서 일시정지
# (심부름 가서 "엄마 이거 살까요?" 물어보는 거)
#-----------------------------------------
print(f"\n#3. Interrupts in tools - 부품 검색 후 주문 승인")
from typing import TypedDict

from langchain.tools import tool  # 도구 만드는 데코레이터
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langchain.messages import ToolMessage  # 도구가 보내는 메시지


class AgentState(TypedDict):
    messages: list[dict]  # 대화 내용 저장


@tool  # 이거 붙이면 AI가 쓸 수 있는 도구가 됨
def search_and_order_part(query: str, quantity: int = 1):
    """
    레고 부품을 검색하고 주문하는 도구
    AI가 "이 도구 써야겠다!" 하면 자동으로 실행됨
    """

    # 1. DB에서 부품 검색 (마트에서 물건 찾기)
    results = list(collection.find(
        {"$or": [  # 이 중에 하나라도 맞으면 가져와
            {"name": {"$regex": query, "$options": "i"}},      # 이름에 포함
            {"keywords": {"$regex": query, "$options": "i"}},  # 키워드에 포함
            {"partId": query}                                   # 번호 일치
        ]},
        {"_id": 0, "name": 1, "partId": 1, "category": 1}
    ).limit(3))  # 최대 3개만

    # 못 찾았으면 바로 끝
    if not results:
        return f"'{query}' 검색 결과가 없습니다."

    # 2. ★ 일시정지! ★ 사람한테 "이거 살까요?" 물어보기
    response = interrupt({
        "action": "order_part",
        "query": query,           # 뭘 검색했는지
        "results": results,       # 찾은 물건들
        "quantity": quantity,     # 몇 개
        "message": "이 부품을 주문하시겠습니까?",
    })

    # 3. 사람이 approve(승인) 했으면 주문
    if response.get("action") == "approve":
        selected = response.get("selected_part", results[0])  # 선택한 거 (기본: 첫번째)
        final_qty = response.get("quantity", quantity)
        return f"✅ 주문 완료: {selected['name']} (ID: {selected['partId']}) x {final_qty}개"

    # 승인 안 했으면 취소
    return "❌ 주문이 취소되었습니다."


# 도구 목록
tools = [search_and_order_part]
tools_by_name = {t.name: t for t in tools}  # 이름으로 찾을 수 있게 딕셔너리로

# AI한테 "이 도구 쓸 수 있어" 알려주기
model_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState):
    """
    AI 노드: 사용자 메시지 보고 뭐 할지 결정
    "음... 부품 주문하래? 그럼 search_and_order_part 도구 써야겠다!"
    """
    result = model_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [result]}


def tools_node(state: AgentState):
    """
    도구 실행 노드: AI가 "이 도구 써!" 하면 여기서 실제로 실행
    ★ search_and_order_part 안에 interrupt 있으니까 여기서 멈춤! ★
    """
    result = []
    # AI가 부른 도구들 하나씩 실행
    for tool_call in state["messages"][-1].tool_calls:
        tool_func = tools_by_name[tool_call["name"]]  # 도구 찾기
        observation = tool_func.invoke(tool_call["args"])  # 실행!
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: AgentState):
    """
    분기점: AI가 도구 쓰려고 하면 tools로, 아니면 끝
    "도구 쓸 거야?" → "응" → tools 노드로
                   → "아니" → 끝
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"  # 도구 쓰러 가!
    return "end"  # 끝!


# 그래프 조립
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)  # AI 노드
builder.add_node("tools", tools_node)  # 도구 실행 노드
builder.add_edge(START, "agent")       # 출발 → AI
# 조건부 연결: AI가 도구 쓰면 tools로, 아니면 끝
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", END)         # 도구 → 끝

# MongoDB checkpointer 사용! (위에서 만든 거)
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "parts-workflow-001"}}

# 실행! "brick 부품 5개 주문해줘" 라고 말함
initial = graph.stream(
    {"messages": [
        {"role": "user", "content": "brick 부품 5개 주문해줘"}
    ]},
    config=config,
    stream_mode="values"  # 중간 과정도 보여줘
)

# 결과 출력 (interrupt 만나면 멈춤)
for chunk in initial:
    print(chunk)
    if "__interrupt__" in chunk:
        print(f"\n🛑 interrupt 발생: {chunk['__interrupt__']}")
        break

# 사람이 대답
ans = input("\n주문 승인? (approve/reject): ").strip()

# 대답 들고 다시 시작
resumed = graph.stream(
    Command(resume={"action": ans, "quantity": 5}),
    config=config,
    stream_mode="values"
)
for chunk in resumed:
    print(chunk)


#-----------------------------------------
# 4번 예제: 맞을 때까지 계속 물어보기
# (비밀번호 틀리면 "다시 입력하세요" 뜨는 거랑 같음)
#-----------------------------------------
print(f"\n#4. Validating human input - 부품 ID 검증")
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class PartInputState(TypedDict):
    part_id: str | None     # 부품 번호 (처음엔 None)
    part_info: dict | None  # 부품 정보 (처음엔 None)


def get_part_id_node(state: PartInputState):
    """
    유효한 부품 ID 받을 때까지 계속 물어보기

    작동 방식:
    1. "부품 ID 입력해" → 일시정지
    2. 사람이 입력 → DB에서 확인
    3. 있으면 → 끝!
       없으면 → "그런 거 없어. 다시 입력해" → 1번으로
    """
    prompt = "주문할 부품 ID를 입력해주세요:"

    while True:  # 무한 반복 (맞출 때까지)
        # ★ 일시정지! ★ 사람한테 물어봄
        answer = interrupt(prompt)

        # DB에서 진짜 있는지 확인
        part = collection.find_one(
            {"partId": str(answer)},
            {"_id": 0, "name": 1, "partId": 1, "category": 1}
        )

        if part:  # 찾았다!
            return {"part_id": str(answer), "part_info": part}

        # 못 찾았으면 메시지 바꾸고 다시 물어볼 준비
        # (다음 resume 때 이 prompt로 interrupt 됨)
        prompt = f"'{answer}'는 존재하지 않는 부품 ID입니다. 다시 입력해주세요:"


# 그래프 (진짜 간단 - 노드 1개)
builder = StateGraph(PartInputState)
builder.add_node("get_part_id", get_part_id_node)
builder.add_edge(START, "get_part_id")
builder.add_edge("get_part_id", END)

# MongoDB checkpointer 사용!
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "input-001"}}

# 1) 첫 실행 - "부품 ID 입력해" 하고 멈춤
print("\n--- 1) 첫 실행 (부품 ID 요청) ---")
for event in graph.stream({"part_id": None, "part_info": None}, config=config, stream_mode="values"):
    print(event)

# 2) 틀린 거 입력 - "99999는 없어. 다시 입력해" 하고 또 멈춤
print("\n--- 2) 잘못된 입력 (존재하지 않는 ID) ---")
for event in graph.stream(Command(resume="99999"), config=config, stream_mode="values"):
    print(event)

# 3) 맞는 거 입력 - "찾았다! 끝!" 하고 종료
print("\n--- 3) 올바른 입력 (존재하는 ID: 3001) ---")
for event in graph.stream(Command(resume="3001"), config=config, stream_mode="values"):
    print(event)


# =========================================================
# 부록: interrupt 쓸 때 규칙들 (지키면 버그 안 남)
# =========================================================
""" Appendix
#-----------------------------------------
# interrupt 5가지 규칙
# (안 지키면 이상하게 작동함 주의!)
#-----------------------------------------

# 규칙 1: interrupt를 try/except로 감싸지 마라
# ❌ 나쁜 예
def bad_node(state):
    try:
        answer = interrupt("질문")  # 이러면 안 됨!
    except:
        pass

# ✅ 좋은 예
def good_node(state):
    answer = interrupt("질문")  # interrupt는 밖에
    try:
        do_something()  # 위험한 건 따로
    except:
        pass


# 규칙 2: interrupt 순서 바꾸지 마라
# 항상 같은 순서로! 조건문으로 건너뛰기 금지!
def node_a(state):
    name = interrupt("이름?")   # 항상 1번
    age = interrupt("나이?")    # 항상 2번
    city = interrupt("도시?")   # 항상 3번
    return {"name": name, "age": age, "city": city}


# 규칙 3: interrupt에는 간단한 값만
# 숫자, 문자열, True/False, 리스트, 딕셔너리만 OK
# 클래스 객체 같은 복잡한 거 넣으면 저장 안 됨
def node_a(state):
    name = interrupt("이름?")     # OK: 문자열 받음
    count = interrupt(42)         # OK: 숫자 보냄
    approved = interrupt(True)    # OK: True/False


# 규칙 4: interrupt 전에는 "다시 해도 괜찮은 것"만
# resume하면 노드가 처음부터 다시 실행됨!
def node_a(state):
    # ✅ 이건 OK - 여러 번 해도 결과 같음
    db.upsert_user(user_id=state["user_id"], status="pending")

    approved = interrupt("승인?")

    # ❌ 이건 위험 - interrupt 전에 돈 보내면 resume할 때 또 보냄!
    # send_money(1000)  # 이거 interrupt 전에 있으면 안 됨!


# 규칙 5: 서브그래프 안에서 interrupt → 부모도 다시 실행됨
# 그냥 알아두기만 하면 됨
"""


"""
#-----------------------------------------
# 디버깅용 interrupt (고급)
# 특정 노드 전/후에서 강제로 멈추게 하기
# LangGraph Studio에서 테스트할 때 유용
#-----------------------------------------

graph = builder.compile(
    interrupt_before=["node_a"],      # node_a 실행 "전"에 멈춤
    interrupt_after=["node_b"],       # node_b 실행 "후"에 멈춤
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "debug-1"}}

# 멈출 때까지 실행
graph.invoke(inputs, config=config)

# 다시 시작 (None 넣으면 그냥 계속 진행)
graph.invoke(None, config=config)
"""
