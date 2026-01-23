import os
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

load_dotenv()

# 1. 모델 설정
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

# 2. 상태(State) 정의
class TravelState(TypedDict):
    city: str
    food: str
    place: str
    tip: str
    final_report: str

# 3. 노드(Nodes) 정의
def call_food_agent(state: TravelState):
    msg = llm.invoke(f"{state['city']}의 대표적인 맛집 하나만 추천해줘.")
    return {"food": msg.content}

def call_place_agent(state: TravelState):
    msg = llm.invoke(f"{state['city']}의 유명한 관광지 하나만 추천해줘.")
    return {"place": msg.content}

def call_tip_agent(state: TravelState):
    msg = llm.invoke(f"{state['city']} 여행 시 가장 중요한 꿀팁 하나만 알려줘.")
    return {"tip": msg.content}

def aggregator(state: TravelState):
    report = f"--- {state['city']} 여행 정보 ---\n"
    report += f"🍴 맛집: {state['food']}\n"
    report += f"📍 명소: {state['place']}\n"
    report += f"💡 팁: {state['tip']}"
    return {"final_report": report}

# 4. 워크플로우 구성
builder = StateGraph(TravelState)

builder.add_node("food_node", call_food_agent)
builder.add_node("place_node", call_place_agent)
builder.add_node("tip_node", call_tip_agent)
builder.add_node("aggregator", aggregator)

# 병렬 구조 연결 (START에서 세 노드로 동시에 뻗어나감)
builder.add_edge(START, "food_node")
builder.add_edge(START, "place_node")
builder.add_edge(START, "tip_node")

# 세 노드에서 다시 aggregator로 모임 (Fan-in)
builder.add_edge("food_node", "aggregator")
builder.add_edge("place_node", "aggregator")
builder.add_edge("tip_node", "aggregator")

builder.add_edge("aggregator", END)

# 컴파일
travel_workflow = builder.compile()

# --------------------------------------------------
# 5. 시각화 코드
# --------------------------------------------------
print("\n[Mermaid Syntax]")
print("아래 코드를 복사해서 https://mermaid.live/ 에 붙여넣으세요:")
print("-" * 30)
# xray=True는 내부 구조를 더 상세하게 보여줍니다.
print(travel_workflow.get_graph(xray=True).draw_mermaid())
print("-" * 30)

# 6. 실행
result = travel_workflow.invoke({"city": "도쿄"})
print("\n[최종 결과]")
print(result["final_report"])