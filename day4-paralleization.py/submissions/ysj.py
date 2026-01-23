#기본 설정
import os
from dotenv import load_dotenv
load_dotenv()

#-------------------------------------
#모델 설정
#-------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

#-------------------------------------
#Parallelization(병렬화: 서로 독립인 작업을 동시에 돌림)
#-------------------------------------

# Graph state
class State(TypedDict):
    topic: str
    accommodation: str
    food: str
    activities: str
    combined_output: str


# Nodes
def get_accommodation(state: State):
    """숙소 추천"""
    msg = llm.invoke(f"{state['topic']} 여행 시 머물기 좋은 숙소 3곳을 추천해줘.")
    return {"accommodation": msg.content}


def get_food(state: State):
    """맛집 추천"""
    msg = llm.invoke(f"{state['topic']}에서 유명한 맛집 3곳을 추천해줘.")
    return {"food": msg.content}


def get_activities(state: State):
    """관광지/액티비티 추천"""
    msg = llm.invoke(f"{state['topic']}에서 꼭 가봐야 할 관광지나 액티비티 3가지를 추천해줘.")
    return {"activities": msg.content}


def aggregator(state: State): #합치기
    """Combine into a travel plan"""
    combined = f"--- {state['topic']} 여행 추천 코스 ---\n\n"
    combined += f"🏨 숙소:\n{state['accommodation']}\n\n"
    combined += f"🍽️ 맛집:\n{state['food']}\n\n"
    combined += f"🎡 즐길거리:\n{state['activities']}"
    return {"combined_output": combined}


# Build workflow
parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("get_accommodation", get_accommodation)
parallel_builder.add_node("get_food", get_food)
parallel_builder.add_node("get_activities", get_activities)
parallel_builder.add_node("aggregator", aggregator)

# Add edges to connect nodes (독립적 실행)
parallel_builder.add_edge(START, "get_accommodation")
parallel_builder.add_edge(START, "get_food")
parallel_builder.add_edge(START, "get_activities")
parallel_builder.add_edge("get_accommodation", "aggregator")
parallel_builder.add_edge("get_food", "aggregator")
parallel_builder.add_edge("get_activities", "aggregator")
parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# Show workflow
print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :")
print(parallel_workflow.get_graph(xray=True).draw_mermaid())

# Invoke
state = parallel_workflow.invoke({"topic": "제주도"})
print(state["combined_output"])