#기본 설정
import os
from dotenv import load_dotenv
load_dotenv()

#-------------------------------------
#모델 설정
#-------------------------------------
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

#-------------------------------------
#Prompt chaining : “이전 LLM 호출의 결과를 다음 호출 입력으로 넘기는” 순차 처리.
#-------------------------------------
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# 상태 정의
class State(TypedDict):
    topic: str
    story: str
    improved_story: str
    final_story: str


# 노드 1: story
def generate_story(state: State):
    """First LLM call to generate initial story"""

    msg = llm.invoke(f"{state['topic']}에 대해 무서운 이야기를 만들어줘") #상태값 업데이트
    return {"story": msg.content}

# 게이트 함수: check_punchline (add_conditional_edges에서 문자열 분기 키로 사용)
def check_punchline(state: State):
    """Gate function to check if the story has a punchline"""

    # ?나 ! 있으면 “펀치라인이 있다”고 판단
    if "비명" in state["story"] or "밤" in state["story"] or "귀신" in state["story"]:
        return "Pass"
    return "Fail"


# 노드 2: improve_story
def improve_story(state: State):
    """Second LLM call to improve the story"""

    msg = llm.invoke(f"{state['story']} 이 이야기를 더 무섭게 만들어줘")
    return {"improved_story": msg.content}


# 노드 3: polish_story
def polish_story(state: State):
    """Third LLM call for final polish"""
    msg = llm.invoke(f"{state['improved_story']} 이 이야기에 놀라운 반전을 추가해줘")
    return {"final_story": msg.content}


# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate_story", generate_story)
workflow.add_node("improve_story", improve_story)
workflow.add_node("polish_story", polish_story)

# Add edges to connect nodes
workflow.add_edge(START, "generate_story")
workflow.add_conditional_edges(
    "generate_story", check_punchline, {"Fail": "improve_story", "Pass": END}
)
workflow.add_edge("improve_story", "polish_story")
workflow.add_edge("polish_story", END)

# Compile
chain = workflow.compile()

# Show workflow
print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :") #사이트 들어가서 코드 붙여넣기
print(chain.get_graph(xray=True).draw_mermaid())

# Invoke
state = chain.invoke({"topic": "고양이"})
print("Initial story:")
print(state["story"])
print("\n--- --- ---\n")
if "improved_story" in state:
    print("Improved story:")
    print(state["improved_story"])
    print("\n--- --- ---\n")

    print("Final story:")
    print(state["final_story"])
else:
    print("Final story:")
    print(state["story"])