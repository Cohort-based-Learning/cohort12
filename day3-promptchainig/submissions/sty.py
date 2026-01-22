#기본 설정
import os
from dotenv import load_dotenv
load_dotenv()

#-------------------------------------
#모델 설정
#-------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#-------------------------------------
#Prompt chaining : “이전 LLM 호출의 결과를 다음 호출 입력으로 넘기는” 순차 처리.
#-------------------------------------
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# 상태 정의
class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str


# 노드 1: generate_joke
def generate_joke(state: State):
    """First LLM call to generate initial joke"""

    msg = llm.invoke(f"{state['topic']}에 대해 짧은 농담을 만들어줘") #상태값 업데이트
    return {"joke": msg.content}

# 게이트 함수: check_punchline (add_conditional_edges에서 문자열 분기 키로 사용)
def check_punchline(state: State):
    """Gate function to check if the joke has a punchline"""

    # ?나 ! 있으면 “펀치라인이 있다”고 판단
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    return "Fail"


# 노드 2: improve_joke
def improve_joke(state: State):
    """Second LLM call to improve the joke"""

    msg = llm.invoke(f"{state['joke']} 이 농담을 더 재미있게 만들어줘")
    return {"improved_joke": msg.content}


# 노드 3: polish_joke
def polish_joke(state: State):
    """Third LLM call for final polish"""
    msg = llm.invoke(f"{state['improved_joke']} 이 농담에 놀라운 반전을 추가해줘")
    return {"final_joke": msg.content}


# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

# Add edges to connect nodes
workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges(
    "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
)
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

# Compile
chain = workflow.compile()

# Show workflow
print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :") #사이트 들어가서 코드 붙여넣기
print(chain.get_graph(xray=True).draw_mermaid())

# Invoke
state = chain.invoke({"topic": "고양이"})
print("Initial joke:")
print(state["joke"])
print("\n--- --- ---\n")
if "improved_joke" in state:
    print("Improved joke:")
    print(state["improved_joke"])
    print("\n--- --- ---\n")

    print("Final joke:")
    print(state["final_joke"])
else:
    print("Final joke:")
    print(state["joke"])