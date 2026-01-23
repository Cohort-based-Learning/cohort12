#기본 설정
import os
from dotenv import load_dotenv
load_dotenv()

#-------------------------------------
#모델 설정
#-------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

#-------------------------------------
#Parallelization(병렬화: 서로 독립인 작업을 동시에 돌림)
#-------------------------------------

# Graph state
class State(TypedDict):
    topic: str
    grading: str
    wrong_answer_note: str
    tips: str
    combined_output: str


# Nodes
def call_llm_1(state: State):
    """First LLM call to generate grading"""

    msg = llm.invoke(f"'{state['topic']}' 문제에 대한 서술형 채점 기준을 작성해줘.")
    return {"grading": msg.content}


def call_llm_2(state: State):
    """Second LLM call to generate wrong answer note"""

    msg = llm.invoke(f"'{state['topic']}' 문제를 틀렸을 때 작성할 오답노트 양식을 만들어줘.")
    return {"wrong_answer_note": msg.content}


def call_llm_3(state: State):
    """Third LLM call to generate tips"""

    msg = llm.invoke(f"'{state['topic']}' 문제를 풀 때 주의할 점과 팁을 알려줘.")
    return {"tips": msg.content}


def aggregator(state: State): #합치기
    """Combine the grading, wrong answer note and tips into a single output"""

    combined = f"'{state['topic']}' 문제에 대한 학습 가이드\n\n"
    combined += f"--- 서술형 채점 기준 ---\n{state['grading']}\n\n"
    combined += f"--- 오답노트 작성 가이드 ---\n{state['wrong_answer_note']}\n\n"
    combined += f"--- 주의사항 및 팁 ---\n{state['tips']}"
    return {"combined_output": combined}


# Build workflow
parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# Add edges to connect nodes 독립적 실행이 포인트
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# Show workflow
print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :") #사이트 들어가서 코드 붙여넣기
print(parallel_workflow.get_graph(xray=True).draw_mermaid())

# Invoke
state = parallel_workflow.invoke({"topic": "이차방정식의 근의 공식 유도"})
print(state["combined_output"])