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
from pydantic import BaseModel, Field
from typing import Literal #주어진 보기 안에서 선택하도록 강제

#-------------------------------------
#Evaluator-optimizer(평가-개선 루프) *예제와 다르게 개선 루프 3회 제한
#-------------------------------------
# # Graph state
#
# class State(TypedDict):
#     joke: str
#     topic: str
#     feedback: str
#     funny_or_not: str
#     attempts: int  # 개선 시도 횟수 카운터
#
#
# # 평가 결과를 구조화할 스키마(Feedback)
# class Feedback(BaseModel):
#     # evaluator의 출력 형식을 강제로 고정
#     grade: Literal["funny", "not funny"] = Field(
#         description="Decide if the joke is funny or not.",
#     )
#     feedback: str = Field(
#         description="If the joke is not funny, provide feedback on how to improve it.",
#     )
#
#
# # evaluator(평가 LLM) 만들기
# evaluator = llm.with_structured_output(Feedback)
#
#
# # 노드 1: llm_call_generator (생성기)
# def llm_call_generator(state: State):
#     """LLM generates a joke"""
#
#     # 시도 횟수 카운트 (없으면 0으로 시작)
#     current_attempts = state.get("attempts") or 0
#     new_attempts = current_attempts + 1
#
#     if state.get("feedback"):
#         msg = llm.invoke(
#             f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
#         )
#     else:
#         msg = llm.invoke(f"Write a joke about {state['topic']}")
#
#     # 어떤 내용을 생성했는지 출력
#     print(f"[GENERATE] Joke {new_attempts}: {msg.content[:100]}\n")
#
#     return {"joke": msg.content, "attempts": new_attempts}
#
#
# # 노드 2: llm_call_evaluator (평가자)
# def llm_call_evaluator(state: State):
#     """LLM evaluates the joke"""
#
#     grade = evaluator.invoke(f"Grade the joke {state['joke']}")
#
#     # 어떤 내용을 평가했는지 출력
#     print(f"[EVALUATE] Joke {state['attempts']}: {grade.grade}\n")
#
#     return {"funny_or_not": grade.grade, "feedback": grade.feedback}
#
#
# # 라우팅 함수
# def route_joke(state: State):
#     """Route back to joke generator or end based upon feedback from the evaluator"""
#
#     if state["funny_or_not"] == "funny":
#         print("[END] 웃긴 농담으로 판정되어 종료")
#
#         return "Accepted"
#     elif state["funny_or_not"] == "not funny":
#         # 3회 이상 시도했다면 여기서 멈춤 (Accepted 리턴 -> END)
#         if state.get("attempts", 0) >= 3:
#             print("[END] 3회 개선 시도 도달 → 강제 종료")
#             return "Accepted"
#
#         print("[LOOP] 개선 피드백 반영하여 재시도")
#         return "Rejected + Feedback"
#
#
# # Build workflow
# optimizer_builder = StateGraph(State)
#
# # Add the nodes
# optimizer_builder.add_node("llm_call_generator", llm_call_generator)
# optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)
#
# # Add edges to connect nodes
# optimizer_builder.add_edge(START, "llm_call_generator")
# optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
# optimizer_builder.add_conditional_edges(
#     "llm_call_evaluator",
#     route_joke,
#     {  # Name returned by route_joke : Name of next node to visit
#         "Accepted": END,
#         "Rejected + Feedback": "llm_call_generator",
#     },
# )
#
# # Compile the workflow
# optimizer_workflow = optimizer_builder.compile()

#-------------------------------------
# (New) YouTube 제목 생성기
#-------------------------------------

class State(TypedDict):
    video_topic: str  # 입력: 영상 주제
    title: str        # 생성: 영상 제목
    feedback: str     # 평가 피드백
    is_clickable: str # 평가 결과 (yes/no)
    attempts: int     # 시도 횟수

class Evaluation(BaseModel):
    grade: Literal["yes", "no"] = Field(description="제목이 클릭을 유도하고 매력적인지 판단 (yes/no)")
    feedback: str = Field(description="제목이 매력적이지 않다면 개선할 점을 제안")

evaluator = llm.with_structured_output(Evaluation)

def llm_call_generator(state: State):
    current_attempts = state.get("attempts") or 0
    new_attempts = current_attempts + 1

    if state.get("feedback"):
        prompt = f"Create a catchy YouTube title about '{state['video_topic']}'. Reflect this feedback: {state['feedback']}"
    else:
        prompt = f"Create a catchy YouTube title about '{state['video_topic']}'."
    
    msg = llm.invoke(prompt)
    print(f"[GENERATE] Title {new_attempts}: {msg.content}")
    return {"title": msg.content, "attempts": new_attempts}

def llm_call_evaluator(state: State):
    grade = evaluator.invoke(f"Is this YouTube title catchy and clickable? Title: {state['title']}")
    print(f"[EVALUATE] Clickable? {grade.grade}")
    return {"is_clickable": grade.grade, "feedback": grade.feedback}

def route_evaluation(state: State):
    if state["is_clickable"] == "yes":
        print("[END] 매력적인 제목으로 판정되어 종료")
        return "Accepted"
    elif state["is_clickable"] == "no":
        if state.get("attempts", 0) >= 3:
            print("[END] 3회 시도 초과 -> 종료")
            return "Accepted"
        print("[LOOP] 피드백 반영하여 재시도")
        return "Rejected + Feedback"

optimizer_builder = StateGraph(State)
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_evaluation,
    {"Accepted": END, "Rejected + Feedback": "llm_call_generator"}
)
optimizer_workflow = optimizer_builder.compile()

# Show the workflow
print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :") #사이트 들어가서 코드 붙여넣기
print(optimizer_workflow.get_graph(xray=True).draw_mermaid())

# Invoke
state = optimizer_workflow.invoke({"video_topic": "AI로 코딩하는 방법"})
print(f"\n최종 결과: {state['title']}")