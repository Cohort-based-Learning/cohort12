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
# Graph state

class State(TypedDict):
    scenario: str
    theme: str
    feedback: str
    is_provocative: str
    attempts: int  # 개선 시도 횟수 카운터


# 평가 결과를 구조화할 스키마(Evaluation)
class Evaluation(BaseModel):
    # evaluator의 출력 형식을 강제로 고정
    grade: Literal["provocative", "boring"] = Field(
        description="Decide if the scenario is provocative and stimulating enough.",
    )
    feedback: str = Field(
        description="If the scenario is boring, provide feedback on how to make it more shocking.",
    )


# evaluator(평가 LLM) 만들기
evaluator = llm.with_structured_output(Evaluation)


# 노드 1: llm_call_generator (생성기)
def llm_call_generator(state: State):
    """LLM generates a makjang drama scenario"""

    # 시도 횟수 카운트 (없으면 0으로 시작)
    current_attempts = state.get("attempts") or 0
    new_attempts = current_attempts + 1

    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a shocking 'Makjang' drama synopsis about {state['theme']} but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a shocking 'Makjang' drama synopsis about {state['theme']}. It must be extreme.")

    # 어떤 내용을 생성했는지 출력
    print(f"[GENERATE] Scenario {new_attempts}: {msg.content[:100]}...\n")

    return {"scenario": msg.content, "attempts": new_attempts}


# 노드 2: llm_call_evaluator (평가자)
def llm_call_evaluator(state: State):
    """LLM evaluates the scenario"""

    grade = evaluator.invoke(f"Grade the scenario: {state['scenario']}")

    # 어떤 내용을 평가했는지 출력
    print(f"[EVALUATE] Scenario {state['attempts']}: {grade.grade}\n")

    return {"is_provocative": grade.grade, "feedback": grade.feedback}


# 라우팅 함수
def route_scenario(state: State):
    """Route back to generator or end based upon feedback from the evaluator"""

    if state["is_provocative"] == "provocative":
        print("[END] 자극적인 시나리오로 판정되어 종료")

        return "Accepted"
    elif state["is_provocative"] == "boring":
        # 3회 이상 시도했다면 여기서 멈춤 (Accepted 리턴 -> END)
        if state.get("attempts", 0) >= 3:
            print("[END] 3회 개선 시도 도달 → 강제 종료")
            return "Accepted"

        print("[LOOP] 개선 피드백 반영하여 재시도")
        return "Rejected + Feedback"


# Build workflow
optimizer_builder = StateGraph(State)

# Add the nodes
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

# Add edges to connect nodes
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_scenario,
    {  # Name returned by route_scenario : Name of next node to visit
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

# Compile the workflow
optimizer_workflow = optimizer_builder.compile()

# Show the workflow
print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :") #사이트 들어가서 코드 붙여넣기
print(optimizer_workflow.get_graph(xray=True).draw_mermaid())

# Invoke
state = optimizer_workflow.invoke({"theme": "출생의 비밀과 복수"})
print(f"\n최종 결과:\n{state['scenario']}")