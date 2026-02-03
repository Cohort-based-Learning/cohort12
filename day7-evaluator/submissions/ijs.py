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
from typing import Literal, List #주어진 보기 안에서 선택하도록 강제

#-------------------------------------
#Evaluator-optimizer(평가-개선 루프) *예제와 다르게 개선 루프 3회 제한
#심사위원 3명이 각각 0~100점 평가, 총합 250점 이상이면 합격
#-------------------------------------
# Graph state

class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    scores: List[int]  # 3명의 심사위원 점수
    total_score: int   # 총점
    pass_or_fail: str  # 합격/불합격
    attempts: int  # 개선 시도 횟수 카운터


# 평가 결과를 구조화할 스키마(JudgeScore) - 각 심사위원용
class JudgeScore(BaseModel):
    score: int = Field(
        description="Score the joke from 0 to 100 based on how funny it is.",
        ge=0,
        le=100
    )
    feedback: str = Field(
        description="Provide feedback on the joke and how to improve it.",
    )
    review: str = Field(
        description="Write a detailed review/commentary about the joke in Korean as a judge would say on a TV show.",
    )


# 3명의 심사위원 evaluator 생성 (각각 다른 관점)
judge1 = llm.with_structured_output(JudgeScore)
judge2 = llm.with_structured_output(JudgeScore)
judge3 = llm.with_structured_output(JudgeScore)


# 노드 1: llm_call_generator (생성기)
def llm_call_generator(state: State):
    """LLM generates a joke"""

    # 시도 횟수 카운트 (없으면 0으로 시작)
    current_attempts = state.get("attempts") or 0
    new_attempts = current_attempts + 1

    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    
    # 어떤 내용을 생성했는지 출력
    print(f"[GENERATE] Joke {new_attempts}: {msg.content[:100]}\n")

    return {"joke": msg.content, "attempts": new_attempts}


# 노드 2: llm_call_evaluator (3명의 심사위원이 평가)
def llm_call_evaluator(state: State):
    """3 LLM judges evaluate the joke"""

    joke = state['joke']

    # 심사위원 1: 안성재 - 유머 감각 기준
    result1 = judge1.invoke(
        f"You are 안성재 (Ahn Sung-jae), a famous Korean chef judge known for your sharp wit and high standards. Focus on cleverness and wit. Score this joke from 0 to 100 and provide a detailed review in Korean as you would on a TV show: {joke}"
    )

    # 심사위원 2: 강레오 - 창의성 기준
    result2 = judge2.invoke(
        f"You are 강레오 (Chef Leo Kang), a passionate and expressive Korean chef judge. Focus on creativity and originality. Score this joke from 0 to 100 and provide a detailed review in Korean as you would on a TV show: {joke}"
    )

    # 심사위원 3: 에드워드 리 - 전달력 기준
    result3 = judge3.invoke(
        f"You are 에드워드 리 (Edward Lee), a Korean-American chef judge known for your warm and thoughtful critiques. Focus on delivery and timing. Score this joke from 0 to 100 and provide a detailed review in Korean as you would on a TV show: {joke}"
    )

    scores = [result1.score, result2.score, result3.score]
    total_score = sum(scores)

    # 합격/불합격 판정
    pass_or_fail = "pass" if total_score >= 250 else "fail"

    # 피드백 통합
    combined_feedback = f"""
    [안성재 (유머감각)] 점수: {result1.score}/100 - {result1.feedback}
    [강레오 (창의성)] 점수: {result2.score}/100 - {result2.feedback}
    [에드워드 리 (전달력)] 점수: {result3.score}/100 - {result3.feedback}
    """

    # 평가 결과 출력
    print(f"[EVALUATE] Attempt {state['attempts']}:")
    print(f"\n  👨‍🍳 안성재 심사위원: {result1.score}점")
    print(f"     심사평: {result1.review}")
    print(f"\n  👨‍🍳 강레오 심사위원: {result2.score}점")
    print(f"     심사평: {result2.review}")
    print(f"\n  👨‍🍳 에드워드 리 심사위원: {result3.score}점")
    print(f"     심사평: {result3.review}")
    print(f"\n  📊 총점: {total_score}/300 ({'합격' if pass_or_fail == 'pass' else '불합격'})\n")

    return {
        "scores": scores,
        "total_score": total_score,
        "pass_or_fail": pass_or_fail,
        "feedback": combined_feedback
    }


# 라우팅 함수
def route_joke(state: State):
    """Route back to joke generator or end based upon feedback from the evaluators"""

    if state["pass_or_fail"] == "pass":
        print(f"[END] 합격! 총점 {state['total_score']}/300 (250점 이상)")
        return "Accepted"
    elif state["pass_or_fail"] == "fail":
        # 3회 이상 시도했다면 여기서 멈춤 (Accepted 리턴 -> END)
        if state.get("attempts", 0) >= 3:
            print(f"[END] 3회 개선 시도 도달 → 강제 종료 (최종 점수: {state['total_score']}/300)")
            return "Accepted"

        print(f"[LOOP] 불합격 (총점: {state['total_score']}/300) → 피드백 반영하여 재시도")
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
    route_joke,
    {  # Name returned by route_joke : Name of next node to visit
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
state = optimizer_workflow.invoke({"topic": "Cats"})
print("\n" + "="*50)
print("최종 농담:")
print(state["joke"])
print(f"\n최종 점수: {state.get('total_score', 0)}/300")
print(f"심사위원별 점수: {state.get('scores', [])}")
