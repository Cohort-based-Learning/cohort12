import os
from typing import Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain & LangGraph 관련 모듈
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

# 환경 변수 로드
load_dotenv()

# 1. 모델 설정 (사용자 요청 기반 최신 모델)
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

# 2. 그래프 상태(State) 정의
class EmailState(TypedDict):
    requirement: str    # 사용자 요청사항
    draft: str          # 생성된 이메일 초안
    feedback: str       # 평가자의 피드백
    is_professional: str # 평가 결과 ("yes" or "no")
    attempts: int       # 반복 횟수 카운터

# 3. 평가 결과 구조화 (Pydantic)
class Evaluation(BaseModel):
    is_professional: Literal["yes", "no"] = Field(
        description="이메일이 충분히 격식 있고 명확한가요?"
    )
    feedback: str = Field(
        description="전문성이 부족하다면 어떤 점을 고쳐야 할지 상세히 적어주세요."
    )

# 구조화된 출력을 사용하는 평가 모델 생성
evaluator_llm = llm.with_structured_output(Evaluation)

# ---------------------------------------------------------
# 4. 노드(Node) 함수 정의
# ---------------------------------------------------------

def generator_node(state: EmailState):
    """이메일 초안을 작성하거나 피드백을 반영해 수정합니다."""
    attempts = state.get("attempts", 0) + 1
    
    prompt = f"요청사항: {state['requirement']}\n"
    if state.get("feedback"):
        prompt += f"이전 피드백 반영: {state['feedback']}\n"
        prompt += "위 피드백을 반영하여 더 완벽한 비즈니스 이메일로 수정하세요."
    else:
        prompt += "위 내용을 바탕으로 예의 바른 비즈니스 이메일 초안을 작성하세요."

    response = llm.invoke(prompt)
    print(f"\n[Generator] {attempts}번째 시도: 초안 작성 완료")
    return {"draft": response.content, "attempts": attempts}

def evaluator_node(state: EmailState):
    """작성된 이메일을 아주 깐깐하게 검토합니다."""
    print("\n🧐 [Evaluator] 깐깐한 상사가 검토 중입니다...")
    
    # 평가 지침을 더 구체적으로 줍니다.
    review_prompt = f"""
    다음 이메일을 비즈니스 관점에서 검토하세요:
    {state['draft']}
    
    [필수 합격 기준]
    1. 정확한 '결제 예정일(날짜)'이 명시되어 있는가? (없으면 무조건 no)
    2. 지연 사유가 구체적인가? (불분명하면 no)
    3. 격식이 완벽한가?
    """
    
    result = evaluator_llm.invoke(review_prompt)
    
    print(f"   - 결과: {result.is_professional}")
    print(f"   - 피드백: {result.feedback}")
    
    return {"is_professional": result.is_professional, "feedback": result.feedback}
def router_logic(state: EmailState):
    """평가 결과와 시도 횟수에 따라 다음 경로를 결정합니다."""
    if state["is_professional"] == "yes":
        print("✅ 검토 통과: 결과가 만족스럽습니다.")
        return "Accepted"
    
    if state["attempts"] >= 3:
        print("⚠️ 최대 시도 횟수 도달: 현재 결과에서 종료합니다.")
        return "Accepted"
    
    print(f"❌ 보완 필요: {state['feedback']}")
    return "Retry"

# ---------------------------------------------------------
# 5. 그래프 빌드 (Workflow)
# ---------------------------------------------------------

workflow = StateGraph(EmailState)

# 노드 추가
workflow.add_node("drafter", generator_node)
workflow.add_node("reviewer", evaluator_node)

# 엣지 연결
workflow.add_edge(START, "drafter")
workflow.add_edge("drafter", "reviewer")

# 조건부 엣지(라우팅) 추가
workflow.add_conditional_edges(
    "reviewer",
    router_logic,
    {
        "Accepted": END,
        "Retry": "drafter"
    }
)

# 컴파일
app = workflow.compile()

# ---------------------------------------------------------
# 6. 워크플로우 시각화 및 실행
# ---------------------------------------------------------

# (1) 워크플로우 그래프 출력
print("\n" + "="*50)
print("📊 아래 Mermaid 코드를 복사해서 https://mermaid.live/ 에 붙여넣으세요:")
print("="*50)
print(app.get_graph().draw_mermaid())
print("="*50 + "\n")

# (2) 실제 실행
initial_input = {
    "requirement": "돈 늦게 줄 것 같으니까 대충 미안하다고 메일 하나 보내봐. 바쁘니까 짧게 써.",
    "attempts": 0
}

result_state = app.invoke(initial_input)

print("\n🚀 [최종 결과물]")
print("-" * 30)
print(result_state["draft"])