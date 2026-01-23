import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 환경 변수 로드
load_dotenv()

# 모델 설정 (실제 사용 가능한 모델명으로 변경 권장)
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")

# -------------------------------------
# 상태(State) 정의
# -------------------------------------
class EmailState(TypedDict):
    topic: str          # 사용자 입력 주제
    draft: str          # 1차 초안
    rewritten_draft: str # 수정된 초안
    final_email: str    # 최종 결과물

# -------------------------------------
# 노드(Node) 정의
# -------------------------------------

# 노드 1: 초안 작성
def write_draft(state: EmailState):
    """주제에 따른 이메일 초안 작성"""
    print("--- [Node 1] Writing Draft ---")
    prompt = (
        f"'{state['topic']}'에 대한 비즈니스 이메일 초안을 한국어로 작성해줘.\n"
        f"반드시 '하나의 완성된 메일'만 출력하고, 다른 옵션이나 추가 설명은 하지 마."
    )
    msg = llm.invoke(prompt)
    return {"draft": msg.content}

# 게이트 함수: 검사 로직
def check_tone(state: EmailState):
    """
    이메일에 '죄송' 또는 '사과'라는 단어가 너무 많은지 검사하는 단순 로직.
    실제로는 LLM을 사용하여 톤을 분석할 수도 있습니다.
    """
    print("--- [Gate] Checking Tone ---")
    draft_text = state["draft"]
    
    # 단순 키워드 검사: 사과하는 내용이 포함되어 있으면 수정을 위해 Fail 반환
    if "죄송" in draft_text or "사과" in draft_text or "미안" in draft_text:
        return "Fail" # 수정 필요
    return "Pass" # 수정 불필요

# 노드 2: 표현 수정 (Fail인 경우 실행)
def rewrite_email(state: EmailState):
    """저자세인 표현을 전문적인 표현으로 수정"""
    print("--- [Node 2] Rewriting Email ---")
    msg = llm.invoke(f"다음 이메일 내용에서 '죄송하다'는 표현을 '양해를 구한다'거나 더 전문적이고 긍정적인 비즈니스 표현으로 바꿔서 다시 작성해줘:\n\n{state['draft']}")
    return {"rewritten_draft": msg.content}

# 노드 3: 번역 (수정 후 실행)
def translate_email(state: EmailState):
    """최종적으로 영어로 번역"""
    print("--- [Node 3] Translating to English ---")
    # rewritten_draft가 있으면 그것을, 없으면 draft를 사용
    source_text = state.get("rewritten_draft", state["draft"])
    msg = llm.invoke(f"다음 내용을 비즈니스 영어 이메일로 번역해줘:\n\n{source_text}")
    return {"final_email": msg.content}

# -------------------------------------
# 워크플로우(Graph) 빌드
# -------------------------------------
workflow = StateGraph(EmailState)

# 노드 추가
workflow.add_node("write_draft", write_draft)
workflow.add_node("rewrite_email", rewrite_email)
workflow.add_node("translate_email", translate_email)

# 엣지(연결) 정의
workflow.add_edge(START, "write_draft")

# 조건부 분기 설정
workflow.add_conditional_edges(
    "write_draft",     # 시작 노드
    check_tone,        # 분기 판단 함수
    {
        "Fail": "rewrite_email", # 검사 통과 못함 -> 수정 노드로 이동
        "Pass": END              # 검사 통과 -> 바로 종료
    }
)

# 수정 후에는 번역 노드로 이동
workflow.add_edge("rewrite_email", "translate_email")
# 번역 후 종료
workflow.add_edge("translate_email", END)

# 컴파일
app = workflow.compile()

# -------------------------------------
# 실행 및 결과 출력
# -------------------------------------

# 1. 머메이드 그래프 출력
print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :") #사이트 들어가서 코드 붙여넣기
print("Mermaid Graph Code:")
print(app.get_graph().draw_mermaid())
print("\n" + "="*30 + "\n")

# 2. 케이스 A: 사과 내용이 포함될 가능성이 높은 주제 (분기: Fail -> Rewrite -> Translate)
print(">>> Case A: 배송 지연 (수정 로직 타게 됨)")
state_a = app.invoke({"topic": "배송이 3일 지연되어 고객에게 알림"})

print(f"\n[Initial Draft]:\n{state_a['draft']}")
if 'final_email' in state_a:
    print(f"\n[Improved & Translated]:\n{state_a['final_email']}")
else:
    print("\n[Ended without modification]")

print("\n" + "="*30 + "\n")

# 3. 케이스 B: 단순 공지 (분기: Pass -> END)
print(">>> Case B: 사내 워크숍 경품 당첨 안내 (수정 없이 종료)")
state_b = app.invoke({"topic": "사내 워크숍 경품 당첨 안내"})

print(f"\n[Initial Draft]:\n{state_b['draft']}")
if 'final_email' in state_b:
    print(f"\n[Improved & Translated]:\n{state_b['final_email']}")
else:
    print("\n[Ended without modification]")