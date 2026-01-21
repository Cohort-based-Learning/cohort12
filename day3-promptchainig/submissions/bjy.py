
import os
import sys
from typing import TypedDict, Literal, List
from dotenv import load_dotenv

# -------------------------------------
# 환경 설정
# -------------------------------------
# .env 파일 로드 (지정된 경로)
env_path = r"c:\\Users\\301\\dev\\study26\\path\\to\\your\\app\\.env"
load_dotenv(env_path)

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# LLM 설정 (Gemini)
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    sys.exit(1)


# -------------------------------------
# 통합 State 정의
# -------------------------------------
class State(TypedDict):
    # 공통 입력 및 제어
    user_input: str       # 사용자가 입력한 초기 요청/주제
    task_type: int        # 1~6번 메뉴 선택
    step_count: int       # 루프 제한용 카운터 (쇼핑 방어전 등)
    
    # 공통 데이터 흐름
    draft: str            # 1차 생성물 (Generate 결과)
    critique: str         # 평가/심사 결과 (Check 결과)
    feedback: str         # 구체적인 개선 피드백 내용
    improved: str         # 개선된 결과물 (Improve 결과)
    final_output: str     # 최종 결과물 (Polish/Extract 결과)

    # 병렬 생성용 (코드 리뷰 등)
    option_a: str
    option_b: str
    option_c: str


# -------------------------------------
# 공통 도우미 함수 (노드 로직)
# -------------------------------------

# === 1. Cold Email Generator Nodes ===
def email_generate(state: State):
    msg = llm.invoke(f"다음 상황에 맞는 콜드 메일 초안을 작성해줘:\n{state['user_input']}")
    return {"draft": msg.content}

def email_check(state: State):
    # 평가: 예의/훅 체크
    msg = llm.invoke(f"다음 콜드 메일 초안을 평가해줘. 너무 딱딱하거나 훅이 약하면 'Fail', 충분히 공손하고 매력적이면 'Pass'라고만 답해줘.\n\n[초안]\n{state['draft']}")
    result = "Pass" if "Pass" in msg.content else "Fail"
    print(f"\n[Check Result]: {result}\n[Critique]: {msg.content}")
    return {"critique": result, "feedback": msg.content if result == "Fail" else ""}

def email_improve(state: State):
    msg = llm.invoke(f"다음 피드백을 반영하여 메일을 더 공손하고 매력적인 톤으로 수정해줘:\n\n[피드백]\n{state['feedback']}\n\n[원본]\n{state['draft']}")
    return {"improved": msg.content}

def email_polish(state: State):
    source = state.get("improved") if state.get("improved") else state["draft"]
    msg = llm.invoke(f"다음 메일의 마지막에 위트 있는 추신(P.S.)이나 행동 유도(Call To Action)를 추가하여 마무리해줘:\n\n{source}")
    return {"final_output": msg.content}


# === 2. Code Review Nodes ===
def code_generate(state: State):
    msg = llm.invoke(f"다음 파이썬 코드에 대한 상세한 코드 리뷰(버그, 비효율성 지적)를 해줘:\n```python\n{state['user_input']}\n```")
    return {"draft": msg.content} # 여기서는 draft가 리뷰 내용

def code_parallel_improve(state: State):
    # 병렬 생성을 흉내내거나 실제로 병렬 호출 (여기서는 순차 호출하지만 State에 저장)
    # 실제 병렬 실행은 LangGraph에서 분기시켜야 하지만, 구현 편의상 하나의 노드에서 3번 호출
    code = state['user_input']
    review = state['draft']
    
    opt_a = llm.invoke(f"다음 리뷰를 참고하여 '성능 최적화'에 집중한 수정 코드를 짜줘:\n{review}\n\n[코드]\n{code}")
    opt_b = llm.invoke(f"다음 리뷰를 참고하여 '가독성'에 집중한 수정 코드를 짜줘:\n{review}\n\n[코드]\n{code}")
    opt_c = llm.invoke(f"다음 리뷰를 참고하여 'Pythonic하고 짧은(숏코딩)' 수정 코드를 짜줘:\n{review}\n\n[코드]\n{code}")
    
    return {"option_a": opt_a.content, "option_b": opt_b.content, "option_c": opt_c.content}

def code_select_polish(state: State):
    # 3가지 중 하나를 선택하고(여기서는 LLM이 추천) Docstring 추가
    selection_prompt = f"""
    다음 3가지 코드 옵션 중 가장 균형 잡히고 좋은 것을 하나 골라서, 
    상세한 주석(Docstring)과 타입 힌트를 추가하여 최종 코드를 완성해줘.
    
    [Option A: 성능]\n{state['option_a']}
    [Option B: 가독성]\n{state['option_b']}
    [Option C: 숏코딩]\n{state['option_c']}
    """
    msg = llm.invoke(selection_prompt)
    return {"final_output": msg.content}


# === 3. Menu Decision Nodes ===
def menu_generate(state: State):
    msg = llm.invoke(f"다음 상황(기분/날씨)에 맞는 점심 메뉴 1개를 추천해줘:\n{state['user_input']}")
    return {"draft": msg.content}

def menu_check(state: State):
    # 가상의 사용자 체크: 칼로리가 높거나 어제 먹은 것과 겹치는지?
    # 여기서는 LLM이 '깐깐한 헬스 트레이너' 페르소나로 빙의해서 판단
    msg = llm.invoke(f"추천된 메뉴: '{state['draft']}'. 이 메뉴가 800kcal 이상이거나 너무 기름지면 'Fail', 건강하고 좋으면 'Pass'라고 답해줘. 이유도 짧게 적어줘.")
    result = "Pass" if "Pass" in msg.content else "Fail"
    print(f"\n[Check Result]: {result}\n[Critique]: {msg.content}")
    return {"critique": result, "feedback": msg.content}

def menu_improve(state: State):
    msg = llm.invoke(f"이전 추천('{state['draft']}')이 다음 이유로 거절당했어: '{state['feedback']}'. 더 건강하거나 새로운 메뉴로 다시 추천해줘.")
    return {"improved": msg.content}

def menu_polish(state: State):
    menu = state.get("improved", state["draft"])
    msg = llm.invoke(f"메뉴 '{menu}'를 파는 맛집을 찾는 꿀팁이나, 집에서 해먹을 수 있는 아주 간단한 레시피 팁을 한 줄 덧붙여줘.")
    return {"final_output": f"추천 메뉴: {menu}\n\n Tip: {msg.content}"}


# === 4. Instagram Caption Nodes ===
def insta_generate(state: State):
    msg = llm.invoke(f"이 사진 분위기('{state['user_input']}')에 어울리는 인스타그램 캡션을 작성해줘. 여러 가지 옵션을 주지 말고, 가장 어울리는 '딱 하나'의 완성된 글만 작성해. 설명이나 서론/결론 없이 캡션 내용만 출력해. 말투는 감성 힙스터 느낌으로.")
    print(f"\n[Step 1: Generated Draft]\n{msg.content}")
    return {"draft": msg.content}

def insta_check(state: State):
    msg = llm.invoke(f"다음 캡션을 평가해줘. 해시태그가 없거나 이모지가 부족하면 'Fail', 충분하면 'Pass'라고만 답해줘.\n\n[캡션]\n{state['draft']}")
    result = "Pass" if "Pass" in msg.content else "Fail"
    print(f"\n[Step 2: Check Result]: {result}")
    return {"critique": result}

def insta_improve(state: State):
    msg = llm.invoke(f"""이전 캡션이 너무 밋밋하거나 아재 같아. 요즘 MZ 세대가 쓰는 힙한 말투, 유행하는 밈(Meme), 귀여운 이모지, 그리고 센스 있는 해시태그들을 듬뿍 추가해서 완전히 '인스타 감성'으로 바꿔줘.
    
    [주의사항]
    1. 원본 글의 **상황과 핵심 내용(장소, 분위기, 행동 등)**은 절대 바꾸지 말고 그대로 유지해. 없는 내용을 지어내지 마.
    2. 줄바꿈을 많이 써서 가독성을 높여줘.
    
    [원본]
    {state['draft']}""")
    print(f"\n[Step 3: Improved Version]\n{msg.content}")
    return {"improved": msg.content}

def insta_polish(state: State):
    source = state.get("improved", state["draft"])
    msg = llm.invoke(f"다음 글의 힙한 분위기를 유지하면서, 마지막에 팔로워들과 소통할 수 있는 가벼운 질문 하나를 툭 던지듯이 추가해줘 (너무 딱딱하지 않게, '너네는 어때?' 같은 느낌으루):\n\n{source}")
    return {"final_output": msg.content}


# === 5. Shopping Defender Nodes (Loop) ===
def shop_generate(state: State):
    msg = llm.invoke(f"사용자가 '{state['user_input']}'를 사고 싶대. 일단은 '왜요? 돈 아깝지 않아요?' 하는 식으로 가볍게 딴지를 걸면서 시작해. 무조건적인 공감보다는 회의적인 반응을 보여줘. (절대 사용자에게 질문하지 마. 그냥 네 의견만 말해)")
    print(f"\n[AI의 첫 번째 반응]: {msg.content}")
    return {"draft": msg.content, "step_count": 0}

def shop_check(state: State):
    count = state["step_count"]
    
    # 첫 진입(count=0)이거나, 이미 반복중일 때 사용자에게 물어봄
    if count >= 10:
        print("\n(시스템: AI가 지쳤습니다. 당신의 승리입니다.)")
        return {"critique": "Pass", "feedback": "그래 졌다. 사라 사.", "step_count": count}

    print(f"\n[시스템: 현재 {count + 1}번째 방어 중...]")
    print(f"AI의 말을 듣고 마음이 바뀌셨나요? (구매 포기하려면 'Y', 그래도 산다면 'N' 입력)")
    user_decision = input(">> ").strip().upper()

    if user_decision == 'Y':
        return {"critique": "Pass", "step_count": count}
    else:
        # N을 누르면 다시 공격(Fail 반환)
        return {"critique": "Fail", "step_count": count + 1}

def shop_improve(state: State):
    # 팩트 폭격
    msg = llm.invoke(f"""사용자가 '{state['user_input']}'를 사겠다고 고집을 피우고 있어.
    절대 추천하거나 질문하지 마. 오로지 '사지 말아야 할 현실적인 이유'만 들어서 공격해.
    경제적 낭비, 활용도 저하, 대체품 존재, 순식간에 질림 등 팩트로 뼈를 때려줘.
    말투는 단호박처럼 단호하게, "그거 사봤자 3일이면 먼지 쌓입니다." 처럼 말해.
    (주의: 헤더나 서론 없이 바로 본론만 말해)""")
    print(f"\n[AI의 팩트 폭격]: {msg.content}")
    return {"improved": msg.content}

def shop_polish(state: State):
    msg = llm.invoke(f"사용자가 구매를 포기했거나, 우리가 허락했어. '{state['user_input']}'와 관련해서 합리적인 소비 팁이나 저축 조언을 해주며 훈훈하게 마무리해줘.")
    final_msg = f"{state.get('improved', '')}\n\n[결론]: {msg.content}"
    return {"final_output": final_msg}


# === 6. Intent Extractor Nodes (Direct) ===
def intent_analyze(state: State):
    msg = llm.invoke(f"사용자 발화: '{state['user_input']}'\n이 발화의 의도가 '상품 검색'인지 분석해줘. 상품 검색이면 'Search', 아니면 의도를 짧게 요약해줘.")
    return {"draft": msg.content}

def intent_check(state: State):
    # Search 의도이고, 대상이 명확한가?
    msg = llm.invoke(f"분석 결과: '{state['draft']}'. 사용자 발화: '{state['user_input']}'.\n사용자가 찾으려는 상품(Target)과 속성(Feature)이 명확한가? 명확하면 'Pass', 불명확하거나 상품 검색이 아니면 'Fail'이라고 답해줘.")
    result = "Pass" if "Pass" in msg.content else "Fail"
    return {"critique": result}

def intent_improve(state: State):
    msg = llm.invoke(f"사용자의 의도나 찾으려는 상품이 불명확해. 사용자에게 구체적으로 무엇을 원하는지 되물어보는 질문을 만들어줘:\n'{state['user_input']}'")
    return {"final_output": f"질문이 필요합니다: {msg.content}"} # 여기서 종료 시킴 (Fail 분기)

def intent_extract(state: State):
    # Pass 시 실행: JSON 추출
    schema = """{
    "intent": "product_search",
    "main_keyword": "핵심 검색어(예: 자동 변기 세정기)",
    "related_keywords": ["동의어", "유사 표현", "관련 도구(예: 변기솔, 세정볼)"],
    "specific_features": ["구체적 속성1(예: 변기 내부 부착)", "구체적 속성2(예: 젤 타입)"],
    "user_need": "사용자가 해결하고자 하는 구체적인 문제(예: 솔질 없이 변기 때를 없애고 싶음)"
}
"""
    msg = llm.invoke(f"사용자 발화: '{state['user_input']}'\n위 내용에서 검색에 필요한 정보를 풍부하게 확장하여 추출해서 다음 JSON 형식으로 출력해줘(마크다운 코드블록 없이 순수 JSON만). related_keywords에는 동의어나 함께 검색될만한 구체적인 제품명을 포함해줘:\n{schema}")
    return {"final_output": msg.content}


# -------------------------------------
# 그래프 빌더 함수
# -------------------------------------
def build_graph(task_num: int):
    workflow = StateGraph(State)

    if task_num == 1: # Cold Email
        workflow.add_node("generate", email_generate)
        workflow.add_node("check", email_check)
        workflow.add_node("improve", email_improve)
        workflow.add_node("polish", email_polish)
        
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "check")
        workflow.add_conditional_edges("check", lambda x: x["critique"], {"Pass": "polish", "Fail": "improve"})
        workflow.add_edge("improve", "polish")
        workflow.add_edge("polish", END)

    elif task_num == 2: # Code Review
        workflow.add_node("generate", code_generate)
        workflow.add_node("improve_parallel", code_parallel_improve)
        workflow.add_node("polish", code_select_polish)
        
        # Check 단계 생략하고 바로 리뷰 -> 병렬 개선 -> 폴리싱 (흐름 변형)
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "improve_parallel")
        workflow.add_edge("improve_parallel", "polish")
        workflow.add_edge("polish", END)

    elif task_num == 3: # Menu
        workflow.add_node("generate", menu_generate)
        workflow.add_node("check", menu_check)
        workflow.add_node("improve", menu_improve)
        workflow.add_node("polish", menu_polish)
        
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "check")
        workflow.add_conditional_edges("check", lambda x: x["critique"], {"Pass": "polish", "Fail": "improve"})
        workflow.add_edge("improve", "polish")
        workflow.add_edge("polish", END)

    elif task_num == 4: # Instagram
        workflow.add_node("generate", insta_generate)
        workflow.add_node("check", insta_check)
        workflow.add_node("improve", insta_improve)
        workflow.add_node("polish", insta_polish)
        
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "check")
        workflow.add_conditional_edges("check", lambda x: x["critique"], {"Pass": "polish", "Fail": "improve"})
        workflow.add_edge("improve", "polish")
        workflow.add_edge("polish", END)

    elif task_num == 5: # Shopping Loop
        workflow.add_node("generate", shop_generate)
        workflow.add_node("check", shop_check) # 여기서 count 증가
        workflow.add_node("improve", shop_improve)
        workflow.add_node("polish", shop_polish)

        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "check")
        workflow.add_conditional_edges("check", lambda x: x["critique"], {"Pass": "polish", "Fail": "improve"})
        workflow.add_edge("improve", "check") # Loop Back!
        workflow.add_edge("polish", END)

    elif task_num == 6: # Intent Extraction
        workflow.add_node("analyze", intent_analyze)
        workflow.add_node("check", intent_check)
        workflow.add_node("extract", intent_extract)
        workflow.add_node("ask_more", intent_improve) # 질문 생성

        workflow.add_edge(START, "analyze")
        workflow.add_edge("analyze", "check")
        
        # Pass -> Extract -> END
        # Fail -> Ask More -> END
        workflow.add_conditional_edges("check", lambda x: x["critique"], {"Pass": "extract", "Fail": "ask_more"})
        workflow.add_edge("extract", END)
        workflow.add_edge("ask_more", END)

    return workflow.compile()


# -------------------------------------
# 메인 실행 루프
# -------------------------------------
def main():
    print("========================================")
    print("Prompt Chaining Labs (Powered by Gemini)")
    print("========================================")
    print("1. 콜드 메일 작성기")
    print("2. 코드 리뷰 및 리팩토링 봇")
    print("3. 결정 장애 해결사 (점심 메뉴)")
    print("4. 인스타그램 캡션 요리사")
    print("5. 쇼핑 충동 구매 방어전")
    print("6. 의도 파악 및 의미 추출기")
    print("========================================")

    try:
        choice = int(input("원하는 기능을 선택하세요 (1-6): "))
        if choice < 1 or choice > 6:
            print("잘못된 선택입니다.")
            return
    except ValueError:
        print("숫자를 입력해주세요.")
        return

    # 입력 프롬프트 설정
    prompts = {
        1: "누구에게, 어떤 목적으로 메일을 보내시나요? (예: 투자자에게 우리 스타트업 소개)",
        2: "리뷰할 파이썬 코드를 붙여넣어 주세요:",
        3: "오늘 기분이나 날씨, 특별히 땡기는 맛이 있나요? (예: 비 오는데 뜨끈한 국물)",
        4: "인스타그램에 올릴 사진의 상황을 묘사해주세요.감성적인 글과 해시태그를 만들어 드려요!\n(예: '오랜만에 친구들과 한강 피크닉 와서 치맥 먹는 중! 날씨 너무 좋다')",
        5: "사고 싶은 물건은 무엇인가요? (예: 아이패드 프로 6세대)",
        6: "찾으시는 상품이나 상황을 말씀해주세요: (예: 손시려운데 따뜻하게 하는거 찾아줘)"
    }

    if choice == 2:
        print(f"\n[입력]: {prompts[choice]}")
        print("(입력을 마치려면 줄바꿈 후 'END'를 입력하고 엔터를 치세요)")
        
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "END":
                break
            lines.append(line)
        user_input = "\n".join(lines)
    else:
        user_input = input(f"\n[입력]: {prompts[choice]}\n>> ")
    
    # State 초기화
    initial_state = State(
        user_input=user_input, 
        task_type=choice, 
        step_count=0,
        draft="",
        critique="",
        feedback="",
        improved="",
        final_output="",
        option_a="",
        option_b="",
        option_c=""
    )

    print(f"\n {choice}번 기능 실행 중... (Gemini가 생각하고 있습니다)")
    
    # 그래프 빌드 및 실행
    app = build_graph(choice)
    result_state = app.invoke(initial_state)

    print("\n================ [결과] ================")
    if choice == 2:
        # 코드 리뷰는 폴리싱 전 선택 단계가 있을 수 있으나, 여기서는 최종 polished가 저장된 걸로 가정
        print(result_state["final_output"])
    elif choice == 6:
        # 의도 파악은 JSON 코드블록 출력
        print(result_state["final_output"])
    else:
        print(result_state["final_output"])
    print("========================================")

# === For LangGraph Studio Visualization ===
# Studio는 컴파일된 그래프 객체를 직접 가리켜야 하므로, 미리 생성해둡니다.
# (CLI 실행 시에는 영향 없음)
graph_1_email = build_graph(1)
graph_2_code = build_graph(2)
graph_3_menu = build_graph(3)
graph_4_insta = build_graph(4)
graph_5_shop = build_graph(5)
graph_6_intent = build_graph(6)

# === Master Graph (For Full Visualization) ===
def build_master_graph():
    workflow = StateGraph(State)
    
    # 각 서브그래프를 노드로 등록
    workflow.add_node("email_flow", graph_1_email)
    workflow.add_node("code_flow", graph_2_code)
    workflow.add_node("menu_flow", graph_3_menu)
    workflow.add_node("insta_flow", graph_4_insta)
    workflow.add_node("shop_flow", graph_5_shop)
    workflow.add_node("intent_flow", graph_6_intent)

    # 라우팅 로직 (초기 분기)
    def route_tasks(state: State):
        task_map = {
            1: "email_flow",
            2: "code_flow",
            3: "menu_flow",
            4: "insta_flow",
            5: "shop_flow",
            6: "intent_flow"
        }
        return task_map.get(state["task_type"])

    # 시각화를 위해 라우팅 맵을 명시적으로 전달
    workflow.add_conditional_edges(
        START, 
        route_tasks,
        {
            "email_flow": "email_flow",
            "code_flow": "code_flow", 
            "menu_flow": "menu_flow",
            "insta_flow": "insta_flow",
            "shop_flow": "shop_flow",
            "intent_flow": "intent_flow"
        }
    )
    
    # 모든 플로우는 종료(END)로 연결
    workflow.add_edge("email_flow", END)
    workflow.add_edge("code_flow", END)
    workflow.add_edge("menu_flow", END)
    workflow.add_edge("insta_flow", END)
    workflow.add_edge("shop_flow", END)
    workflow.add_edge("intent_flow", END)

    return workflow.compile()

graph_master = build_master_graph()

if __name__ == "__main__":
    main()