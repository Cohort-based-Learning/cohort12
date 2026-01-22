# 기본 설정
import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

# .env 파일 로드
load_dotenv(r"C:\\Users\\301\\dev\\study26\\path\\to\\your\\app\\.env")

# Gemini 모델 설정
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# ----------------------------------------------------------------------
# 공통 State 정의
# ----------------------------------------------------------------------
class UniversalState(TypedDict):
    topic: str              # 입력 주제
    output_1: str           # 첫 번째 병렬 작업 결과
    output_2: str           # 두 번째 병렬 작업 결과
    output_3: str           # 세 번째 병렬 작업 결과
    final_output: str       # 최종 취합 결과

# ----------------------------------------------------------------------
# 함수 정의
# ----------------------------------------------------------------------

# 1. 여행 계획 어시스턴트
def node_travel_1(state: UniversalState):
    msg = llm.invoke(f"{state['topic']} 여행 시 꼭 가봐야 할 인기 관광지 3곳을 추천해줘.")
    return {"output_1": f"[관광지]\n{msg.content}"}

def node_travel_2(state: UniversalState):
    msg = llm.invoke(f"{state['topic']} 맛집 3곳을 추천해줘.")
    return {"output_2": f"[맛집]\n{msg.content}"}

def node_travel_3(state: UniversalState):
    msg = llm.invoke(f"{state['topic']} 숙소 위치 3곳을 추천해줘.")
    return {"output_3": f"[숙소]\n{msg.content}"}

def agg_travel(state: UniversalState):
    prompt = f"""
    {state['topic']} 여행 정보를 정리해줘.
    {state['output_1']}\n{state['output_2']}\n{state['output_3']}
    위 내용을 바탕으로 1일차 여행 코스를 제안해줘.
    """
    msg = llm.invoke(prompt)
    return {"final_output": msg.content}

# 2. SNS 마케팅 패키지
def node_sns_1(state: UniversalState):
    msg = llm.invoke(f"'{state['topic']}' 제품에 어울리는 인스타그램 감성 캡션과 해시태그를 작성해줘.")
    return {"output_1": f"[인스타그램]\n{msg.content}"}

def node_sns_2(state: UniversalState):
    msg = llm.invoke(f"'{state['topic']}' 제품 리뷰 블로그 포스팅 초안을 작성해줘.")
    return {"output_2": f"[블로그]\n{msg.content}"}

def node_sns_3(state: UniversalState):
    msg = llm.invoke(f"'{state['topic']}' 홍보를 위한 유튜브 숏츠 대본을 작성해줘.")
    return {"output_3": f"[유튜브]\n{msg.content}"}

def agg_sns(state: UniversalState):
    return {"final_output": f"--- SNS 마케팅 패키지 ---\n\n{state['output_1']}\n\n{state['output_2']}\n\n{state['output_3']}"}

# 3. 토론/논쟁 판결기
def node_debate_1(state: UniversalState):
    msg = llm.invoke(f"주제: '{state['topic']}'에 대해 찬성하는 입장의 강력한 논거 3가지를 말해줘.")
    return {"output_1": f"[찬성]\n{msg.content}"}

def node_debate_2(state: UniversalState):
    msg = llm.invoke(f"주제: '{state['topic']}'에 대해 반대하는 입장의 강력한 논거 3가지를 말해줘.")
    return {"output_2": f"[반대]\n{msg.content}"}

def node_debate_3(state: UniversalState):
    msg = llm.invoke(f"주제: '{state['topic']}'와 관련된 객관적 팩트나 통계 자료를 찾아줘.")
    return {"output_3": f"[팩트체크]\n{msg.content}"}

def agg_debate(state: UniversalState):
    prompt = f"""
    주제: {state['topic']}
    {state['output_1']}\n{state['output_2']}\n{state['output_3']}
    양측의 의견과 팩트를 종합하여 이 논쟁에 대한 최종 판결과 요약을 해줘.
    """
    msg = llm.invoke(prompt)
    return {"final_output": msg.content}

# 4. 학습 튜터
def node_study_1(state: UniversalState):
    msg = llm.invoke(f"'{state['topic']}' 개념을 초등학생도 이해하게 쉽게 설명해줘.")
    return {"output_1": f"[쉬운 설명]\n{msg.content}"}

def node_study_2(state: UniversalState):
    msg = llm.invoke(f"'{state['topic']}' 개념을 전공자 수준으로 깊이 있게 설명해줘.")
    return {"output_2": f"[전문 설명]\n{msg.content}"}

def node_study_3(state: UniversalState):
    msg = llm.invoke(f"'{state['topic']}' 이해도를 확인하기 위한 객관식 퀴즈 3문제를 만들어줘.")
    return {"output_3": f"[퀴즈]\n{msg.content}"}

def agg_study(state: UniversalState):
    return {"final_output": f"--- 학습 자료: {state['topic']} ---\n\n{state['output_1']}\n\n{state['output_2']}\n\n{state['output_3']}"}

# 5. 건강 식단 컨설턴트
def node_diet_1(state: UniversalState):
    msg = llm.invoke(f"사용자 특성: '{state['topic']}'. 영양 밸런스를 맞춘 하루 식단을 추천해줘.")
    return {"output_1": f"[식단]\n{msg.content}"}

def node_diet_2(state: UniversalState):
    msg = llm.invoke(f"사용자 특성: '{state['topic']}'. 무리가 가지 않는 운동 루틴을 추천해줘.")
    return {"output_2": f"[운동]\n{msg.content}"}

def node_diet_3(state: UniversalState):
    msg = llm.invoke(f"사용자 특성: '{state['topic']}'. 건강을 위해 피해야 할 음식이나 습관을 알려줘.")
    return {"output_3": f"[주의사항]\n{msg.content}"}

def agg_diet(state: UniversalState):
    prompt = f"""
    {state['topic']} 사용자를 위한 건강 가이드 리포트를 작성해줘.
    {state['output_1']}\n{state['output_2']}\n{state['output_3']}
    """
    msg = llm.invoke(prompt)
    return {"final_output": msg.content}

# 6. 선물 추천 코디네이터
def node_gift_1(state: UniversalState):
    msg = llm.invoke(f"'{state['topic']}'에게 줄 감동적인(정성 위주) 선물을 추천해줘.")
    return {"output_1": f"[감동형]\n{msg.content}"}

def node_gift_2(state: UniversalState):
    msg = llm.invoke(f"'{state['topic']}'에게 줄 실용적인 선물을 추천해줘.")
    return {"output_2": f"[실용형]\n{msg.content}"}

def node_gift_3(state: UniversalState):
    msg = llm.invoke(f"'{state['topic']}'에게 줄 이색적이고 특별한 경험 선물을 추천해줘.")
    return {"output_3": f"[이색형]\n{msg.content}"}

def agg_gift(state: UniversalState):
    return {"final_output": f"--- 선물 추천 리스트 ---\n대상: {state['topic']}\n\n{state['output_1']}\n\n{state['output_2']}\n\n{state['output_3']}"}

# 7. 뉴스 인사이트 분석
def node_news_1(state: UniversalState):
    msg = llm.invoke(f"뉴스 주제: '{state['topic']}'. 이것이 경제 시장(주식, 부동산 등)에 미칠 영향을 분석해줘.")
    return {"output_1": f"[경제적 영향]\n{msg.content}"}

def node_news_2(state: UniversalState):
    msg = llm.invoke(f"뉴스 주제: '{state['topic']}'. 일반 소비자 가계에 미칠 영향을 분석해줘.")
    return {"output_2": f"[소비자 영향]\n{msg.content}"}

def node_news_3(state: UniversalState):
    msg = llm.invoke(f"뉴스 주제: '{state['topic']}'. 관련 기업들에게 미칠 영향을 분석해줘.")
    return {"output_3": f"[기업 영향]\n{msg.content}"}

def agg_news(state: UniversalState):
    prompt = f"""
    뉴스 주제: {state['topic']}
    {state['output_1']}\n{state['output_2']}\n{state['output_3']}
    위 분석들을 종합하여 핵심 인사이트 리포트를 작성해줘.
    """
    msg = llm.invoke(prompt)
    return {"final_output": msg.content}


# ----------------------------------------------------------------------
# Workflow Builder 함수
# ----------------------------------------------------------------------
def create_workflow(nodes, aggregator_func):
    builder = StateGraph(UniversalState)
    builder.add_node("branch_1", nodes[0])
    builder.add_node("branch_2", nodes[1])
    builder.add_node("branch_3", nodes[2])
    builder.add_node("aggregator", aggregator_func)
    builder.add_edge(START, "branch_1")
    builder.add_edge(START, "branch_2")
    builder.add_edge(START, "branch_3")
    builder.add_edge("branch_1", "aggregator")
    builder.add_edge("branch_2", "aggregator")
    builder.add_edge("branch_3", "aggregator")
    builder.add_edge("aggregator", END)
    return builder.compile()

# ----------------------------------------------------------------------
# Graph Export (LangGraph Server용)
# ----------------------------------------------------------------------
graph_travel = create_workflow([node_travel_1, node_travel_2, node_travel_3], agg_travel)
graph_sns = create_workflow([node_sns_1, node_sns_2, node_sns_3], agg_sns)
graph_debate = create_workflow([node_debate_1, node_debate_2, node_debate_3], agg_debate)
graph_study = create_workflow([node_study_1, node_study_2, node_study_3], agg_study)
graph_diet = create_workflow([node_diet_1, node_diet_2, node_diet_3], agg_diet)
graph_gift = create_workflow([node_gift_1, node_gift_2, node_gift_3], agg_gift)
graph_news = create_workflow([node_news_1, node_news_2, node_news_3], agg_news)

# ----------------------------------------------------------------------
# Master Router Graph (전체 통합 그래프)
# ----------------------------------------------------------------------
class MasterState(TypedDict):
    topic: str
    category: str      # 라우팅 카테고리
    final_output: str

def router_node(state: UniversalState):
    """주제를 분석하여 어떤 작업을 수행할지 결정하는 라우터"""
    prompt = f"""
    사용자의 입력 주제: '{state['topic']}'
    
    이 주제가 다음 중 어디에 가장 적합한지 1, 2, 3, 4, 5, 6, 7 중 하나의 숫자만 답변해.
    
    1. 여행 계획 (도시, 관광지)
    2. SNS 마케팅 (제품 홍보)
    3. 토론/논쟁 (찬반 논거)
    4. 학습 튜터 (개념 설명)
    5. 건강 식단 (식단, 운동)
    6. 선물 추천 (기념일 등)
    7. 뉴스 분석 (경제 영향)
    
    답변은 오직 숫자만.
    """
    msg = llm.invoke(prompt)
    category = msg.content.strip()
    return {"category": category}

def route_decision(state: UniversalState):
    """category 값에 따라 다음 노드 결정"""
    cat = state.get("category", "")
    if "1" in cat: return "go_travel"
    if "2" in cat: return "go_sns"
    if "3" in cat: return "go_debate"
    if "4" in cat: return "go_study"
    if "5" in cat: return "go_diet"
    if "6" in cat: return "go_gift"
    if "7" in cat: return "go_news"
    return END

# Master Graph 빌드
master_builder = StateGraph(UniversalState)
master_builder.add_node("router", router_node)

# 각 서브 그래프를 노드로 추가 (Hierarchical Graph)
master_builder.add_node("run_travel", graph_travel)
master_builder.add_node("run_sns", graph_sns)
master_builder.add_node("run_debate", graph_debate)
master_builder.add_node("run_study", graph_study)
master_builder.add_node("run_diet", graph_diet)
master_builder.add_node("run_gift", graph_gift)
master_builder.add_node("run_news", graph_news)

master_builder.add_edge(START, "router")

# 조건부 엣지 설정
master_builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "go_travel": "run_travel",
        "go_sns": "run_sns",
        "go_debate": "run_debate",
        "go_study": "run_study",
        "go_diet": "run_diet",
        "go_gift": "run_gift",
        "go_news": "run_news"
    }
)

# 각 서브 그래프 실행 후 종료
master_builder.add_edge("run_travel", END)
master_builder.add_edge("run_sns", END)
master_builder.add_edge("run_debate", END)
master_builder.add_edge("run_study", END)
master_builder.add_edge("run_diet", END)
master_builder.add_edge("run_gift", END)
master_builder.add_edge("run_news", END)

graph_master = master_builder.compile()

# ----------------------------------------------------------------------
# Main Execution (CLI Menu)
# ----------------------------------------------------------------------
def main():
    programs = {
        "1": ("여행 계획 어시스턴트", graph_travel),
        "2": ("SNS 마케팅 패키지", graph_sns),
        "3": ("토론/논쟁 판결기", graph_debate),
        "4": ("학습 튜터", graph_study),
        "5": ("건강 식단 컨설턴트", graph_diet),
        "6": ("선물 추천 코디네이터", graph_gift),
        "7": ("뉴스 인사이트 분석", graph_news),
    }

    print("\n" + "="*50)
    print("LangGraph 병렬 처리 프로그램 모음")
    print("="*50)
    for key, (name, _) in programs.items():
        print(f"{key}. {name}")
    print("="*50)

    choice = input("실행할 프로그램 번호를 입력하세요: ")
    
    if choice not in programs:
        print("잘못된 번호입니다.")
        sys.exit()

    name, workflow = programs[choice]
    topic = input(f"\n[{name}] 주제를 입력하세요: ")

    print(f"\n'{name}' 실행 중... (병렬 처리 시작)\n")
    
    result = workflow.invoke({"topic": topic})
    
    print("\n" + "="*50)
    print(f"결과 보고서: {name} ✨")
    print("="*50)
    print(result['final_output'])

if __name__ == "__main__":
    main()