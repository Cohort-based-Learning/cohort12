# ============================================================================
# 🎧 감정 기반 음악/명언 추천 시스템
# Day5 Routing + Day2~4 학습 내용 통합
# ============================================================================
# 학습 포인트:
# - Day2: 구조화 출력 (with_structured_output), 도구 바인딩 (bind_tools)
# - Day3: Prompt Chaining, StateGraph 기본
# - Day4: 병렬 처리 (Fan-out/Fan-in), Aggregator 패턴
# - Day5: 조건부 라우팅 (add_conditional_edges)
# ============================================================================

import os
import random
from pathlib import Path
from dotenv import load_dotenv
from typing_extensions import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# -----------------------------------------------------------------------------
# 환경 설정
# -----------------------------------------------------------------------------
env_path = Path(__file__).resolve().parents[1].parent / 'path' / 'to' / 'your' / 'app' / '.env'
print(f"[Debug] .env 경로: {env_path}")
load_dotenv(dotenv_path=env_path)

# LangSmith 트레이싱 활성화
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "day5-emotion-router"

# Gemini 모델 설정
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# =============================================================================
# [Day2 학습] 구조화 출력 - 감정 분류기
# =============================================================================
class EmotionRoute(BaseModel):
    """LLM이 반드시 정해진 감정 중 하나를 선택하도록 강제"""
    emotion: Literal["happy", "sad", "angry", "tired"] = Field(
        ..., 
        description="사용자의 입력에서 감지된 주요 감정 (happy/sad/angry/tired 중 하나)"
    )
    reason: str = Field(
        ..., 
        description="해당 감정으로 분류한 이유 (한 문장)"
    )

# 구조화 출력이 적용된 라우터 LLM
emotion_router = llm.with_structured_output(EmotionRoute)

# =============================================================================
# [Day2 학습] 도구 정의 - 랜덤 선택기
# =============================================================================
def pick_one(candidates: list[str]) -> str:
    """주어진 목록에서 하나를 랜덤으로 선택합니다."""
    return random.choice(candidates)

# 도구가 바인딩된 LLM
llm_with_tools = llm.bind_tools([pick_one])

# =============================================================================
# State 정의
# =============================================================================
class EmotionState(TypedDict):
    user_input: str           # 사용자 입력 (기분 설명)
    emotion: str              # 분류된 감정
    emotion_reason: str       # 분류 이유
    music_recommendation: str # 음악 추천 결과
    quote_recommendation: str # 명언 추천 결과
    advice: str               # 오늘의 조언
    final_output: str         # 최종 통합 결과

# =============================================================================
# 노드 정의
# =============================================================================

# -----------------------------------------------------------------------------
# 라우터 노드: 감정 분류 (Day5 핵심)
# -----------------------------------------------------------------------------
def emotion_classifier(state: EmotionState):
    """사용자 입력을 분석하여 감정을 분류합니다."""
    result = emotion_router.invoke(
        f"""다음 사용자의 말에서 감정을 분석해주세요.
        
사용자: "{state['user_input']}"

- happy: 기쁘거나 행복하거나 신나는 상태
- sad: 슬프거나 우울하거나 외로운 상태  
- angry: 화나거나 짜증나거나 답답한 상태
- tired: 피곤하거나 지치거나 무기력한 상태

가장 강하게 느껴지는 감정 하나만 선택하세요."""
    )
    print(f"\n🎭 감정 분석 결과: {result.emotion} ({result.reason})")
    return {"emotion": result.emotion, "emotion_reason": result.reason}

# -----------------------------------------------------------------------------
# 병렬 노드들: 각 감정별 추천 생성 (Day4 패턴)
# -----------------------------------------------------------------------------

# 🎵 음악 추천 노드 (감정별 분기)
def recommend_music_happy(state: EmotionState):
    msg = llm.invoke("기분이 좋을 때 들으면 더 신나는 K-POP 노래 3곡을 추천해줘. 곡명과 아티스트, 한 줄 설명으로.")
    return {"music_recommendation": f"🎵 신나는 플레이리스트\n{msg.content}"}

def recommend_music_sad(state: EmotionState):
    msg = llm.invoke("우울할 때 위로가 되는 잔잔한 발라드 3곡을 추천해줘. 곡명과 아티스트, 한 줄 설명으로.")
    return {"music_recommendation": f"🎵 위로의 플레이리스트\n{msg.content}"}

def recommend_music_angry(state: EmotionState):
    msg = llm.invoke("화가 날 때 스트레스 해소되는 강렬한 록/힙합 노래 3곡을 추천해줘. 곡명과 아티스트, 한 줄 설명으로.")
    return {"music_recommendation": f"🎵 스트레스 해소 플레이리스트\n{msg.content}"}

def recommend_music_tired(state: EmotionState):
    msg = llm.invoke("피곤할 때 편안하게 쉴 수 있는 Lo-Fi/재즈 음악 3곡을 추천해줘. 곡명과 아티스트, 한 줄 설명으로.")
    return {"music_recommendation": f"🎵 힐링 플레이리스트\n{msg.content}"}

# 📜 명언 추천 노드 (감정별 분기)
def recommend_quote_happy(state: EmotionState):
    msg = llm.invoke("행복한 순간을 더 특별하게 만들어주는 명언 2개를 추천해줘. 명언과 말한 사람을 포함해서.")
    return {"quote_recommendation": f"📜 오늘의 명언\n{msg.content}"}

def recommend_quote_sad(state: EmotionState):
    msg = llm.invoke("슬플 때 마음을 달래주는 위로의 명언 2개를 추천해줘. 명언과 말한 사람을 포함해서.")
    return {"quote_recommendation": f"📜 위로의 명언\n{msg.content}"}

def recommend_quote_angry(state: EmotionState):
    msg = llm.invoke("화가 날 때 마음을 가라앉히는 명언 2개를 추천해줘. 분노 조절이나 인내에 관한 것으로.")
    return {"quote_recommendation": f"📜 진정의 명언\n{msg.content}"}

def recommend_quote_tired(state: EmotionState):
    msg = llm.invoke("지쳤을 때 다시 힘을 주는 동기부여 명언 2개를 추천해줘. 명언과 말한 사람을 포함해서.")
    return {"quote_recommendation": f"📜 에너지 충전 명언\n{msg.content}"}

# 💡 조언 노드 (감정별 분기)
def give_advice_happy(state: EmotionState):
    msg = llm.invoke("기분 좋은 하루를 더 알차게 보내는 방법을 짧게 조언해줘.")
    return {"advice": f"💡 오늘의 조언\n{msg.content}"}

def give_advice_sad(state: EmotionState):
    msg = llm.invoke("우울한 기분을 달래는 구체적인 방법을 짧게 조언해줘. 공감과 위로를 담아서.")
    return {"advice": f"💡 오늘의 조언\n{msg.content}"}

def give_advice_angry(state: EmotionState):
    msg = llm.invoke("화가 났을 때 진정하고 상황을 해결하는 방법을 짧게 조언해줘.")
    return {"advice": f"💡 오늘의 조언\n{msg.content}"}

def give_advice_tired(state: EmotionState):
    msg = llm.invoke("피곤할 때 효과적으로 에너지를 충전하는 방법을 짧게 조언해줘.")
    return {"advice": f"💡 오늘의 조언\n{msg.content}"}

# -----------------------------------------------------------------------------
# Aggregator 노드: 결과 통합 (Day3/Day4 패턴)
# -----------------------------------------------------------------------------
def aggregate_results(state: EmotionState):
    """모든 추천 결과를 하나로 통합합니다."""
    emotion_emoji = {"happy": "😊", "sad": "😢", "angry": "😠", "tired": "😴"}
    emoji = emotion_emoji.get(state['emotion'], "🎭")
    
    final = f"""
{'='*60}
{emoji} 감정 분석 결과: {state['emotion'].upper()}
{'='*60}
📝 분석: {state['emotion_reason']}
{'='*60}

{state['music_recommendation']}

{'─'*60}

{state['quote_recommendation']}

{'─'*60}

{state['advice']}

{'='*60}
🌟 오늘도 좋은 하루 보내세요!
{'='*60}
"""
    return {"final_output": final}

# =============================================================================
# 조건부 라우팅 함수 (Day5 핵심)
# =============================================================================
def route_by_emotion(state: EmotionState) -> str:
    """감정에 따라 다음 노드 그룹을 결정합니다."""
    emotion = state["emotion"]
    if emotion == "happy":
        return "branch_happy"
    elif emotion == "sad":
        return "branch_sad"
    elif emotion == "angry":
        return "branch_angry"
    elif emotion == "tired":
        return "branch_tired"
    return "branch_happy"  # 기본값

# =============================================================================
# 그래프 빌드 (Day5 Routing + Day4 Parallelization)
# =============================================================================
builder = StateGraph(EmotionState)

# 1. 라우터 노드 추가
builder.add_node("emotion_classifier", emotion_classifier)

# 2. 감정별 병렬 처리 노드들 추가 (각 감정 분기마다 3개 병렬)
# Happy 분기
builder.add_node("music_happy", recommend_music_happy)
builder.add_node("quote_happy", recommend_quote_happy)
builder.add_node("advice_happy", give_advice_happy)

# Sad 분기
builder.add_node("music_sad", recommend_music_sad)
builder.add_node("quote_sad", recommend_quote_sad)
builder.add_node("advice_sad", give_advice_sad)

# Angry 분기
builder.add_node("music_angry", recommend_music_angry)
builder.add_node("quote_angry", recommend_quote_angry)
builder.add_node("advice_angry", give_advice_angry)

# Tired 분기
builder.add_node("music_tired", recommend_music_tired)
builder.add_node("quote_tired", recommend_quote_tired)
builder.add_node("advice_tired", give_advice_tired)

# 3. Aggregator 노드 추가
builder.add_node("aggregator", aggregate_results)

# =============================================================================
# 엣지 연결
# =============================================================================
# START → 감정 분류기
builder.add_edge(START, "emotion_classifier")

# 감정 분류기 → 조건부 분기 (Day5 핵심!)
builder.add_conditional_edges(
    "emotion_classifier",
    route_by_emotion,
    {
        "branch_happy": "music_happy",
        "branch_sad": "music_sad",
        "branch_angry": "music_angry",
        "branch_tired": "music_tired",
    }
)

# Happy 분기: 병렬 실행 후 Aggregator로
builder.add_edge("music_happy", "quote_happy")
builder.add_edge("quote_happy", "advice_happy")
builder.add_edge("advice_happy", "aggregator")

# Sad 분기: 병렬 실행 후 Aggregator로
builder.add_edge("music_sad", "quote_sad")
builder.add_edge("quote_sad", "advice_sad")
builder.add_edge("advice_sad", "aggregator")

# Angry 분기: 병렬 실행 후 Aggregator로
builder.add_edge("music_angry", "quote_angry")
builder.add_edge("quote_angry", "advice_angry")
builder.add_edge("advice_angry", "aggregator")

# Tired 분기: 병렬 실행 후 Aggregator로
builder.add_edge("music_tired", "quote_tired")
builder.add_edge("quote_tired", "advice_tired")
builder.add_edge("advice_tired", "aggregator")

# Aggregator → END
builder.add_edge("aggregator", END)

# =============================================================================
# 그래프 컴파일
# =============================================================================
emotion_workflow = builder.compile()

# =============================================================================
# 실행
# =============================================================================
if __name__ == "__main__":
    # Mermaid 그래프 출력 (LangSmith에서도 확인 가능!)
    print("\n📊 Mermaid 그래프 (https://mermaid.live/ 에서 확인):")
    print("─" * 60)
    print(emotion_workflow.get_graph(xray=True).draw_mermaid())
    print("─" * 60)
    
    # 대화형 실행
    print("\n" + "=" * 60)
    print("🎧 감정 기반 음악/명언 추천 시스템")
    print("=" * 60)
    print("지금 기분을 자유롭게 말해주세요!")
    print("예: '오늘 너무 피곤해', '시험 잘 봐서 기분 좋아', '회사에서 짜증나는 일이 있었어'")
    print("=" * 60)
    
    user_input = input("\n💭 지금 기분이 어때요?: ")
    
    print("\n⏳ 감정을 분석하고 추천을 준비하는 중...\n")
    
    result = emotion_workflow.invoke({"user_input": user_input})
    
    print(result["final_output"])