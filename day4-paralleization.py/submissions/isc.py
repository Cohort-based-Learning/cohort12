#기본 설정
import os
from dotenv import load_dotenv
load_dotenv()

#-------------------------------------
#모델 설정
#-------------------------------------
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# State 정의: 각 전문가의 의견을 담을 공간
class MovieState(TypedDict):
    topic: str
    director_vision: str  # 감독: 비주얼 및 컨셉
    writer_script: str    # 작가: 줄거리 및 캐릭터
    critic_review: str    # 비평가: 예상 평점 및 분석
    final_pitch: str      # 최종 기획서

# --- Nodes (전문가들) ---

def director_node(state: MovieState):
    """감독의 시각: 영화의 장르와 비주얼 스타일 결정"""
    prompt = f"'{state['topic']}'을 주제로 한 영화의 장르와 시각적 스타일을 설정해줘."
    msg = llm.invoke(prompt)
    return {"director_vision": msg.content}

def writer_node(state: MovieState):
    """작가의 시각: 핵심 줄거리(로그라인)와 주인공 설정"""
    prompt = f"'{state['topic']}'을 주제로 한 영화의 짧은 줄거리와 매력적인 주인공을 설정해줘."
    msg = llm.invoke(prompt)
    return {"writer_script": msg.content}

def critic_node(state: MovieState):
    """비평가의 시각: 이 영화가 흥행할 이유와 잠재적 위험 요소 분석"""
    prompt = f"'{state['topic']}' 영화가 제작되었을 때 예상되는 관객 반응과 비평가 평점을 분석해줘."
    msg = llm.invoke(prompt)
    return {"critic_review": msg.content}

def producer_aggregator(state: MovieState):
    """프로듀서: 모든 의견을 종합하여 하나의 '피치덱(기획서)' 완성"""
    combined = f"🎥 [영화 기획서: {state['topic']}]\n\n"
    combined += f"1. 연출 의도 (Director):\n{state['director_vision']}\n\n"
    combined += f"2. 시놉시스 (Writer):\n{state['writer_script']}\n\n"
    combined += f"3. 시장 분석 (Critic):\n{state['critic_review']}\n\n"
    combined += "-------------------------------------------\n"
    combined += "결론: 이 영화는 반드시 투자받아야 합니다!"
    return {"final_pitch": combined}

# --- Build Graph ---

builder = StateGraph(MovieState)

builder.add_node("director", director_node)
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)
builder.add_node("producer", producer_aggregator)

# 병렬 실행 시작
builder.add_edge(START, "director")
builder.add_edge(START, "writer")
builder.add_edge(START, "critic")

# 모든 분석이 끝나면 프로듀서에게 집결
builder.add_edge("director", "producer")
builder.add_edge("writer", "producer")
builder.add_edge("critic", "producer")

builder.add_edge("producer", END)

movie_workflow = builder.compile()

# 실행
result = movie_workflow.invoke({"topic": "우주에서 길을 잃은 고양이"})
print(result["final_pitch"])