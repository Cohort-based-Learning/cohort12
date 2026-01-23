# =========================
# 기본 설정
# =========================
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model  # init_chat_model 임포트
from langchain_community.tools.tavily_search import TavilySearchResults  # 검색 도구 임포트
from pydantic import BaseModel, Field

load_dotenv()

# =========================
# 모델 설정
# =========================
llm = init_chat_model(
    "gemini-2.0-flash",
    model_provider="google_genai"
)

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# =========================
# Parallelization
# =========================

# Graph state
class State(TypedDict):
    topic: str
    summary: str
    Characteristic: str
    digression: str
    combined_output: str


# -------------------------
# Nodes
# -------------------------

def call_llm_1(state: State):
    """First LLM call to generate summary"""
    msg = llm.invoke(f"{state['topic']}에 대한 개요를 알려줘")
    return {"summary": msg.content}


def call_llm_2(state: State):
    """Second LLM call to generate Characteristic"""
    msg = llm.invoke(f"{state['topic']}에 대한 특징을 알려줘")
    return {"Characteristic": msg.content}


def call_llm_3(state: State):
    """Third LLM call to generate digression"""
    msg = llm.invoke(f"{state['topic']}에 대한 여담에 대해 알려줘")
    return {"digression": msg.content}


def aggregator(state: State):
    """Combine the summary, Characteristic and digression into a single output"""
    combined = f"{state['topic']}에 대한 개요, 특징, 여담을 만들어줘!\n\n"
    combined += f"STORY:\n{state['summary']}\n\n"
    combined += f"JOKE:\n{state['Characteristic']}\n\n"
    combined += f"POEM:\n{state['digression']}"
    return {"combined_output": combined}


# -------------------------
# Build workflow
# -------------------------

parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# Add edges (병렬 실행 구조)
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")

parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")

parallel_builder.add_edge("aggregator", END)

parallel_workflow = parallel_builder.compile()

# -------------------------
# Show workflow graph
# -------------------------
print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :")
print(parallel_workflow.get_graph(xray=True).draw_mermaid())

# -------------------------
# Invoke
# -------------------------
state = parallel_workflow.invoke({"topic": "강아지"})
print(state["combined_output"])
