#기본 설정
import os
from langgraph.graph import END
from langgraph.graph import StateGraph, START
from pymongo import MongoClient # 몽고DB 연결
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()

#-------------------------------------
#모델 설정 / MongoDB 연결
#-------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI # Google Gemini API 사용
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
MongoDB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MongoDB_URI)
DB = client["brickers"]
collection = DB["ldraw_parts"]

#-------------------------------------
#Parallelization(병렬화: 서로 독립인 작업을 동시에 돌림)
#-------------------------------------

# Graph state
class State(TypedDict):
    parts: str
    data: str
    design: str
    ad: str
    combined_output: str


# Nodes
def parts(state: State):
    """First LLM call to generate initial parts"""

    msg = llm.invoke(f"{state['parts']}의 다음 정보 3가지 항목을 알려줘 - 부품 번호, 이름, 카테고리") #상태값 업데이트
    return {"data": msg.content}


def design(state: State):
    """Second LLM call to generate design ideas"""

    msg = llm.invoke(f"{state['parts']}를 활용한 다양한 레고 디자인 아이디어를 제안해봐")
    return {"design": msg.content}


def ad(state: State):
    """Third LLM call to generate ad"""

    msg = llm.invoke(f"{state['parts']}를 홍보하는 창의적인 광고 문구를 작성해줘")
    return {"ad": msg.content}


def aggregator(state: State): #합치기
    """Combine the data, design and ad into a single output"""

    combined = f"{state['parts']}에 대한 정보, 디자인 아이디어, 광고 문구를 만들어줘!\n\n"
    combined += f"DATA:\n{state['data']}\n\n"
    combined += f"DESIGN:\n{state['design']}\n\n"
    combined += f"AD:\n{state['ad']}\n"
    return {"combined_output": combined}


# Build workflow
parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("parts", parts)
parallel_builder.add_node("design", design)
parallel_builder.add_node("ad", ad)
parallel_builder.add_node("aggregator", aggregator)

# Add edges to connect nodes 독립적 실행이 포인트
parallel_builder.add_edge(START, "parts")
parallel_builder.add_edge(START, "design")
parallel_builder.add_edge(START, "ad")
parallel_builder.add_edge("parts", "aggregator")
parallel_builder.add_edge("design", "aggregator")
parallel_builder.add_edge("ad", "aggregator")
parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# Show workflow
print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :") #사이트 들어가서 코드 붙여넣기
print(parallel_workflow.get_graph(xray=True).draw_mermaid())

# Invoke
state = parallel_workflow.invoke({"parts": "플레이트"})
print(state["combined_output"])