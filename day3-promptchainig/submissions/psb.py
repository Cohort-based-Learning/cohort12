#기본 설정
import os
from pymongo import MongoClient # 몽고DB 연결
from langchain_core.tools import tool # 도구 정의
from langgraph.prebuilt import ToolNode, tools_condition # 도구 노드
from langgraph.graph.message import add_messages # 메시지 추가
from langchain_core.messages import BaseMessage, SystemMessage # 메시지 타입
from dotenv import load_dotenv # 환경변수 로드
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
#Prompt chaining : “이전 LLM 호출의 결과를 다음 호출 입력으로 넘기는” 순차 처리.
#-------------------------------------
from typing import Annotated, TypedDict # 타입 정의
from langgraph.graph import StateGraph, START, END # 상태 그래프


# 상태 정의
class State(TypedDict): # 상태 딕셔너리
    messages: Annotated[list[BaseMessage], add_messages] # 메시지 리스트
    
# 도구 수정
@tool
def search_parts(keyword: str):
    """
    브릭 이름으로 검색 (예: 'Plate', 'Brick', 'Technic')
    해당 키워드가 포함된 부품 최대 5개를 찾아 정보를 반환
    """
    print(f"\n DB에서 '{keyword}' 검색 중...") # 확인용 로그
    
    # 정규표현식($regex)을 써서 이름에 키워드가 포함된 걸 찾습니다.
    results = list(collection.find(
        {
            "$or": [
            {"name": {"$regex": keyword, "$options": "i"}}, # i: 대소문자 무시
            {"keywords": {"$regex": keyword, "$options": "i"}}, # 키워드(태그) 검색
            {"partId": {"$regex": keyword, "$options": "i"}},  # 부품 번호 검색
            ]
        },
        {"_id": 0, "partId": 1, "name": 1, "category": 1} # 가져올 필드
    ).limit(5)) # 5개만 제한
    
    if results:
        return str(results)
    return "검색 결과가 없습니다."

# 모델에 검색 도구 장착!
llm_with_tools = llm.bind_tools([search_parts])

# 노드 정의: 챗봇 뚝배기
def chatbot(state: State):
    # 대화 흐름을 보고 답변할지, 도구를 쓸지 결정
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", ToolNode([search_parts]))

# Add edges to connect nodes
workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "chatbot") # 결과 보고 다시 챗봇으로 

# Compile workflow
app = workflow.compile()

if __name__ == "__main__":
    # 상황: 레고 DB는 다 영어로 되어 있습니다. (예: Plate, Brick)
    # 그런데 사용자가 굳이 '한글'로 물어봅니다.
    
    query = "이름에 '바퀴'라고 적힌 거 찾아"
    print(f"User: {query}")
    print("-" * 50)

    # ★ 시스템 메시지: 포기하지 말고 다른 단어로 찾으라고 시킴 (LangGraph의 맛)
    sys_msg = SystemMessage(content="""
    니는 레고 박사다.
    1. 입력한 키워드로 먼저 검색해라
    2. 만약 '결과 없음'이 나오면, 즉시 영어 단어나 유의어로 바꿔서 **다시 검색 도구를 호출**할것.
    3. 최소 2번은 시도해보고 그래도 없으면 그때 포기해.
    """)

    inputs = {"messages": [sys_msg, ("user", query)]}
    
    for event in app.stream(inputs):
        for key, value in event.items():
            last_msg = value["messages"][-1]
            
            # 도구 호출 로그
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                 print(f"\n [AI 시도] '{last_msg.tool_calls[0]['args']['keyword']}' 검색 시도...")
            
            # 최종 답변
            elif last_msg.content:
                print(f"\n [최종 답변] {last_msg.content}")

#메르메이드 그래프 시각화
print("\n" + "="*50)
print("▼ 아래 코드를 복사해서 https://mermaid.live 에 붙여넣으세요 ▼")
try:
    print(app.get_graph().draw_mermaid()) 
except Exception:
    # 랭그래프 버전에 따라 draw_mermaid()가 안 될 수도 있어서 예외처리 함
    print("graph.draw_mermaid() 기능을 사용할 수 없는 환경입니다.")
print("="*50)

# # Invoke
# state = chain.invoke({"topic": "강아지"})
# print("Initial joke:")    
# print(state["joke"])
# print("\n--- --- ---\n")
# if "improved_joke" in state:
#     print("Improved joke:")
#     print(state["improved_joke"])
#     print("\n--- --- ---\n")

#     print("Final joke:")
#     print(state["final_joke"])
# else:
#     print("Final joke:")
#     print(state["joke"])