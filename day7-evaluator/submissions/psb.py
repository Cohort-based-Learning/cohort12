#기본 설정
import os
from dotenv import load_dotenv
load_dotenv()

#-------------------------------------
#모델 설정
#-------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI 
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal #주어진 보기 안에서 선택하도록 강제

# DB 설정
from pymongo import MongoClient
MongoDB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MongoDB_URI)
DB = client["brickers"]
collection = DB["ldraw_parts"]

#-------------------------------------
#Evaluator-optimizer(평가-개선 루프) *예제와 다르게 개선 루프 3회 제한
#-------------------------------------
# Graph state

class State(TypedDict):
    target_part: str      # 사용자가 요청한 부품명 (예: "Brick 2x4")
    db_data: str          # DB에서 찾아낸 부품 정보 (Context)
    description: str      # AI가 작성한 설명글 (Draft)
    feedback: str         # 평가자의 피드백
    grade: str            # 평가 결과 (pass / rewrite)
    attempts: int         # 시도 횟수

# 평가 결과를 구조화할 스키마(Feedback)
class Feedback(BaseModel):
    # evaluator의 출력 형식을 강제로 고정
    grade: Literal["pass", "rewrite"] = Field(
        description="설명글이 완벽하면 'pass', 수정이 필요하면 'rewrite'를 선택하세요."
    )
    feedback: str = Field(
        description="수정이 필요하다면 구체적으로 어떤 정보(ID, 색상 등)가 누락되었는지, 혹은 톤앤매너를 어떻게 고쳐야 할지 조언하세요."
    )


# evaluator(평가 LLM) 만들기
evaluator = llm.with_structured_output(Feedback)


# 노드 1: llm_call_generator (생성기)
def llm_call_generator(state: State):
    """DB 정보를 바탕으로 설명글을 작성하거나, 피드백을 반영해 수정합니다."""

    # 시도 횟수 카운트 (없으면 0으로 시작)
    current_attempts = state.get("attempts") or 0
    new_attempts = current_attempts + 1

    # 첫 시도일 때만 DB 검색 수행 (Context 확보)
    db_context = state.get("db_data")
    if not db_context:
        print(f"🔍 [SEARCH] '{state['target_part']}' DB 검색 중...")
        search_res = list(collection.find(
            {
                "$or": [
                    {"name": {"$regex": state['target_part'], "$options": "i"}}, # 파일 이름
                    {"keywords": {"$regex": state['target_part'], "$options": "i"}}, # 키워드
                    {"partId": {"$regex": state['target_part'], "$options": "i"}} # 부품 ID
                ]
            },
        {"_id": 0, "name": 1, "partId": 1, "keywords": 1, "category": 1}
        ).limit(1))
        
        if search_res:
            db_context = str(search_res[0])
        else:
            db_context = "DB에 해당 부품 정보가 없습니다."

    # 프롬프트 구성
    if state.get("feedback"):
        # 재시도: 피드백 반영
        prompt = f"""
        당신은 레고 마케팅 전문가입니다. 
        아래 [기존 초안]을 [피드백]에 맞춰서 훨씬 더 매력적이고 정확하게 다시 작성하세요.
        
        [부품 정보]: {db_context}
        [기존 초안]: {state['description']}
        [피드백]: {state['feedback']}
        """
    else:
        # 첫 시도: 신규 작성
        prompt = f"""
        당신은 레고 마케팅 전문가입니다. 
        제공된 [부품 정보]를 바탕으로 쇼핑몰에 올릴 상세하고 매력적인 상품 소개글을 작성하세요.
        반드시 부품의 ID와 이름을 정확하게 명시해야 합니다.
        
        [부품 정보]: {db_context}
        """

    msg = llm.invoke(prompt)
    
    print(f"\n📝 [DRAFT {new_attempts}] 생성됨:\n{msg.content[:100]}...") # 앞부분만 살짝 출력

    return {
        "description": msg.content, 
        "attempts": new_attempts, 
        "db_data": db_context # DB 정보 저장해두기
    }


# 노드 2: llm_call_evaluator (평가자)
def llm_call_evaluator(state: State):
    """작성된 설명글을 깐깐하게 평가합니다."""

    prompt = f"""
    당신은 세상에서 가장 성격이 꼬인 악덕 편집장입니다.
    아래 [설명글]이 다음 '기준'을 모두 만족하지 못하면 가차 없이 'rewrite'를 주고 독설을 퍼부으세요.
    
    [평가 기준 - 하나라도 틀리면 탈락]
    1. [이모지 폭탄]: 본문에 이모지가 **정확히 7개** 포함되어야 함. (무조건 7개)
    2. [오글거림]: 문장의 시작은 무조건 "주목하라, 레고 덕후들이여!" 로 시작해야 함.
    3. [특정 단어]: 본문에 "지갑 털릴 준비 되셨나요?" 라는 문구가 반드시 포함되어야 함.
    4. [형식]: 마지막 줄은 반드시 해시태그 3개(#레고 #브릭 #필수템)로 끝나야 함.
    
    [DB 정보]: {state['db_data']}
    [설명글]: {state['description']}
    """
    
    result = evaluator.invoke(prompt)
    
    print(f"[EVAL {state['attempts']}] 편집장 판정: {result.grade.upper()}")
    if result.grade == "rewrite":
        print(f"   🔥 독설 피드백: {result.feedback}")

    return {"grade": result.grade, "feedback": result.feedback}

# 라우팅 함수
def route_decision(state: State):
    if state["grade"] == "pass":
        print("[SUCCESS] 통과! 완료합니다.")
        return "Accepted"
    
    if state["attempts"] >= 3:
        print("[STOP] 3회 시도 초과. 강제 종료합니다.")
        return "Accepted"
    
    print("[LOOP] 다시 작성하러 갑니다...")
    return "Retry"


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
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "Accepted": END,
        "Retry": "llm_call_generator",
    },
)

# Compile the workflow
optimizer_workflow = optimizer_builder.compile()

# Show the workflow
print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :") #사이트 들어가서 코드 붙여넣기
print(optimizer_workflow.get_graph(xray=True).draw_mermaid())

# Invoke
print("-" * 50)
# 예: 'Brick' 검색해서 마케팅 문구 써줘
result = optimizer_workflow.invoke({"target_part": "Brick"}) 

print("\n" + "="*50)
print("[최종 결과물]")
print(result["description"])
print("="*50)