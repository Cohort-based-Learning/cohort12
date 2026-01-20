#기본 설정
# pip install langchain_core langchain-anthropic langgraph

from pathlib import Path
from dotenv import load_dotenv
import os

# .env 파일 경로 지정 (현재 파일 기준: .../study26/day2.../submissions/bjy.py)
# 목표: .../study26/path/to/your/app/.env
current_dir = Path(__file__).resolve().parent
# parents[1] -> study26 폴더 (2가 아니라 1이어야 함!)
env_path = current_dir.parents[1] / 'path' / 'to' / 'your' / 'app' / '.env'

print(f"[Debug] .env 경로 확인: {env_path}")
load_dotenv(dotenv_path=env_path)

#-------------------------------------
#모델 설정
#-------------------------------------
# 모델 설정 (Gemini로 변경)
# -------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI

# .env에 있는 GOOGLE_API_KEY를 자동으로 사용합니다.
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# 기본 llm invoke
print(f"기본 llm invoke \n {llm.invoke('칼슘 CT 점수는 고콜레스테롤과 어떤 관련이 있나요?')}")
print()
print(f"기본 llm invoke \n {llm.invoke('2에 3을 곱하면?')}")

#-------------------------------------
#llm + 기능(증강)
#-------------------------------------

# 1. 구조화된 출력을 위한 증강
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.") #웹검색에 최적화된 쿼리 텍스트
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    ) #이 쿼리가 사용자의 요청과 관련이 있는 이유

#LLM에 구조화된 출력을 위한 증강
structured_llm = llm.with_structured_output(SearchQuery) 

#Invoke
output = structured_llm.invoke("칼슘 CT 점수는 고콜레스테롤과 어떤 관련이 있나요?")
print(f"\n1. 구조화된 출력을 위한 증강 \n {output}")

# 2. 도구를 위한 증강
# Define a tool
def multiply(a: int, b: int) -> int:
    return a * b

#LLM에 도구를 위한 증강
llm_with_tools = llm.bind_tools([multiply])

#Invoke
msg = llm_with_tools.invoke("2에 3을 곱하면?")
print(f"\n2. 도구를 위한 증강 \n {msg}")

# 실제로 LLM이 생성한 도구호출 목록
print(f"\n도구호출 목록: {msg.tool_calls}")

# 본인만의 구조와 도구를 만들어서 적용해보세요:)

# -------------------------------------
# 3. 나만의 도구: 비밀 암호 생성기 
# -------------------------------------

def make_secret_code(text: str) -> str:
    """
    입력된 텍스트를 비밀 암호로 변환
    (글자를 뒤집고 공백을 언더바로 변경)
    """
    return f"SECRET_CODE[{text[::-1].replace(' ', '_')}]"

import random

# -------------------------------------
# 4. 또 다른 도구: 결정 장애 해결사 (랜덤 뽑기) 
# -------------------------------------
def pick_one(candidates: list[str]) -> str:
    """
    주어진 목록 중에서 하나를 랜덤으로 선택합니다.
    예: ["짜장면", "짬뽕"] -> "짬뽕"
    """
    return random.choice(candidates)

# LLM에 내 도구들 등록하기 (비밀암호 + 랜덤뽑기)
llm_with_tools = llm.bind_tools([make_secret_code, pick_one])

# -------------------------------------
# 실행 시나리오 (Sequential Flow)
# -------------------------------------
print(f"\n--- 비밀 요원 점심 메뉴 추천 서비스 ---")

# 1. 이름 입력 받기
name = input("먼저, 당신의 이름을 알려주시오: ")

# 2. 이름을 암호화 (LLM에게 시키기)
# "내 이름을 암호로 바꿔줘"라고 요청하면 도구를 사용함
response1 = llm_with_tools.invoke(f"'{name}'을 비밀 암호로 만들어줘.")

secret_name = "UNKNOWN"
# 도구가 호출되었는지 확인하고 실행
if response1.tool_calls:
    for tool_call in response1.tool_calls:
        if tool_call['name'] == "make_secret_code":
            secret_name = make_secret_code(**tool_call['args'])

print(f"\n시스템: 당신의 비밀 요원 코드는 [{secret_name}] 입니다.")
print("시스템: 보안 확인 완료. 이제 점심 메뉴를 골라드리겠습니다.")

# 3. 점심 메뉴 추천 (자동으로 이어가기)
menu_candidates = "짜장면, 짬뽕, 볶음밥, 돈까스, 김치찌개"
print(f"\n(후보: {menu_candidates})")
response2 = llm_with_tools.invoke(f"이 중에서 점심 메뉴 하나만 무작위로 골라줘: {menu_candidates}")

# 도구 실행 결과 확인
if response2.tool_calls:
    for tool_call in response2.tool_calls:
        if tool_call['name'] == "pick_one":
            picked_menu = pick_one(**tool_call['args'])
            print(f"\n오늘의 추천 메뉴: [{picked_menu}]")
else:
    print(f"\n추천 결과: {response2.content}")