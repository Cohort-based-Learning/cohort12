#기본 설정
# pip install langchain_core langchain-google-genai langgraph

from dotenv import load_dotenv
load_dotenv()

#-------------------------------------
#모델 설정
#-------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 기본 llm invoke
print(f"기본 llm invoke \n {llm.invoke('이탈리아의 대표적인 음식 3가지만 알려줘')}")
print()
print(f"기본 llm invoke \n {llm.invoke('프랑스 요리에서 버터가 중요한 이유는?')}")

#-------------------------------------
#llm + 기능(증강) - 식문화 탐구편
#-------------------------------------

# 1. 구조화된 출력을 위한 증강: 음식 정보 추출
from pydantic import BaseModel, Field
from typing import List

class FoodInfo(BaseModel):
    dish_name: str = Field(..., description="음식의 이름")
    country: str = Field(..., description="음식의 기원 국가")
    ingredients: List[str] = Field(..., description="주요 재료 목록")
    flavor_profile: str = Field(..., description="맛의 특징이나 식감")

# LLM에 구조화된 출력을 위한 증강
structured_llm = llm.with_structured_output(FoodInfo)

# Invoke
food_query = "똠얌꿍은 태국의 대표적인 수프로, 새우, 레몬그라스, 라임, 고추 등을 넣어 끓이며 맵고 시고 단 맛이 조화를 이룹니다."
output = structured_llm.invoke(food_query)
print(f"\n1. 구조화된 출력을 위한 증강 (음식 정보) \n {output}")


# 2. 도구를 위한 증강: 식사 예절 조회
# Define a tool
def get_dining_etiquette(country: str) -> str:
    """특정 국가의 식사 예절을 조회합니다."""
    if country == "Japan":
        return "젓가락으로 음식을 주고받지 마세요. 밥그릇은 들고 먹는 것이 좋습니다."
    elif country == "France":
        return "식사 중에는 두 손을 식탁 위에 보이게 두세요. 팔꿈치는 올리지 않습니다."
    else:
        return "해당 국가의 특별한 식사 예절 정보를 찾을 수 없습니다."

# LLM에 도구를 위한 증강
llm_with_tools = llm.bind_tools([get_dining_etiquette])

# Invoke
etiquette_query = "일본에서 밥 먹을 때 조심해야 할 점이 있어?"
msg = llm_with_tools.invoke(etiquette_query)
print(f"\n2. 도구를 위한 증강 (식사 예절) \n {msg}")

# 실제로 LLM이 생성한 도구호출 목록
print(f"\n도구호출 목록: {msg.tool_calls}")

# 본인만의 구조와 도구를 만들어서 적용해보세요