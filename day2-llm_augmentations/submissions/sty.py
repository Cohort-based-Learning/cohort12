#기본 설정
# pip install langchain_core langchain-google-genai langgraph python-dotenv

from dotenv import load_dotenv
load_dotenv()

#-------------------------------------
#모델 설정
#-------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("="*50)
print("🤖 LLM 증강 예제 - STY")
print("="*50)

# 기본 llm invoke
print(f"\n📝 기본 llm invoke:")
print(llm.invoke('오늘 기분이 좋아!').content)

#-------------------------------------
#llm + 기능(증강)
#-------------------------------------

# 1. 구조화된 출력을 위한 증강 - 음식 추천
from pydantic import BaseModel, Field
from typing import List

class FoodRecommendation(BaseModel):
    """음식 추천 결과"""
    food_name: str = Field(description="추천 음식 이름")
    cuisine_type: str = Field(description="음식 종류 (한식/중식/일식/양식 등)")
    calories: int = Field(description="대략적인 칼로리 (kcal)")
    reason: str = Field(description="추천 이유")
    ingredients: List[str] = Field(description="주요 재료 3가지")

structured_llm = llm.with_structured_output(FoodRecommendation)

print("\n" + "="*50)
print("🍽️ 1. 구조화된 출력 - 음식 추천")
print("="*50)
output = structured_llm.invoke("점심으로 뭐 먹을지 추천해줘. 매운 거 좋아해!")
print(f"음식: {output.food_name}")
print(f"종류: {output.cuisine_type}")
print(f"칼로리: {output.calories}kcal")
print(f"이유: {output.reason}")
print(f"재료: {', '.join(output.ingredients)}")


# 2. 여러 도구를 위한 증강
import random
from datetime import datetime

def add(a: int, b: int) -> int:
    """두 숫자를 더합니다"""
    return a + b

def subtract(a: int, b: int) -> int:
    """두 숫자를 뺍니다"""
    return a - b

def multiply(a: int, b: int) -> int:
    """두 숫자를 곱합니다"""
    return a * b

def divide(a: int, b: int) -> float | str:
    """두 숫자를 나눕니다"""
    if b == 0:
        return "0으로 나눌 수 없습니다"
    return a / b

def get_current_time() -> str:
    """현재 시간을 반환합니다"""
    return datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분 %S초")

def roll_dice(sides: int = 6) -> int:
    """주사위를 굴립니다. sides는 주사위 면의 수입니다."""
    return random.randint(1, sides)

def get_weather(city: str) -> str:
    """도시의 날씨를 반환합니다 (가상 데이터)"""
    weathers = ["맑음 ☀️", "흐림 ☁️", "비 🌧️", "눈 ❄️", "안개 🌫️"]
    temps = random.randint(-5, 35)
    weather = random.choice(weathers)
    return f"{city}의 날씨: {weather}, 온도: {temps}°C"

def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    """BMI를 계산합니다"""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        status = "저체중"
    elif bmi < 25:
        status = "정상"
    elif bmi < 30:
        status = "과체중"
    else:
        status = "비만"
    return f"BMI: {bmi:.1f} ({status})"

# 모든 도구 바인딩
tools = [add, subtract, multiply, divide, get_current_time, roll_dice, get_weather, calculate_bmi]
llm_with_tools = llm.bind_tools(tools)

print("\n" + "="*50)
print("🔧 2. 도구 증강 테스트")
print("="*50)

# 테스트 질문들
questions = [
    "지금 몇 시야?",
    "주사위 굴려줘!",
    "서울 날씨 어때?",
    "100 나누기 7은?",
    "키 175cm에 몸무게 70kg이면 BMI가 어떻게 돼?"
]

for q in questions:
    print(f"\n💬 질문: {q}")
    msg = llm_with_tools.invoke(q)
    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            print(f"  🔧 호출 도구: {tool_call['name']}")
            print(f"  📥 인자: {tool_call['args']}")

            # 실제 도구 실행
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            # 도구 찾아서 실행
            for tool in tools:
                if tool.__name__ == tool_name:
                    result = tool(**tool_args)
                    print(f"  📤 결과: {result}")
                    break
    else:
        print(f"  💭 응답: {msg.content[:100]}...")

print("\n" + "="*50)
print("✅ 완료!")
print("="*50)
