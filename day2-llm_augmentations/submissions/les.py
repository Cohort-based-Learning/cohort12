# pip install -U langchain-openai openai python-dotenv pydantic

from dotenv import load_dotenv
load_dotenv()  # .env에서 OPENAI_API_KEY 로드

#-------------------------------------
# 모델 설정 (OpenAI)
#-------------------------------------
from langchain_openai import ChatOpenAI

# 모델 추천:
# - gpt-4o: 성능 위주
# - gpt-4o-mini: 비용/속도 위주
# (OpenAI 문서에서도 Chat Completions 쪽 대표 모델로 gpt-4o / gpt-4o-mini를 안내함) :contentReference[oaicite:1]{index=1}
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 기본 llm invoke
print("기본 llm invoke\n", llm.invoke("칼슘 CT 점수는 고콜레스테롤과 어떤 관련이 있나요?").content)
print()
print("기본 llm invoke\n", llm.invoke("2에 3을 곱하면?").content)
print("기본 llm invoke\n", llm.invoke("3에 5을 곱하면?").content)
print("기본 llm invoke\n", llm.invoke("4에 6을 곱하면?").content)
print("기본 llm invoke\n", llm.invoke("7에 3을 곱하면?").content)

#-------------------------------------
# llm + 기능(증강)
#-------------------------------------

# 1) 구조화된 출력을 위한 증강
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    # OpenAI structured output/툴콜링은 "필수 필드"가 더 안정적이라 기본값 None은 빼는 걸 추천
    search_query: str = Field(description="Query optimized for web search.")
    justification: str = Field(description="Why this query is relevant to the user's request.")

structured_llm = llm.with_structured_output(SearchQuery)

output = structured_llm.invoke("칼슘 CT 점수는 고콜레스테롤과 어떤 관련이 있나요?")
print("\n1. 구조화된 출력을 위한 증강\n", output)

# 2) 도구를 위한 증강
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b    

@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """Divide a by b."""
    return a / b

llm_with_tools = llm.bind_tools([multiply, add, subtract])

msg = llm_with_tools.invoke("2에 3을 곱하면?")
msg = llm_with_tools.invoke("3에 5을 더하면?")
msg = llm_with_tools.invoke("4에 6을 빼면?")
msg = llm_with_tools.invoke("7에 3을 나누면?")
print("\n2. 도구를 위한 증강\n", msg.content)
print("\n도구호출 목록:", msg.tool_calls)