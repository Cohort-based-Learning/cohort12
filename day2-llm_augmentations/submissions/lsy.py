#기본 설정
# pip install langchain_core langchain-anthropic langgraph

from dotenv import load_dotenv
load_dotenv()

#-------------------------------------
#모델 설정
#-------------------------------------
#from langchain_anthropic import ChatAnthropic
#llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-5-nano",
)

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