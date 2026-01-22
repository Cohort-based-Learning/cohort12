#기본 설정
# pip install langchain_core langchain-anthropic langgraph

import os
from dotenv import load_dotenv
load_dotenv()

#-------------------------------------
#모델 설정
#-------------------------------------
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")

#-------------------------------------
#llm + 기능(증강)
#-------------------------------------

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages

# 1. 구조화된 출력 스키마
class CalculationResult(BaseModel):
    steps: List[int] = Field(description="계산 과정의 숫자 리스트")
    count: int = Field(description="총 계산 횟수")
    is_success: bool = Field(description="1에 도달했는지 여부")

# 2. 도구 정의
@tool
def process_number(n: int) -> int:
    """숫자가 홀수이면 1을 빼고 2로 나누고, 짝수이면 2로 나눕니다."""
    if n % 2 == 0:
        return n // 2
    else:
        return (n - 1) // 2

# 기능 증강
llm_with_tools = llm.bind_tools([process_number])
structured_llm = llm.with_structured_output(CalculationResult)

# 3. LangGraph 상태 및 노드 정의
class State(TypedDict):
    # Annotated와 add_messages를 사용해야 기존 메시지 뒤에 새 메시지가 붙습니다.
    messages: Annotated[list[BaseMessage], add_messages] 
    final_data: CalculationResult 

def agent(state: State):
    """AI가 계산을 계속할지 결정합니다. (전체 메시지 기록을 참조)"""
    # 현재까지의 모든 대화(질문+도구결과)를 모델에 전달
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def execute_tool(state: State):
    """실제로 도구를 실행하고 결과를 기록에 추가합니다."""
    last_message = state["messages"][-1]
    tool_results = []
    for tool_call in last_message.tool_calls:
        result = process_number.invoke(tool_call["args"])
        # ToolMessage는 반드시 이전 AI메시지의 tool_call_id를 참조해야 합니다.
        tool_results.append(ToolMessage(tool_call_id=tool_call["id"], content=str(result)))
    return {"messages": tool_results}

def summarizer(state: State):
    """모든 루프가 끝나면 structured_llm을 통해 결과를 객체화합니다."""
    prompt = "지금까지의 모든 계산 과정을 분석해서 CalculationResult 형식으로 응답해줘."
    # 누적된 메시지 전체를 바탕으로 최종 요약 생성
    result = structured_llm.invoke(state["messages"] + [HumanMessage(content=prompt)])
    return {"final_data": result}

# 4. 그래프 구성 (에러 해결 핵심: 노드 이름 매칭)
workflow = StateGraph(State)

workflow.add_node("agent", agent)
workflow.add_node("tools", execute_tool)  # 이름을 'tools'로 설정하여 에러 방지
workflow.add_node("summarizer", summarizer)

workflow.set_entry_point("agent")

# 조건부 로직 (Router)
def should_continue(state: State):
    last_message = state["messages"][-1]
    # 도구 호출 요청이 있으면 'tools' 노드로
    if last_message.tool_calls:
        return "tools"
    # 도구 호출이 없으면(즉, 결과가 1이 되어 AI가 계산을 멈추면) 'summarizer'로
    return "summarizer"

workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent") # 계산 후 다시 확인하러 이동
workflow.add_edge("summarizer", END)

app = workflow.compile()

# 5. 실행
input_msg = {"messages": [HumanMessage(content="시작 숫자는 21야. 1이 될 때까지 계산해주고 요약해줘.")]}

print("--- 루프 시작 ---")
final_state = app.invoke(input_msg, {"recursion_limit": 50})

# 6. 최종 구조화된 데이터 확인
output = final_state["final_data"]
print("\n--- 최종 구조화된 결과 ---")
print(f"과정: {output.steps}")
print(f"횟수: {output.count}")
print(f"성공: {output.is_success}")