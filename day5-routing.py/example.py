#기본 설정
import os
from dotenv import load_dotenv
load_dotenv()

#-------------------------------------
#모델 설정
#-------------------------------------
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

#-------------------------------------
#Routing (입력값 처리 후 컨텍스트별 작업으로 전달)
#-------------------------------------
from typing_extensions import Literal 
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# LLM이 반드시 step에 "poem"/"story"/"joke" 중 하나를 내게 강제.
class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        None, description="유저의 입력에 대해 poem/stroy/joke 중 하나를 선택하여라"
    )


# 구조화 출력 강제된 라우터
router = llm.with_structured_output(Route)


# State
class State(TypedDict):
    input: str #사용자 입력
    decision: str #분기 결정
    output: str #출력


# Nodes
def llm_call_1(state: State):
    """Write a story"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_2(state: State):
    """Write a joke"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_3(state: State):
    """Write a poem"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


# 라우터 노드
def llm_call_router(state: State):
    """Route the input to the appropriate node"""

    # Run the augmented LLM with structured output to serve as routing logic
    decision = router.invoke(
        [
            SystemMessage(
                content="유저의 입력에 대해 poem/stroy/joke 중 하나를 선택하여라"
            ),
            HumanMessage(content=state["input"]),
        ]
    )
    print(f"Decision: {decision.step}")
    return {"decision": decision.step}


# Conditional edge function to route to the appropriate node
def route_decision(state: State):
    # Return the node name you want to visit next
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"


# Build workflow
router_builder = StateGraph(State)

# Add nodes
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)

# Add edges to connect nodes
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

# Compile workflow
router_workflow = router_builder.compile()

# Show the workflow
print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :") #사이트 들어가서 코드 붙여넣기
print(router_workflow.get_graph(xray=True).draw_mermaid())

# Invoke
answer = router_workflow.invoke({"input": "고양이에 대해 짧은 농담을 적어줘."})
print(f"\nAnswer: {answer}")