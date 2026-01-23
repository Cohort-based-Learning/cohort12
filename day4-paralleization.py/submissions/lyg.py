#기본 설정
import os
from dotenv import load_dotenv
load_dotenv()

#-------------------------------------
#모델 설정
#-------------------------------------
# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    #model="claude-3-5-sonnet-20241022"
    # model= "openai:gpt-4.1",
    model="gpt-5-nano",
)


from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

#-------------------------------------
#Parallelization(병렬화: 서로 독립인 작업을 동시에 돌림)
#-------------------------------------

# Graph state
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str


# Nodes
def call_llm_1(state: State):
    """First LLM call to generate initial joke"""

    msg = llm.invoke(f"{state['topic']}에 대한 농담을 만들어줘")
    return {"joke": msg.content}


def call_llm_2(state: State):
    """Second LLM call to generate story"""

    msg = llm.invoke(f"{state['topic']}에 대한 이야기를 만들어줘")
    return {"story": msg.content}


def call_llm_3(state: State):
    """Third LLM call to generate poem"""

    msg = llm.invoke(f"{state['topic']}에 대한 시를 만들어줘")
    return {"poem": msg.content}


def aggregator(state: State): #합치기
    """Combine the joke, story and poem into a single output"""

    combined = f"{state['topic']}에 대한 이야기, 농담, 시를 만들어줘!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}


# Build workflow
parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# Add edges to connect nodes 독립적 실행이 포인트
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# Show workflow
print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :") #사이트 들어가서 코드 붙여넣기
print(parallel_workflow.get_graph(xray=True).draw_mermaid())

# Invoke
state = parallel_workflow.invoke({"topic": "고양이"})
print(state["combined_output"])