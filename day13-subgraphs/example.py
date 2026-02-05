"""
- 멀티에이전트 시스템 구축 시
- 동일한 로직을 여러 그래프에서 재사용하고 싶을 때
- 팀 간에 그래프 특정 부분을 분리해 개발하고 싶을 때 (단, 입출력 스키마 준수는 필요)

*subgraph와 parent graph간의 통신 방식 정의 필요
1. invoke a graph from a node
2. add a graph as a node
"""

#------------------------------------------
#Invoke a graph from a node
"""
Parent graph
  └ parent_1
      └ child graph
          └ child_1

*서브그래프를 노드 함수 안에서 직접 호출
*부모/자식 state는 분리
*그래서 입출력 변환을 호출자가 직접 해야 함
"""
#------------------------------------------

print(f"\n### invoke a graph from a node ###\n")
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

# Define subgraph
class SubgraphState(TypedDict):
    # note: 부모 그래프와 공유되지 않는 state
    bar: str
    baz: str

def subgraph_node_1(state: SubgraphState):
    return {"baz": "baz"}

def subgraph_node_2(state: SubgraphState):
    return {"bar": state["bar"] + state["baz"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

# Define parent graph
class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

def node_2(state: ParentState):
    # 상태를 서브그래프 state로 변환
    response = subgraph.invoke({"bar": state["foo"]})
    # 응답을 부모 state로 변환
    return {"foo": response["bar"]}


builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :") #사이트 들어가서 코드 붙여넣기
print(graph.get_graph(xray=True).draw_mermaid())

for chunk in graph.stream({"foo": "foo"}, subgraphs=True):
    print(chunk)


#------------------------------------------
#Add a graph as a node
"""
point: shared messages key

Parent graph (state: foo)
  ├ node_1
  └ node_2 (subgraph)
       ├ subgraph_node_1
       └ subgraph_node_2

*컴파일된 graph 자체를 부모 그래프의 노드로 추가
*부모 state 중 공유 키는 그대로, 서브그래프의 private 키는 서브그래프 내부에서만 사용
*결과로 공유 키 업데이트만 부모로 전달
"""
#------------------------------------------

print(f"\n### add a graph as a node ###\n")

from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

# Define subgraph
class SubgraphState(TypedDict):
    foo: str  # shared with parent graph state
    baz: str  # private to SubgraphState

def subgraph_node_1(state: SubgraphState):
    return {"baz": "baz"}

def subgraph_node_2(state: SubgraphState):
    # 이 노드는 서브그래프 내에서만 사용 가능한 상태 키('bar')를 사용하고 있으며
    # 공유 상태 키('foo')에 대한 업데이트를 전송하고 있습니다.
    return {"foo": state["foo"] + state["baz"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

# Define parent graph
class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :") #사이트 들어가서 코드 붙여넣기
print(graph.get_graph(xray=True).draw_mermaid())

for chunk in graph.stream({"foo": "foo"}, subgraphs=True):
    print(chunk)


# ------------------------------------------
# View subgraph state: only in interrupt
# ------------------------------------------

print(f"\n### view subgraph state: only in interrupt ###\n")

from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

class State(TypedDict):
    foo: str

# Subgraph

def subgraph_node_1(state: State):
    value = interrupt("Provide value:")
    return {"foo": state["foo"] + value}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")

subgraph = subgraph_builder.compile()

# Parent graph

builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}

graph.invoke({"foo": ""}, config)
parent_state = graph.get_state(config)
print(f"[parent_state]: {parent_state}\n")

# interrupt 동안에만 subgraphs=True로 서브그래프 내부 snapshot을 펼쳐서 볼 수 있음
subgraph_state = graph.get_state(config, subgraphs=True).tasks[0].state
print(f"[subgraph_state]: {subgraph_state}\n")

# resume the subgraph
graph.invoke(Command(resume="bar"), config)

# 실행이 끝나면 또 못 봄
subgraph_state = graph.get_state(config, subgraphs=True).tasks[0].state
print(f"\n[interrupt 후]: {subgraph_state}")