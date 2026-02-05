"""
단기 메모리(Short-term): 에이전트의 state에 넣어, 여러 턴 대화를 가능하게 함
같은 thread 안에서만 이어지는 기억 (대화 히스토리 같은 것)

장기 메모리(Long-term): 세션이 바뀌어도 유지되는 사용자별 / 앱 전체 데이터를 저장
thread가 달라도 유지되는 기억 (유저 프로필/선호/설정 같은 것)
"""

# ---------------------------------------
# Add short-term memory
# ---------------------------------------
# # use in production (postgres, mongodb, redis)
from dotenv import load_dotenv

load_dotenv()

# use in subgraphs
# 서브그래프의 메모리는 기본적으로 부모 그래프가 관리, 필요하면 서브그래프만 따로 독립 메모리를 가질 수 있음 > 멀티에이전트
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict
from langgraph.types import Command, interrupt


class State(TypedDict):
    foo: str


# Subgraph
def subgraph_node_1(state: State):
    answer = interrupt({"question": "continue?"})
    return {"foo": state["foo"] + answer}


subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()
# subgraph = subgraph_builder.compile(checkpointer=True) # 부모 그래프와 별도로 메모리를 관리할 수 있음

# Parent graph
builder = StateGraph(State)
builder.add_node("subgraph_node_1", subgraph)
builder.add_edge(START, "subgraph_node_1")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}

result = graph.invoke({"foo": "a"}, config)
print(result["__interrupt__"])

graph.invoke(Command(resume="X"), config)

history = list(graph.get_state_history(config))

subgraph_namespaces = set()

for snap in history:
    for task in snap.tasks or []:
        state = task.state
        ns = state.get("configurable", {}).get("checkpoint_ns") if state else None
        if ns:
            subgraph_namespaces.add(ns)

results = {}
for ns in sorted(subgraph_namespaces):
    sub_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ns}}
    results[ns] = list(graph.get_state_history(sub_config))

print(results)  # ✅ print 딱 1번


# ---------------------------------------
# Manage short-term memory (LLM에 매 호출마다 그대로 들어가, 토큰 제한에 걸릴 수 있음)
# ---------------------------------------

##1. Trim ##
print("\n========trim=========\n")
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

model = init_chat_model("gpt-5-nano")


def call_model(state: MessagesState):
    messages = trim_messages(
        state["messages"],
        strategy="last",  # 어디를 남길지(last: 마지막, first: 첫번째)
        token_counter=count_tokens_approximately,  # 빠른 근사 토큰 계산
        max_tokens=128,  # 결과 메시지들의 총 토큰 수 ≤ 128
        start_on="human",  # 잘린 결과가 HumanMessage부터 시작
        end_on=(
            "human",
            "tool",
        ),  # 어디까지 끝낼지(ToolMessage 뒤에 AI가 바로 오지 않게 정리)
    )
    # messages = state["messages"] #제한이 없으면 다 기억함
    response = model.invoke(messages)
    return {"messages": [response]}


checkpointer = InMemorySaver()
builder = StateGraph(MessagesState)
builder.add_node(call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
result1 = graph.invoke({"messages": "안녕 내 이름은 김철수야."}, config)
result1["messages"][-1].pretty_print()
config = {"configurable": {"thread_id": "1"}}
result2 = graph.invoke({"messages": "내 취미는 프리다이빙이야."}, config)
result2["messages"][-1].pretty_print()


final_response = graph.invoke({"messages": "내 이름은 뭐야?"}, config)
final_response["messages"][-1].pretty_print()  # 철수를 잊음

print("\n", [(message.type, message.content) for message in final_response["messages"]])

# 2. Delete ##
print("\n========delete=========\n")
from langchain.messages import RemoveMessage


def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        print("\n==메시지 개수가 제한을 초과했습니다!==\n")
        # remove the earliest two messages
        return {
            "messages": [RemoveMessage(id=m.id) for m in messages[:2]]
        }  # RemoveMessage(id=REMOVE_ALL_MESSAGES) : 모든 메시지 삭제


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


builder = StateGraph(MessagesState)
builder.add_sequence([call_model, delete_messages])
builder.add_edge(START, "call_model")

checkpointer = InMemorySaver()
app = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "2"}}
for event in app.stream(
    {"messages": [{"role": "user", "content": "안녕!"}]},
    config,
    stream_mode="values",
):
    print(
        "first invoke",
        [(message.type, message.content) for message in event["messages"]],
    )

for event in app.stream(
    {"messages": [{"role": "user", "content": "내 이름이 뭐야?"}]},
    config,
    stream_mode="values",
):
    print(
        "second invoke",
        [(message.type, message.content) for message in event["messages"]],
    )

# checkpointer.delete_thread(thread_id) #thread 삭제

##3. Summarize ##
# pip install langmem
print("\n========summarize=========\n")
from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import InMemorySaver

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, RemoveMessage

model = init_chat_model("gpt-5-nano")


class State(MessagesState):
    summary: str


# 요약 노드
def summarize_conversation(state: State):
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is a summary of the conversation to date:\n{summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    # 메시지 정리 (최근 2개만 유지)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    return {
        "summary": response.content,
        "messages": delete_messages,
    }


def call_model(state: State):
    messages = state["messages"]

    if state.get("summary"):
        messages = [
            HumanMessage(content=f"(Conversation summary)\n{state['summary']}")
        ] + messages

    response = model.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("summarize", summarize_conversation)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "summarize")

graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "1"}}

graph.invoke({"messages": "안녕"}, config)
final = graph.invoke({"messages": "내 이름은 뭐야?"}, config)

final["messages"][-1].pretty_print()
print("\nSummary:", final["summary"])

# print(f"\n\n ========history=========\n:{list(graph.get_state_history(config))}")
