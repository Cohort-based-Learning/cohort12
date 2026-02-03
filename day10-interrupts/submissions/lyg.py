#pip install langgraph.checkpoint.sqlite

"""
interrupt() : 그래프 실행 정지>체크포인터에 state, node, 다음 실행 위치 전부 저장
result["__interrupt__"] : interrupt() 호출 결과
graph.invoke(Command(resume=True), config=config) : interrupt() 호출했던 그 자리로 돌아가서 interrupt()의 반환값을 True로 주고 다시 실행
"""

#-----------------------------------------
#Approve or reject(승인/거부)
#-----------------------------------------
print(f"\n#Approve or reject")
from typing import Literal, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI

class ApprovalState(TypedDict):
    action_details: str
    status: Optional[Literal["pending", "approved", "rejected"]]


def approval_node(state: ApprovalState) -> Command[Literal["proceed", "cancel"]]:
    # Expose details so the caller can render them in a UI
    decision = interrupt({
        "question": "승인하시겠습니까?",
        "details": state["action_details"],
    })

    # Route to the appropriate node after resume
    return Command(goto="proceed" if decision else "cancel")


def proceed_node(state: ApprovalState):
    print("*승인 노드로 왔습니다.")
    return {"status": "approved"}


def cancel_node(state: ApprovalState):
    print("*거부 노드로 왔습니다.")
    return {"status": "rejected"}


builder = StateGraph(ApprovalState)
builder.add_node("approval", approval_node)
builder.add_node("proceed", proceed_node)
builder.add_node("cancel", cancel_node)
builder.add_edge(START, "approval")
builder.add_edge("proceed", END)
builder.add_edge("cancel", END)

# Use a more durable checkpointer in production
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "approval-123"}}
initial = graph.invoke(
    {"action_details": "500만원 송금", "status": "pending"},
    config=config,
)
print(f"interrupt() 호출 결과: {initial['__interrupt__']}")

# Resume with the decision; True routes to proceed, False to cancel
ans = input("승인? (True/False): ").strip()
resumed = graph.invoke(Command(resume=ans), config=config) # True/False
print(f"resume 결과: {resumed["status"]}")


#-----------------------------------------
#Review and edit state(내용 리뷰 및 수정)
#-----------------------------------------
print(f"\n#Review and edit state")
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class ReviewState(TypedDict):
    generated_text: str


def review_node(state: ReviewState):
    # Ask a reviewer to edit the generated content
    updated = interrupt({
        "instruction": "Review and edit this content",
        "content": state["generated_text"],
    })
    return {"generated_text": updated}


builder = StateGraph(ReviewState)
builder.add_node("review", review_node)
builder.add_edge(START, "review")
builder.add_edge("review", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "review-42"}}
initial = graph.invoke({"generated_text": "Initial draft"}, config=config)
print(f"interrupt() 호출 결과: {initial['__interrupt__']}")

# Resume with the edited text from the reviewer
final_state = graph.invoke(
    Command(resume="Improved draft after review"),
    config=config,
)
print(f"resume 결과: {final_state["generated_text"]}")


#-----------------------------------------
#Interrupts in tools(툴 내부에서 Interrupts 적용)
#-----------------------------------------
print(f"\n#Interrupts in tools")
from typing import TypedDict

from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langchain.messages import ToolMessage

from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    messages: list[dict]


# 툴 함수 안에 interrupts를 적용할 수 있음
@tool
def send_email(to: str, subject: str, body: str):
    """Send an email to a recipient."""

    # Pause before sending; payload surfaces in result["__interrupt__"]
    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "Approve sending this email?",
    })

    if response.get("action") == "approve":
        final_to = response.get("to", to)
        final_subject = response.get("subject", subject)
        final_body = response.get("body", body)

        # 실제 발송한다면, 여기에 구현
        print(f"[send_email] to={final_to} subject={final_subject} body={final_body}")
        return f"Email sent to {final_to}"

    # else
    return "Email cancelled by user"

# model = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools([send_email])
model = ChatOpenAI(model="gpt-4o-mini").bind_tools([send_email])

# tool 이름으로 실제 tool 객체를 찾기 위한 딕셔너리
tools_by_name = {"send_email": send_email}


def agent_node(state: AgentState):
    # LLM may decide to call the tool; interrupt pauses before sending
    result = model.invoke(state["messages"])
    return {"messages": state["messages"] + [result]}


def tools_node(state: AgentState):
    """Tool을 실제로 실행하는 노드 - 여기서 interrupt() 발생"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])  # 여기서 send_email이 실행되고 interrupt() 발생
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: AgentState):
    """LLM이 tool을 호출했는지 확인"""
    last_message = state["messages"][-1]
    # AIMessage에 tool_calls가 있으면 tools 노드로, 없으면 END로
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "agent")
# 조건부 엣지: agent 실행 후 tool 호출 여부에 따라 분기
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", END)

# with 문을 사용하여 컨텍스트 매니저로 checkpointer 생성
with SqliteSaver.from_conn_string("tool-approval.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "email-workflow"}}
    initial = graph.stream(  
    {"messages": [
                {
                    "role": "user", 
                    "content": "alice@example.com에게 제목 '다음 주 회의 안내', 내용 '다음 주 월요일 오전 10시에 회의가 있습니다.'로 이메일을 보내주세요."
                }
            ]},
    config=config,
    stream_mode="debug"
    )
    for chunk in initial:
        print(chunk)
    
        if "__interrupt__" in chunk:
            interrupt_payload = chunk["__interrupt__"]
            print("\n\ninterrupt() 호출 결과:", interrupt_payload,"\n\n")
            break

    # 승인된 상태로 재개하고 선택적으로 인수를 수정
    print("\n\n")
    ans = input("승인? (approve/reject): ").strip()
    print()
    resumed = graph.stream(
        Command(resume={"action": ans, "subject": "Updated subject"}),
        config=config,
        stream_mode="debug"
    )
    for chunk in resumed:
        print(chunk)


#-----------------------------------------
#Validating human input
#-----------------------------------------
print(f"\n#Validating human input")
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class FormState(TypedDict):
    age: int | None


def get_age_node(state: FormState):
    prompt = "나이가 어떻게 되세요?"

    while True: #유효한 입력을 받을 때까지 반복
        answer = interrupt(prompt)

        if isinstance(answer, int) and answer > 0:
            return {"age": answer}

        prompt = f"'{answer}'는 올바른 나이가 아닙니다. 양수를 입력해주세요."


builder = StateGraph(FormState)
builder.add_node("collect_age", get_age_node)
builder.add_edge(START, "collect_age")
builder.add_edge("collect_age", END)

with SqliteSaver.from_conn_string("forms.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "form-1"}}

    print("\n--- 1) 첫 실행 (질문 발생) ---")
    for event in graph.stream({"age": None}, config=config, stream_mode="values"):
        print(event)

    print("\n--- 2) 잘못된 입력(resume='thirty') ---")
    for event in graph.stream(Command(resume="thirty"), config=config, stream_mode="values"):
        print(event)

    print("\n--- 3) 올바른 입력(resume=30) ---")
    for event in graph.stream(Command(resume=30), config=config, stream_mode="values"):
        print(event)



""" Appendix
#-----------------------------------------
#Rules of interrupts
#interrupt를 호출하면 실행이 일시 중단되었다가 재개 시 노드 전체가 처음부터 다시 실행
#-----------------------------------------

#1. 규칙 1: interrupt를 try/except로 감싸지 마라
# interrupt는 try 밖에 두고, 실패 가능 코드만 try로 감싼다
def node_a(state: State):
    interrupt("What's your name?")  # 먼저 안전하게 '중단'

    try:
        fetch_data()  # 실패할 수 있는 코드만 try로 감싼다
    except Exception as e:
        print(e)

    return state

#2. 규칙 2: 한 노드 안에서 interrupt 호출 “순서”를 바꾸지 마라
# 항상 같은 순서로 interrupt 호출
# 노드 내에서 인터럽트 호출을 조건부로 건너뛰지 않기, 인터럽트 호출을 반복하지 않기
def node_a(state: State):
    name = interrupt("What's your name?")
    age = interrupt("What's your age?")
    city = interrupt("What's your city?")

    return {"name": name, "age": age, "city": city}

#3. 규칙 3: interrupt에는 “단순하고 직렬화 가능한 값”만 주고받아라
# interrupt payload는 JSON으로 저장 가능하게 만들기
def node_a(state: State):
    name = interrupt("What's your name?")
    count = interrupt(42)
    approved = interrupt(True)

    return {"name": name, "count": count, "approved": approved}

#4. 규칙 4: interrupt 앞에 있는 코드는 “다시 실행돼도 문제 없는 행동”만 해야 한다
# interrupt 이전엔 “중복 실행돼도 안전한 작업”만
def node_a(state: State):
    db.upsert_user(
        user_id=state["user_id"],
        status="pending_approval"
    )

    approved = interrupt("Approve this change?")
    return {"approved": approved}

#5. 규칙 5: 서브그래프 안에서 interrupt가 걸리면, 부모 노드도, 서브그래프 노드도 “처음부터 다시 실행"
def node_in_parent_graph(state: State):
    some_code()  # <-- 재개(resume) 시 다시 실행됨
    subgraph_result = subgraph.invoke(some_input)
    # ...
def node_in_subgraph(state: State):
    some_other_code()  # <-- 재개 시 다시 실행됨
    result = interrupt("What's your name?")
    # ...
"""


"""
#-----------------------------------------
#Debugging with interrupts
#LangGraph 그래프를 실행 중 특정 노드 전·후에서 일부러 멈추게 해서 상태를 확인하며 한 단계씩 실행해보는 디버깅 방식
*Langgraph studio에서 테스트 할 수 있음
#-----------------------------------------

graph = builder.compile(
    interrupt_before=["node_a"],  
    interrupt_after=["node_b", "node_c"],  
    checkpointer=checkpointer,
)

# Pass a thread ID to the graph
config = {
    "configurable": {
        "thread_id": "some_thread"
    }
}

# Run the graph until the breakpoint
graph.invoke(inputs, config=config)  

# Resume the graph
graph.invoke(None, config=config)
"""