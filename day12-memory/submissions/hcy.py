import uuid
from typing import TypedDict, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage, SystemMessage
from langchain_core.messages.utils import trim_messages

# 환경 변수 로드
load_dotenv()
model = ChatOpenAI(model="gpt-4o", temperature=0.7)

print("🐧 펭벨로퍼의 기억 관리 3종 세트 실습 시작! 🐧\n")

# ====================================================
# 1. ✂️ Trim (자르기): MZ세대 급식체 번역기
# ====================================================

print("\n=== [1] Trim 실습: MZ세대 번역기 ===")

def mz_translator_node(state: MessagesState):
    # 1. Trim (최신 대화 유지)
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=len,
        max_tokens=2,
        start_on="human",
        include_system=True,
        allow_partial=False
    )

    # 2. 프롬프트 (질문에는 대답, 나머진 번역)
    system_prompt = """
    너는 MZ세대 급식체 번역기다.
    사용자의 말을 요즘 유행하는 말투로 바꿔라.
    
    [규칙]
    1. 사용자가 "방금 내가 뭐라고 했어?" 같은 '기억력 테스트' 질문을 하면,
       번역하지 말고 네가 기억하는 대로 정직하게 대답해라.
    2. 기억이 안 나면 "몰?루", "기억 안 나는데?"라고 해라.
    3. 그 외의 모든 말은 급식체로 번역해라.
    """

    prompt = [SystemMessage(content=system_prompt)] + trimmed_messages
    response = model.invoke(prompt)
    return {"messages": [response]}

workflow_trim = StateGraph(MessagesState)
workflow_trim.add_node("translator", mz_translator_node)
workflow_trim.add_edge(START, "translator")
app_trim = workflow_trim.compile(checkpointer=InMemorySaver())

config_trim = {"configurable": {"thread_id": "mz_bot_1"}}

# --- [1] 실행 ---
print("User: 안녕하세요, 반갑습니다.")
app_trim.invoke({"messages": [HumanMessage(content="안녕하세요, 반갑습니다.")]}, config_trim)
print("Bot:", app_trim.get_state(config_trim).values["messages"][-1].content)

print("\nUser: 이거 정말 맛있네요.")
app_trim.invoke({"messages": [HumanMessage(content="이거 정말 맛있네요.")]}, config_trim)
print("Bot:", app_trim.get_state(config_trim).values["messages"][-1].content)

print("\n👉 확인: 이전 대화를 기억하는지 테스트")
final = app_trim.invoke({"messages": [HumanMessage(content="방금 내가 뭐라고 했게?")]}, config_trim)
print("Bot:", final["messages"][-1].content)


# ====================================================
# 2. 🗑️ Delete (지우기): 비밀요원 (Burn after reading)
# ====================================================

print("\n\n=== [2] Delete 실습: 비밀요원 (메시지 소각) ===")

def secret_agent_node(state: MessagesState):
    # 프롬프트로 보안 유지 지시
    prompt = [SystemMessage(content="너는 비밀요원이다. 지령을 접수했다는 말만 하고, 보안을 위해 **절대 지령 내용을 다시 언급하지 마라**. '접수 완료. 메시지 소각합니다.' 라고만 답해.")] + state["messages"]

    response = model.invoke(prompt)
    return {"messages": [response]}

def burn_message_node(state: MessagesState):
    # 사용자 메시지(HumanMessage)만 골라서 삭제
    msgs_to_delete = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            msgs_to_delete.append(RemoveMessage(id=m.id))

    if msgs_to_delete:
        print(f"🔥 [System] {len(msgs_to_delete)}개의 보안 메시지를 소각했습니다.")

    return {"messages": msgs_to_delete}

workflow_del = StateGraph(MessagesState)
workflow_del.add_node("agent", secret_agent_node)
workflow_del.add_node("incinerator", burn_message_node)

workflow_del.add_edge(START, "agent")
workflow_del.add_edge("agent", "incinerator")
workflow_del.add_edge("incinerator", END)

app_del = workflow_del.compile(checkpointer=InMemorySaver())
config_del = {"configurable": {"thread_id": "agent_007"}}

# --- [2] 실행 ---
app_del.invoke({"messages": [HumanMessage(content="타겟은 오늘 밤 8시 강남역에 나타난다.")]}, config_del)

print("\n🔍 [보안 감사] 기록 조회 중...")
history = app_del.get_state(config_del).values["messages"]
for m in history:
    print(f"[{m.type}]: {m.content}")


# ====================================================
# 3. 📝 Summarize (요약): 연애 상담사 (Fix Ver.)
# ====================================================

print("\n\n=== [3] Summarize 실습: 연애 상담사 (Ver. 구구절절) ===")

class CounselorState(MessagesState):
    summary: str

def counseling_node(state: CounselorState):
    # 요약본이 있다면 시스템 메시지로 주입 (컨텍스트 복원)
    summary = state.get("summary", "")
    messages = state["messages"]
    if summary:
        system_msg = SystemMessage(content=f"이전 상담 요약: {summary}")
        messages = [system_msg] + messages

    response = model.invoke(messages)
    return {"messages": [response]}

def summarize_node(state: CounselorState):
    summary = state.get("summary", "")
    new_messages = state["messages"]

    # 메시지가 2개 이상 쌓이면 요약 시작
    if len(new_messages) > 2:
        # ★ 수정된 부분: 메시지 객체를 그대로 넣지 않고, 텍스트로 변환해서 넣음
        # 이렇게 해야 LLM이 '대화'가 아니라 '처리해야 할 텍스트 데이터'로 인식함
        conversation_text = ""
        for m in new_messages:
            role = "User" if isinstance(m, HumanMessage) else "Counselor"
            conversation_text += f"{role}: {m.content}\n"

        prompt = f"""
        당신은 전문 상담 요약가입니다.
        아래의 대화 내용을 바탕으로 기존 요약본을 갱신해주세요.
        
        [기존 요약본]
        {summary if summary else "없음"}
        
        [새로운 대화 내용]
        {conversation_text}
        
        [지시사항]
        1. 구체적인 상황(TMI)보다는 사용자의 '핵심 갈등', '감정 변화', '사건의 본질' 위주로 요약하세요.
        2. 요약문만 출력하세요. (인사말 X)
        """

        summary_llm_msg = model.invoke(prompt)
        new_summary = summary_llm_msg.content

        # 요약 완료 후, 오래된 메시지 삭제 (최근 2개 제외하고 삭제)
        delete_targets = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

        print(f"📝 [Update] 상담 요약 갱신: {new_summary[:40]}...")
        return {"summary": new_summary, "messages": delete_targets}

    return {}

workflow_sum = StateGraph(CounselorState)
workflow_sum.add_node("counselor", counseling_node)
workflow_sum.add_node("summarizer", summarize_node)
workflow_sum.add_edge(START, "counselor")
workflow_sum.add_edge("counselor", "summarizer")
workflow_sum.add_edge("summarizer", END)

app_sum = workflow_sum.compile(checkpointer=InMemorySaver())
config_sum = {"configurable": {"thread_id": "love_clinic_fixed"}}

# --- [3] 실행 ---

# 1. 첫 번째 하소연 (TMI)
long_rant = """
제가 썸남을 한 달 전에 동아리 신입 환영회에서 처음 봤거든요? 
와 근데 진짜 처음엔 뿔테 안경에 체크남방 입고 구석에 박혀있길래 '아.. 진짜 내 스타일 아니다' 생각했단 말이에요.
근데 저번 주에 도서관에서 밤새는데 새벽 2시에 핫식스랑 샌드위치를 사들고 나타난 거예요!! 
알고 보니 인스타 보고 사 왔대요... 그때 안경 벗은 거 보고 심쿵해서 썸 타기 시작했는데...
어제부터 갑자기 카톡 1이 안 사라져요. 인스타는 하는데... 이거 어장관리인가요? 너무 억울해요 ㅠㅠ
"""
print("\nUser: (구구절절 사연 발사... 🚀)")
app_sum.invoke({"messages": [HumanMessage(content=long_rant)]}, config_sum)

# 2. 두 번째 하소연 (여기서 요약 트리거 작동 예상)
print("\nUser: 근데 또 막상 만나자고 하면 바로 나오거든요? 뭐 하자는 건지 모르겠어요.")
app_sum.invoke({"messages": [HumanMessage(content="근데 또 막상 만나자고 하면 바로 나오거든요? 뭐 하자는 건지 모르겠어요.")]}, config_sum)

# 3. 기억력 테스트 (원본 메시지는 삭제되고 요약본만 남은 상태)
print("\nUser: 상담사님, 제 썸남이랑 첫 만남 기억나요? 안경 얘기 기억나요?")
final_counsel = app_sum.invoke({"messages": [HumanMessage(content="제 썸남이랑 첫 만남 기억나요? 안경 얘기 기억나요?")]}, config_sum)
print(f"Bot: {final_counsel['messages'][-1].content}")

# 4. 실제 요약 데이터 확인
print("\n🔍 [상담 일지(Summary)] 최종 조회:")
print(app_sum.get_state(config_sum).values.get("summary"))