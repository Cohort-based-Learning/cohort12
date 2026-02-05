"""
██████╗ ██╗ ██████╗  █████╗  ██████╗██╗  ██╗ █████╗ ██████╗ 
██╔════╝ ██║██╔════╝ ██╔══██╗██╔════╝██║  ██║██╔══██╗██╔══██╗
██║  ███╗██║██║  ███╗███████║██║     ███████║███████║██║  ██║
██║   ██║██║██║   ██║██╔══██║██║     ██╔══██║██╔══██║██║  ██║
╚██████╔╝██║╚██████╔╝██║  ██║╚██████╗██║  ██║██║  ██║██████╔╝
 ╚═════╝ ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ 
                                User Interactive Edition
"""
# 한글 주석: 기가채드 페르소나와 메모리 전략(Trim/Summarize)을 선택할 수 있는 대화형 봇 서비스 파일입니다.
import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import START, StateGraph, MessagesState
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, RemoveMessage, SystemMessage

# 환경 변수 로드
load_dotenv()

def clear_screen():
    """터미널 화면을 정리하는 함수"""
    os.system('cls' if os.name == 'nt' else 'clear')

# MongoDB 설정
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    client.server_info()
    # 인프라 연결 로그는 숨김 처리하여 난잡함을 줄임
except Exception as e:
    print(f"❌ MongoDB 연결 실패: {e}")
    sys.exit(1)

# 기가채드 모델 및 페르소나 설정
model = init_chat_model("gemini-flash-latest", model_provider="google_genai")
# 마크다운 강조 기호를 절대 쓰지 않도록 지침 추가
GIGACHAD_PROMPT = SystemMessage(content=(
    "넌 기가채드(GigaChad)다. 근육질이고 자신감이 넘치며 압도적이다. "
    "말은 짧고 굵게 명언처럼 한다. 나약함은 죄악이다. 무조건 한국어로 대답해라. "
    "중요: 절대로 ** 또는 _ 같은 마크다운 강조 기호를 사용하지 마라. 오직 평문으로만 대답해라."
))

# ------------------------------------------------------------------
# 전략 1: Trim (Focus Mode) - 최신 대화에만 집중
# ------------------------------------------------------------------
def get_trim_graph():
    def call_model_trim(state: MessagesState):
        trimmed_messages = trim_messages(  
            state["messages"],
            strategy="last", 
            token_counter=count_tokens_approximately, 
            max_tokens=80, 
            start_on="human", 
            end_on=("human", "tool"), 
        )
        messages = [GIGACHAD_PROMPT] + trimmed_messages
        response = model.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model_trim)
    builder.add_edge(START, "call_model")
    return builder.compile(checkpointer=MongoDBSaver(client))

# ------------------------------------------------------------------
# 전략 2: Summarize (Wisdom Mode) - 과거를 압축하여 기억
# ------------------------------------------------------------------
class SummaryState(MessagesState):
    summary: str

def get_summary_graph():
    def summarize_conversation(state: SummaryState):
        summary = state.get("summary", "")
        if summary:
            summary_message = f"이전 요약: {summary}\n\n위의 새로운 대화를 포함하여 요약을 확장하라. 핵심만 짧게 기가채드 스타일로 요약하라."
        else:
            summary_message = "위의 대화를 기가채드 스타일로 핵심만 요약하라:"
        
        response = model.invoke(state["messages"] + [HumanMessage(content=summary_message)])
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}

    def call_model_summary(state: SummaryState):
        messages = state["messages"]
        if state.get("summary"):
            messages = [HumanMessage(content=f"[이전의 기억]: {state['summary']}")] + messages
        
        messages = [GIGACHAD_PROMPT] + messages
        response = model.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(SummaryState)
    builder.add_node("summarize", summarize_conversation)
    builder.add_node("call_model", call_model_summary)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", "summarize")
    return builder.compile(checkpointer=MongoDBSaver(client))

# ------------------------------------------------------------------
# 메인 루프
# ------------------------------------------------------------------
def main():
    clear_screen()
    print(__doc__)
    print(" [1] 집중 훈련 (Trim) | [2] 지혜의 샘 (Summarize)")
    
    choice = input("\n > 훈련 방식을 선택하라 (1/2): ").strip()
    user_id = input(" > ID를 입력하라 (default: legend): ").strip() or "legend"
    
    if choice == "1":
        graph = get_trim_graph()
        mode_name = "Focus (Trim)"
    else:
        graph = get_summary_graph()
        mode_name = "Wisdom (Summarize)"

    config = {"configurable": {"thread_id": user_id}}
    
    clear_screen()
    print(f"--- {mode_name} 모드 활성화 (ID: {user_id}) ---")
    print(" [종료하려면 'q'를 입력하라]\n")

    while True:
        try:
            user_input = input(" [You]: ").strip()
            if user_input.lower() in ["q", "exit"]:
                print("\n GigaChad: 오늘 훈련은 여기까지다. 나약해지지 마라.\n")
                break
            
            if not user_input:
                continue

            # 그래프 실행
            events = graph.stream(
                {"messages": [HumanMessage(content=user_input)]}, 
                config, 
                stream_mode="values"
            )
            
            last_msg = None
            for event in events:
                if "messages" in event:
                    last_msg = event["messages"][-1]
            
            if last_msg and last_msg.type == "ai":
                print(f" [GigaChad]: {last_msg.content}\n")
            
        except Exception as e:
            print(f"\n [Error]: {e}")
            break

if __name__ == "__main__":
    main()
