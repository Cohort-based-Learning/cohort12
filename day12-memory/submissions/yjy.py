# =========================================================
# Pokemon LangGraph Adventure (Gemini 2.5 Flash Edition)
# =========================================================
import operator
from typing import Annotated, List, TypedDict, Optional, Union
from typing_extensions import NotRequired

# Google Gemini 임포트
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    BaseMessage,
    RemoveMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

import os
from dotenv import load_dotenv

# 1. 환경 변수 로드 (이게 없거나, 모델 정의보다 밑에 있으면 에러 남)
load_dotenv()

# ---------------------------------------------------------
# 1. 모델 설정 (Gemini 2.5 Flash)
# ---------------------------------------------------------
# API Key는 환경 변수 GOOGLE_API_KEY에 설정되어 있어야 합니다.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# =========================================================
# 2. Battle Subgraph (전투 시스템)
# =========================================================

class BattleState(TypedDict):
    # 부모와 공유하거나 내부에서 쓸 데이터
    battle_result: str
    player_hp: int
    enemy_hp: int
    enemy_name: str
    # 로그는 덮어쓰지 않고 이어붙이기 위해 operator.add 사용
    log: Annotated[List[str], operator.add]

def player_turn(state: BattleState):
    """플레이어 턴: 인터럽트를 걸어 사용자 입력을 받음"""

    # 여기서 실행이 멈추고 사용자에게 값을 요청함
    skill = interrupt(f"[{state['enemy_name']} 체력: {state['enemy_hp']}] 어떤 기술을 쓸까? (전기/몸통박치기/도망)")

    # --- Resume 후 실행되는 부분 ---
    log_entry = []

    if skill == "도망":
        return {
            "battle_result": "escape",
            "log": ["🏃 플레이어가 도망쳤다!"]
        }

    dmg = 35 if skill == "전기" else 15
    new_hp = state["enemy_hp"] - dmg

    log_entry.append(f"⚡ 피카츄의 {skill} 공격! (데미지: {dmg})")

    return {"enemy_hp": new_hp, "log": log_entry}

def enemy_turn(state: BattleState):
    """적 턴: 자동 진행"""
    if state["enemy_hp"] <= 0:
        return {
            "battle_result": "win",
            "log": [f"🌟 {state['enemy_name']}이(가) 쓰러졌다! 승리!"]
        }

    dmg = 10
    new_hp = state["player_hp"] - dmg

    return {
        "player_hp": new_hp,
        "log": [f"💢 {state['enemy_name']}의 반격! (내 체력: {new_hp})"]
    }

def check_battle_end(state: BattleState):
    """종료 조건 확인"""
    if state.get("battle_result") in ["win", "escape"]:
        return END
    if state["player_hp"] <= 0:
        return END # 패배 시에도 종료
    return "player_turn" # 안 끝났으면 다시 플레이어 턴

# 서브그래프 조립
battle_builder = StateGraph(BattleState)
battle_builder.add_node("player_turn", player_turn)
battle_builder.add_node("enemy_turn", enemy_turn)

battle_builder.add_edge(START, "player_turn")
battle_builder.add_edge("player_turn", "enemy_turn")
battle_builder.add_conditional_edges(
    "enemy_turn",
    check_battle_end,
    {
        "player_turn": "player_turn",
        END: END
    }
)

# ★ checkpointer=True가 있어야 부모 그래프와 연결될 때 상태 관리가 가능
battle_subgraph = battle_builder.compile(checkpointer=True)


# =========================================================
# 3. Main Graph (모험 및 기억 관리)
# =========================================================

class MainState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    summary: str
    location: str

    # ★ 수정됨: 서브그래프와 데이터를 공유하려면 MainState에도 키가 있어야 함
    player_hp: int
    enemy_hp: int
    enemy_name: str
    battle_result: str
    log: Annotated[List[str], operator.add]

def adventure_node(state: MainState):
    """스토리 진행 노드"""
    summary = state.get("summary", "모험을 막 시작했다.")

    system_prompt = f"""
    당신은 '포켓몬스터' 게임의 내레이터(Game Master)입니다.
    
    [현재 상태]
    - 위치: {state.get('location', '태초마을')}
    - 지난 줄거리: {summary}
    
    사용자의 행동에 반응하여 짧고 생동감 넘치게 묘사하세요.
    만약 사용자가 '풀숲'으로 가거나 위험한 곳에 가면 "야생 포켓몬이 나타났다!"라고 반드시 언급하세요.
    """

    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}

def router(state: MainState):
    """대화 내용을 보고 배틀 진입 여부 결정"""
    last_msg = state["messages"][-1].content
    if "야생 포켓몬" in last_msg or "승부" in last_msg:
        return "prepare_battle"
    return "memory_manager"

def prepare_battle(state: MainState):
    """배틀 시작 전 초기값 설정"""
    print("\n⚔️ [시스템] 야생의 꼬렛이 나타났다! 배틀 모드로 전환합니다.")
    return {
        "player_hp": 100,
        "enemy_hp": 60,
        "enemy_name": "꼬렛",
        "battle_result": "ready",
        "log": [] # 로그 초기화
    }

def handle_battle_result(state: MainState):
    """배틀이 끝나고 결과 처리"""
    result = state.get("battle_result")

    # 서브그래프에서 쌓인 로그를 가져옴
    battle_logs = "\n".join(state.get("log", []))

    msg_content = ""
    if result == "win":
        msg_content = "배틀에서 멋지게 승리했다! 경험치를 얻었다."
    elif result == "escape":
        msg_content = "무사히 도망쳤다."
    else:
        msg_content = "눈앞이 깜깜해졌다... (패배)"

    # 배틀 로그는 시스템 메시지로 사용자에게 보여줌
    full_msg = f"[배틀 기록]\n{battle_logs}\n\n[결과]: {msg_content}"

    return {"messages": [HumanMessage(content="(배틀 종료됨)"), SystemMessage(content=full_msg)]}

def memory_manager(state: MainState):
    """메시지 요약 및 삭제 (Context Window 관리)"""
    msgs = state["messages"]

    # 메시지가 6개 이하면 정리 안 함
    if len(msgs) <= 6:
        return {}

    print("\n💾 [시스템] 기억 용량 초과! 오래된 대화를 요약합니다...")

    # 요약 수행
    summary_prompt = f"""
    지금까지의 대화 내용을 다음 모험을 위해 한 문단으로 요약해줘.
    기존 요약: {state.get('summary', '')}
    """
    # Gemini에게 요약 요청
    # 주의: invoke에 메시지 리스트로 전달
    summary_res = llm.invoke([
        SystemMessage(content=summary_prompt),
        HumanMessage(content=str(msgs))
    ])

    # 오래된 메시지 삭제 (최근 2개 + 시스템 메시지 제외하고 삭제)
    # 실제로는 RemoveMessage를 사용하여 LangGraph가 처리하게 함
    delete_actions = [RemoveMessage(id=m.id) for m in msgs[:-2] if isinstance(m, (HumanMessage, SystemMessage)) == False]

    return {"summary": summary_res.content, "messages": delete_actions}

# 메인 그래프 조립
builder = StateGraph(MainState)

builder.add_node("adventure", adventure_node)
builder.add_node("prepare_battle", prepare_battle)
builder.add_node("battle_subgraph", battle_subgraph) # 서브그래프 탑재
builder.add_node("battle_result", handle_battle_result)
builder.add_node("memory_manager", memory_manager)

builder.add_edge(START, "adventure")

builder.add_conditional_edges(
    "adventure",
    router,
    {
        "prepare_battle": "prepare_battle",
        "memory_manager": "memory_manager"
    }
)

builder.add_edge("prepare_battle", "battle_subgraph")
builder.add_edge("battle_subgraph", "battle_result")
builder.add_edge("battle_result", "memory_manager")
builder.add_edge("memory_manager", END)

app = builder.compile(checkpointer=InMemorySaver())

# =========================================================
# 🚀 실행 및 Interrupt 처리 로직
# =========================================================

def run_game_loop():
    thread_id = "ash_ketchum_ver3" # ID 변경 (새로운 마음으로 시작)
    config = {"configurable": {"thread_id": thread_id}}

    print(f"🎮 포켓몬 모험을 시작합니다! (ID: {thread_id})")

    while True:
        try:
            user_input = input("\n👤 지우(User): ")
            if user_input.lower() in ["quit", "exit"]:
                break

            # 1. 최초 실행
            events = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config
            )

            # 2. 결과 출력 (일반 대화)
            if "messages" in events and events["messages"]:
                last_msg = events["messages"][-1].content
                print(f"🤖 시스템/AI: {last_msg}")

            # 3. Interrupt 감지 루프
            while True:
                snapshot = app.get_state(config)

                # 더 이상 실행할 태스크가 없으면 루프 탈출
                if not snapshot.next:
                    break

                task = snapshot.tasks[0]
                if task.interrupts:
                    # Interrupt 값(질문) 가져오기
                    question = task.interrupts[0].value
                    print(f"\n✋ [인터럽트 발생] {question}")

                    # 사용자 입력 받기 (Resume)
                    answer = input("   > 선택: ")

                    # Command를 사용해 재개
                    events = app.invoke(Command(resume=answer), config)

                    # ★★★ [수정된 부분] 재개 후 결과 출력 ★★★
                    # 배틀 로그나 최종 승리 메시지가 여기에 담겨 옴
                    if "messages" in events and events["messages"]:
                        last_msg = events["messages"][-1].content
                        print(f"🤖 시스템/AI: {last_msg}")

                else:
                    break

        except Exception as e:
            print(f"오류 발생: {e}")
            break

if __name__ == "__main__":
    run_game_loop()