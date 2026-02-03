import asyncio
import random
from typing import TypedDict, Annotated, Union
from operator import add

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer

# 환경변수 로드
load_dotenv()

# =========================================================
# 1. 공통 상태 정의 (State)
# =========================================================
class GameState(TypedDict):
    history: Annotated[list, add] # 대화 내역 (누적)
    hp: int                       # 몬스터 체력
    action: str                   # 유저의 행동
    dice_result: int              # 주사위 결과
    status: str                   # 현재 게임 상태

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# =========================================================
# 2. ⚔️ 자식 그래프 (Combat Subgraph) - 전투 시스템
# =========================================================
# Node 2-1: 주사위 굴리기 (Custom Stream)
async def roll_dice_node(state: GameState):
    writer = get_stream_writer() # 📢 방송 장비 ON

    writer("🎲 [전투 시스템] 주사위를 집어 듭니다...")
    await asyncio.sleep(0.3)

    writer("🎲 [전투 시스템] 굴러갑니다... 또르르...")
    await asyncio.sleep(0.3)

    # D20 주사위
    result = random.randint(1, 20)

    writer(f"🎲 [전투 시스템] 짠! 눈금: [{result}]")
    await asyncio.sleep(0.3)

    # 데미지 계산
    damage = 0
    if result >= 18:
        writer("🔥 [전투 시스템] 크리티컬 히트!! (데미지 30)")
        damage = 30
    elif result >= 10:
        writer("⚔️ [전투 시스템] 공격 적중! (데미지 10)")
        damage = 10
    else:
        writer("💦 [전투 시스템] 빗나감... (데미지 0)")
        damage = 0

    new_hp = max(0, state["hp"] - damage)
    return {"dice_result": result, "hp": new_hp}

# Node 2-2: 나레이션 (Messages Stream)
async def combat_narrator_node(state: GameState):
    dice = state["dice_result"]
    action = state["action"]
    hp = state["hp"]

    prompt = (
        f"플레이어 행동: '{action}'. 주사위 결과: {dice}(20만점). "
        f"남은 몬스터 HP: {hp}. "
        "이 상황을 판타지 소설처럼 한 문장으로 아주 박진감 넘치게 묘사해줘."
    )

    # LLM 스트리밍
    response = await llm.ainvoke([SystemMessage(content="TRPG 마스터"),
                                  HumanMessage(content=prompt)])

    return {"history": [response]}

# 자식 그래프 조립
combat_workflow = StateGraph(GameState)
combat_workflow.add_node("roll_dice", roll_dice_node)
combat_workflow.add_node("combat_narrator", combat_narrator_node)

combat_workflow.add_edge(START, "roll_dice")
combat_workflow.add_edge("roll_dice", "combat_narrator")
# (END는 자동으로 연결됨)

combat_graph = combat_workflow.compile()


# =========================================================
# 3. 🏰 부모 그래프 (Game World) - 전체 흐름
# =========================================================
# Node 1: 인카운터 (시작)
def encounter_node(state: GameState):
    return {"status": "encounter_started"}

# Node 3: 전투 종료 확인
def check_result_node(state: GameState):
    if state["hp"] == 0:
        return {"status": "victory"}
    else:
        return {"status": "continue"}

# 부모 그래프 조립
parent_workflow = StateGraph(GameState)

parent_workflow.add_node("encounter", encounter_node)
parent_workflow.add_node("combat_phase", combat_graph) # 자식 그래프를 노드로 등록!
parent_workflow.add_node("check_result", check_result_node)

parent_workflow.add_edge(START, "encounter")
parent_workflow.add_edge("encounter", "combat_phase")
parent_workflow.add_edge("combat_phase", "check_result")
parent_workflow.add_edge("check_result", END)

parent_graph = parent_workflow.compile()


# =========================================================
# 🚀 게임 시작 (Streaming 실행)
# =========================================================
async def play_game():
    print("\n🏰 [시스템] 던전에 입장했습니다.")
    print("⚔️ [시스템] 야생의 'Null Pointer Exception' 몬스터가 나타났다! (HP: 100)")

    user_action = input("\n행동을 입력하세요 (예: 불꽃 펀치를 날린다): ")
    print("\n" + "="*40)

    inputs = {
        "hp": 100,
        "action": user_action,
        "history": [],
        "dice_result": 0,
        "status": "start"
    }

    # 1. stream_mode 다중 선택 (custom, messages, updates)
    # 2. subgraphs=True (자식 그래프 내부 생중계)
    async for chunk in parent_graph.astream(
            inputs,
            stream_mode=["custom", "messages", "updates"],
            subgraphs=True # 이게 없으면 주사위 굴리는 과정(custom)이 안 보임!
    ):
        # stream_mode가 list라서 'mode'가 붙고,
        # subgraphs=True 옵션 때문에 출처(주소)인 'namespace'까지 붙어서 총 3개가 옴!
        # 구조: (namespace, mode, data)
        namespace,mode, data = chunk

        # 1. Custom: 주사위 굴리는 과정 (자식 그래프에서 쏘아 올림)
        if mode == "custom":
            print(f"   {data}")

            # 2. Messages: AI의 나레이션
        elif mode == "messages":
            # 메타데이터 필터링 없이 출력 (단순화)
            msg, metadata = data # messages 모드는 (msg, metadata)를 줌
            if msg.content:
                print(msg.content, end="", flush=True)

        # 3. Updates: 데이터 변경 확인
        elif mode == "updates":
            # 자식 그래프(combat_phase)에서 올라온 업데이트인지 확인
            # updates 구조: {'노드이름': {'필드': '값'}}

            # (자식 그래프 내부의 roll_dice 노드가 업데이트한 경우)
            if isinstance(data, dict) and "roll_dice" in data:
                new_hp = data['roll_dice']['hp']
                # print(f"\n   -> (데이터 갱신) 몬스터 HP: {new_hp}")
                # (너무 자주 뜨면 지저분하니까 주석 처리, 필요하면 해제)

            # (자식 그래프 전체가 끝나고 부모에게 보고한 경우)
            if isinstance(data, dict) and "combat_phase" in data:
                final_hp = data['combat_phase']['hp']
                print(f"\n\n[System] 턴 종료! 몬스터 남은 체력: {final_hp}")
                print("-" * 40)

if __name__ == "__main__":
    asyncio.run(play_game())