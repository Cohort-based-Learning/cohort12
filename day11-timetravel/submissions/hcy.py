import uuid
import random
from typing_extensions import TypedDict, NotRequired
from dotenv import load_dotenv

# LangGraph & LangChain 임포트
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI  # OpenAI 모델 사용 시
# from langchain_anthropic import ChatAnthropic # Claude 사용 시

# 0. 환경 변수 로드
load_dotenv()

# --- [1] 모델 및 상태 정의 ---

# 모델 선택
model = ChatOpenAI(model="gpt-4o", temperature=0.7)

class LifeState(TypedDict):
    name: str           # 이름
    talent: str         # 타고난 재능
    career: str         # 직업
    assets: int         # 자산
    happiness: int      # 행복도
    biography: str      # 인생 이야기

# --- [2] 노드(Node) 함수 정의 ---

def childhood(state: LifeState):
    """유년기: 재능 랜덤 부여"""
    talents = ["코딩", "트로트 노래", "주식 투자", "라면 끓이기", "멍 때리기"]
    given_talent = random.choice(talents)

    # LLM에게 짧은 스토리 요청
    msg = model.invoke(f"이름은 {state['name']}, 타고난 재능은 '{given_talent}'입니다. 유년기 에피소드를 한 문장으로 지어주세요.")

    print(f"\n👶 [유년기] '{given_talent}' 재능을 가지고 태어났습니다.")
    return {"talent": given_talent, "biography": msg.content}

def youth_choice(state: LifeState):
    """청년기: 직업 선택 (1회차는 무조건 공무원으로 고정)"""
    # 1회차의 비극: 재능 무시하고 안정적인 선택
    chosen_career = "9급 공무원"

    msg = model.invoke(f"{state['biography']}\n\n이 아이는 자라서 재능({state['talent']})과 상관없이 '{chosen_career}'이 됩니다. 그 이유를 한 문장으로 설명해주세요.")

    print(f"🧑‍🎓 [청년기] 현실과 타협하여 '{chosen_career}'을(를) 선택했습니다.")
    return {"career": chosen_career, "biography": msg.content}

def old_age(state: LifeState):
    """노년기: 인생 결산"""
    # 재능과 직업의 일치 여부에 따른 결과 시뮬레이션
    final_assets = 0
    final_happiness = 0

    # 간단한 로직: 재능과 직업이 연관되면 대박, 아니면 쪽박
    is_matched = False
    if state['talent'] == "코딩" and "창업" in state['career']: is_matched = True
    elif state['talent'] == "트로트 노래" and "가수" in state['career']: is_matched = True
    elif state['talent'] == "주식 투자" and "펀드" in state['career']: is_matched = True
    elif state['talent'] == "라면 끓이기" and "요리사" in state['career']: is_matched = True
    elif state['talent'] == "멍 때리기" and "유튜버" in state['career']: is_matched = True

    if is_matched:
        final_assets = random.randint(50, 100) # 50억~100억
        final_happiness = random.randint(90, 100)
        result_desc = "재능을 꽃피워 엄청난 성공을 거두었습니다!"
    else:
        final_assets = random.randint(1, 5)   # 1억~5억
        final_happiness = random.randint(30, 60)
        result_desc = "평범하지만 다소 아쉬운 삶을 살았습니다."

    print(f"👴 [노년기] 인생 종료. (직업: {state['career']})")
    return {
        "assets": final_assets,
        "happiness": final_happiness,
        "biography": f"\n[노년의 회고] 자산 {final_assets}억, 행복도 {final_happiness}. {result_desc}"
    }

# --- [3] 그래프(Workflow) 연결 ---

workflow = StateGraph(LifeState)

workflow.add_node("childhood", childhood)
workflow.add_node("youth_choice", youth_choice)
workflow.add_node("old_age", old_age)

workflow.add_edge(START, "childhood")
workflow.add_edge("childhood", "youth_choice")
workflow.add_edge("youth_choice", "old_age")
workflow.add_edge("old_age", END)

# ★ Checkpointer 필수! (메모리 저장소)
checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)


# --- [4] 실행: 1회차 인생 (후회의 시작) ---

thread_config = {"configurable": {"thread_id": uuid.uuid4()}}
print(f"=== 🎬 1회차 인생 시작 (Thread ID: {thread_config['configurable']['thread_id']}) ===")

# 초기값
initial_input = {"name": "김개발", "biography": ""}
result_1 = app.invoke(initial_input, thread_config)

print(f"\n[1회차 결과] {result_1['biography']}")
print(f"👉 자산: {result_1['assets']}억 / 행복: {result_1['happiness']}")


# --- [5] Time Travel: 역사 개변! ---

print("\n\n🚧 ...잠시 후, 당신은 인생의 선택을 후회하며 타임머신을 탑니다... 🚧")

# 1. 히스토리 조회
all_states = list(app.get_state_history(thread_config))
# all_states[0]: 노년기(END)
# all_states[1]: 청년기(youth_choice 완료 시점) -> 여기서 직업이 '공무원'으로 결정됨.
# 우리는 이 시점(청년기 완료)의 결과를 '공무원'이 아니라 다른 걸로 바꿔치기 할 겁니다.

target_state = all_states[1]
my_talent = target_state.values['talent']

print(f"\n⏳ 타임머신 도착! 과거의 나: '{my_talent}' 재능을 가지고 막 직업을 고르려 함.")

# 2. 재능에 맞는 직업 찾기
new_career = "백수"
if my_talent == "코딩": new_career = "AI 스타트업 창업"
elif my_talent == "트로트 노래": new_career = "미스터트롯 가수"
elif my_talent == "주식 투자": new_career = "월가 펀드매니저"
elif my_talent == "라면 끓이기": new_career = "5성급 호텔 요리사"
elif my_talent == "멍 때리기": new_career = "힐링 유튜버"

print(f"⚡ [역사 개변] '9급 공무원' 선택을 취소하고 -> '{new_career}'(으)로 변경합니다!")

# 3. 상태 업데이트 (Update State)
# as_node="youth_choice" : 마치 youth_choice 노드가 방금 'new_career'를 출력한 것처럼 조작함
new_config = app.update_state(
    target_state.config,
    {"career": new_career},
    as_node="youth_choice"
)

# 4. 2회차 실행 (변경된 미래 확인)
print("\n=== 🎬 2회차 인생 시작 (변경된 미래) ===")
# new_config를 넣어서, 갈라진 평행우주에서 시작
result_2 = app.invoke(None, new_config)

print(f"\n[2회차 결과] {result_2['biography']}")
print(f"👉 자산: {result_2['assets']}억 / 행복: {result_2['happiness']}")
print("\n🎉 해피 엔딩 (아마도?) 🎉")