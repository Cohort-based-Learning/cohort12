import uuid
from dotenv import load_dotenv
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

# .env 파일로부터 환경 변수 로드
load_dotenv()

# 1. State 정의
class TravelState(TypedDict):
    city: NotRequired[str]
    activity: NotRequired[str]
    restaurant: NotRequired[str]

# 2. 모델 설정
model = init_chat_model(
    "claude-haiku-4-5-20251001",
    temperature=0,
)

# 3. 노드 함수 정의
def select_city(state: TravelState):
    msg = model.invoke("혼자 여행하기 좋은 아시아 도시 하나만 추천해줘. 이름만 말해줘 without emojis")
    return {"city": msg.content.strip()}

def suggest_activity(state: TravelState):
    msg = model.invoke(f"{state['city']}에서 할 수 있는 대표적인 액티비티 하나만 짧게 추천해줘 without emojis")
    return {"activity": msg.content}

def suggest_restaurant(state: TravelState):
    msg = model.invoke(f"{state['city']} {state['activity']} 근처에서 갈만한 식당 하나만 추천해줘 without emojis")
    return {"restaurant": msg.content}

# 4. 그래프 구축
workflow = StateGraph(TravelState)
workflow.add_node("select_city", select_city)
workflow.add_node("suggest_activity", suggest_activity)
workflow.add_node("suggest_restaurant", suggest_restaurant)

workflow.add_edge(START, "select_city")
workflow.add_edge("select_city", "suggest_activity")
workflow.add_edge("suggest_activity", "suggest_restaurant")
workflow.add_edge("suggest_restaurant", END)

# 5. 체크포인트 및 상태 관리 설정
# InMemorySaver: 체크포인트(상태 스냅샷)를 메모리(RAM)에 저장하는 장치입니다.
# 프로그램 종료 시 데이터가 사라지지만, 실행 중에는 과거의 어떤 시점으로든 되돌아갈 수 있게 해줍니다.
checkpointer = InMemorySaver()  # 메모리에 저장하는 '세이브 장치'를 만듭니다.
graph = workflow.compile(checkpointer=checkpointer)  # 그래프가 실행될 때마다 자동으로 저장하도록 연결합니다.

# --- [STEP 1] 초기 실행 ---
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
initial_state = graph.invoke({}, config)

print("\n\n ## 1. 초기 실행 결과 (LLM 추천) ##")
print(f"추천된 도시: {initial_state['city']}")
print(f"활동: {initial_state['activity']}")
print(f"식당: {initial_state['restaurant']}")

# --- [STEP 2] 사용자로부터 새로운 도시 입력 받기 ---
print("\n" + "="*50)
user_city = input("변경하고 싶은 도시 이름을 입력하세요: ")
print("="*50)

# --- [STEP 3] 과거 시점으로 타임 트래블 하여 상태 업데이트 ---
states = list(graph.get_state_history(config))
# states[2] 시점: 도시 결정은 끝났고 활동 추천은 시작되기 전 단계
selected_state = states[2]

# 사용자가 입력한 도시로 상태를 강제 업데이트 (새로운 분기 생성)
new_config = graph.update_state(
    selected_state.config, 
    values={"city": user_city}
)

# --- [STEP 4] 변경된 도시로 이어서 실행 ---
print(f"\n ## 2. '{user_city}'(으)로 일정을 다시 생성합니다... ##")
final_result = graph.invoke(None, new_config)

print(f"\n최종 확정 도시: {final_result['city']}")
print(f"새로운 활동: {final_result['activity']}")
print(f"새로운 식당: {final_result['restaurant']}")

# --- [STEP 5] 히스토리 확인 ---
print("\n\n ## 3. 전체 히스토리 요약 (데이터 분기 확인) ##")
# get_state_history는 기본적으로 '최신 상태 → 과거 상태' 순서(역순)로 결과를 반환합니다.
# 따라서 0번 인덱스가 가장 최근의 상태이며, 마지막 번호가 그래프의 시작 지점입니다.
for i, s in enumerate(graph.get_state_history(config)):
    # 각 시점의 체크포인트 ID를 추출하여 함께 출력합니다.
    ckpt_id = s.config['configurable']['checkpoint_id']
    print(f"[{i}] 다음 노드: {s.next} | ID: {ckpt_id} | 현재 도시 값: {s.values.get('city')}")


'''
[ 메인 워크플로우 흐름 ]

       START
         │
  ┌──────┴──────┐
  │ select_city │ (1. LLM이 도시 결정 / 예: "방콕")
  └──────┬──────┘
         │
  ┌──────┴───────────┐
  │ suggest_activity │ (2. 해당 도시 액티비티 추천)
  └──────┬───────────┘
         │
  ┌──────┴───────────┐
  │suggest_restaurant│ (3. 해당 액티비티 근처 식당 추천)
  └──────┬───────────┘
         │
        END


-----------------------------------------------------------
[ 타임 트래블 및 분기(Fork) 발생 시점 ]

         ● START
         │
         ▼
  [ 1. select_city ] ────────┐ 
         │                   │
         │ (기존 흐름)         │ (update_state: 사용자 입력 "서울")
         │                   │
         ▼                   ▼
  [ 2. suggest_activity ]    [ 2. suggest_activity ]
      (방콕 활동 추천)           (서울 활동 추천)
         │                   │
         ▼                   ▼
  [ 3. suggest_restaurant ]  [ 3. suggest_restaurant ]
      (방콕 식당 추천)           (서울 식당 추천)
         │                   │
         ▼                   ▼
      🏁 END (기존)          🏁 END (새 분기)
'''