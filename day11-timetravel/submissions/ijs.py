"""
🍳 AI 레시피 생성기 with Time Travel
- 재료 분석 → 레시피 생성 → 맛 평가
- Time Travel로 과거로 돌아가 다른 요리법으로 재시도 가능
- Checkpoint: 특정 시점 상태의 스냅샷
- Fork: 기존 결과를 덮지 않고 새로운 분기 생성
"""

from dotenv import load_dotenv
load_dotenv()

import uuid

from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI


class State(TypedDict):
    ingredients: NotRequired[str]      # 재료 목록
    recipe: NotRequired[str]           # 생성된 레시피
    taste_evaluation: NotRequired[str] # 맛 평가


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,  # 창의적인 레시피를 위해 temperature 상향
)


def analyze_ingredients(state: State):
    """냉장고 재료를 분석하여 요리 가능한 재료 목록 생성"""
    msg = model.invoke(
        "냉장고에 있을 법한 재료 5가지를 랜덤하게 선택해줘. "
        "재료만 쉼표로 구분해서 간단히 나열해줘. 이모지 없이."
    )
    return {"ingredients": msg.content}

def generate_recipe(state: State):
    """재료를 기반으로 레시피 생성"""
    msg = model.invoke(
        f"다음 재료로 만들 수 있는 간단한 요리 레시피를 작성해줘:\n"
        f"재료: {state['ingredients']}\n\n"
        f"요리명, 조리시간, 간단한 조리법을 포함해줘. 이모지 없이."
    )
    return {"recipe": msg.content}

def evaluate_taste(state: State):
    """안성재 심사위원 스타일로 레시피 평가"""
    msg = model.invoke(
        f"당신은 미쉐린 3스타 셰프 '안성재' 심사위원입니다.\n"
        f"냉철하고 전문적이지만, 가끔 따뜻한 조언도 해주는 스타일입니다.\n"
        f"'흑백요리사' 프로그램의 심사위원처럼 평가해주세요.\n\n"
        f"다음 레시피를 평가해주세요:\n\n"
        f"{state['recipe']}\n\n"
        f"평가 형식:\n"
        f"1. 첫인상 한마디 (안성재 특유의 날카로운 첫마디)\n"
        f"2. 맛 예상 점수 (100점 만점)\n"
        f"3. 플레이팅/비주얼 점수 (100점 만점)\n"
        f"4. 창의성 점수 (100점 만점)\n"
        f"5. 종합 심사평 (안성재 스타일로 전문적이면서도 인간적인 코멘트)\n"
        f"6. 최종 판정: 합격/불합격 (총점 250점 이상이면 합격)\n\n"
        f"안성재 심사위원답게 한국어로 평가해주세요. 이모지 없이."
    )
    return {"taste_evaluation": msg.content}


# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("analyze_ingredients", analyze_ingredients)
workflow.add_node("generate_recipe", generate_recipe)
workflow.add_node("evaluate_taste", evaluate_taste)

# Add edges to connect nodes
workflow.add_edge(START, "analyze_ingredients")
workflow.add_edge("analyze_ingredients", "generate_recipe")
workflow.add_edge("generate_recipe", "evaluate_taste")
workflow.add_edge("evaluate_taste", END)

# Compile
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
graph

config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
    }
}
state = graph.invoke({}, config)

print("\n\n ==========================================")
print(" 🍳 AI 레시피 생성기 - 초기 실행 결과")
print(" ==========================================")
print(f"\n📦 분석된 재료:\n{state['ingredients']}")
print(f"\n📖 생성된 레시피:\n{state['recipe']}")
print(f"\n⭐ 맛 평가:\n{state['taste_evaluation']}")

# The states are returned in reverse chronological order.
states = list(graph.get_state_history(config))

print("\n\n ==========================================")
print(" 📜 체크포인트 히스토리")
print(" ==========================================")
for idx, s in enumerate(states):
    print(f"\n {idx}. 다음 실행 노드: {s.next}")
    print(f"    체크포인트 ID: {s.config['configurable']['checkpoint_id'][:20]}...")

# 재료 분석 후, 레시피 생성 전 시점 선택
selected_state = states[2]
print("\n\n ==========================================")
print(" ⏰ Time Travel - 과거 시점 선택")
print(" ==========================================")
print(f"체크포인트 ID: {selected_state.config['configurable']['checkpoint_id'][:20]}...")
print(f"다음 실행 노드: {selected_state.next}")
print(f"해당 시점의 재료: {selected_state.values.get('ingredients', 'N/A')}")


# 상태 업데이트 (재료를 다르게 변경하여 새로운 요리 시도)
# LLM을 통해 새로운 랜덤 재료 생성
random_ingredients_msg = model.invoke(
    "앞서 선택한 재료와는 완전히 다른 냉장고 재료 6가지를 랜덤하게 선택해줘. "
    "한식, 양식, 중식, 일식, 이탈리아 요리나 프랑스 요리 등 다양한 요리에 쓸 수 있는 재료로. "
    "재료만 쉼표로 구분해서 간단히 나열해줘. 이모지 없이."
)
new_ingredients = random_ingredients_msg.content
new_config = graph.update_state(selected_state.config, values={"ingredients": new_ingredients})
print(f"\n\n ==========================================")
print(" 🔀 재료 변경 후 분기 생성")
print(" ==========================================")
print(f"새로운 재료: {new_ingredients}")


result = graph.invoke(None, new_config)
print("\n\n ==========================================")
print(" 🍳 변경된 재료로 재실행한 결과")
print(" ==========================================")
print(f"\n📦 재료:\n{result['ingredients']}")
print(f"\n📖 새로운 레시피:\n{result['recipe']}")
print(f"\n⭐ 맛 평가:\n{result['taste_evaluation']}")

# 전체 히스토리 확인
all_states = list(graph.get_state_history(config))
print(f"\n\n ==========================================")
print(f" 📊 전체 히스토리: 총 {len(all_states)}개의 체크포인트")
print(" ==========================================")

"""
(원래 실행 - 랜덤 재료)
start → (재료 분석: 랜덤 5개) → (레시피 생성) → (맛 평가) → END

                       └─ (update_state로 분기: 재료 변경)
                          (재료=김치,참치...) → (레시피 생성: 김치참치볶음밥) → (맛 평가) → END

"""