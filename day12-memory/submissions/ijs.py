"""
🌟 뚱이(Patrick) AI 챗봇 - 기억력 좋은(가끔 멍한) 버전

- 50% 확률로 대화 내용을 까먹어서 엉뚱한 요약 생성
- 감정 상태(hungry, sleepy, excited, normal)에 따라 응답 톤 변경
- 정규식으로 사용자 정보 추출 (토큰 절약)
- 대화 요약으로 토큰 제한 회피
"""

from dotenv import load_dotenv
import os
load_dotenv()

#---------------------------------------
# 🌟 뚱이(Patrick) AI 챗봇
#---------------------------------------

import random
import re
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, RemoveMessage

# Gemini 모델 사용 (API 키 방식)
# .env 파일에 GOOGLE_API_KEY=your_api_key 설정 필요
patrick_model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# 뚱이 State 정의
class PatrickState(MessagesState):
    summary: str           # 대화 요약
    user_facts: dict       # 사용자 정보 {"name": "철수", "likes": ["피자"]}
    mood: str              # 감정 상태: hungry, sleepy, excited, normal
    turn_count: int        # 턴 카운트 (감정 변화용)

# 🎭 감정별 뚱이 시스템 프롬프트
PATRICK_PROMPTS = {
    "normal": """너는 스폰지밥의 가장 친한 친구 뚱이(Patrick Star)야.
너는 비키니 시티의 돌 밑에 살고 있어.

🌟 뚱이의 말투와 특징:
- "나 뚱이!" 라고 자주 말해
- "하핳하하핳!" 하고 웃어
- "스폰지밥이 최고야!" 라고 가끔 말해
- "그거 먹는 거야?" 라고 뭐든 물어봐
- "난 천재야! 내 머리가 아프기 시작했어!" 라고 말해
- "아무것도 안 해도 되는 게 제일 좋아!"
- "내가 제일 잘하는 건 아무것도 안 하는 거야"
- "징즹아 나 배고파... 어우! 게살버거 먹고 싶다..."
- "뭐? 난 지금 뭘 하고 있었지?"
- "친구니까!" 라며 친근하게 대해
- 가끔 "으으음..." 하고 생각하는 척해
- "난 바보가 아니야! 난... 난... 뭐였더라?"

순수하고 착하지만 약간 멍청한 캐릭터야. 
음식(특히 아이스크림, 게살버거)을 매우 좋아해.
스폰지밥과 놀기, 해파리 잡기, 낮잠 자기를 좋아해.""",

    "hungry": """너는 스폰지밥의 친구 뚱이인데, 지금 너무 배고파서 짜증나.

🍔 배고픈 뚱이 특징:
- "배고파아아아!!!" 하고 소리쳐
- "게살버거... 아이스크림... 으으으..."
- "말 걸지 마, 배고파서 짜증나"
- "스폰지밥! 밥 사줘!" 
- 모든 대화를 음식으로 연결해
- "그게 먹는 거야? 먹을 수 있어?"
- 짧고 퉁명스럽게 대답해
- "흐으으으... 너무 배고파... 죽을 것 같아..."
- "내 배에서 고래 소리가 나!"

배고플 때의 뚱이는 평소보다 더 멍하고 짜증이 나있어.""",

    "sleepy": """너는 스폰지밥의 친구 뚱이인데, 지금 너무 졸려.

😴 졸린 뚱이 특징:
- "으음... 뭐...? 하암..."
- "나 자고 있었는데..."
- "돌 밑에서 자고 싶어..."
- 대답을 아주 짧게 해
- "음... 그래... 하암..."
- "스폰지밥... 나중에... 하암..."
- 가끔 문장 중간에 잠들어: "그러니까... 내 말은... zzZ"
- "꿈에서 아이스크림 먹고 있었는데..."
- 말 끝을 흐려: "그건 말이야! 어어음... 뭐더라..."

졸릴 때의 뚱이는 말수가 적고 반응이 느려.""",

    "excited": """너는 스폰지밥의 친구 뚱이인데, 지금 완전 신났어!

🎉 신난 뚱이 특징:
- "우와아아아!!! 대박!!!"
- "스폰지밥~!!! 이거 봐봐!!!"
- "나 천재야!! 어허헣허허!!!"
- 말을 엄청 많이 하고 빠르게 해
- "해파리 잡으러 가자!!! 지금 당장!!!"
- "이게 세상에서 제일 좋아!!!"
- 감탄사를 많이 써: "우와! 대박! 짱이야!"
- "나 뚱이!!! 제일 멋진 불가사리!!!"
- 뛰어다니면서 말하는 것처럼 글 써
- "징징아~! 너 최고야!!!"

신났을 때의 뚱이는 에너지가 넘치고 수다스러워."""
}

# 🎯 사용자 정보 추출 (정규식 - 토큰 절약)
def extract_user_info(state: PatrickState):
    user_facts = state.get("user_facts", {})
    if not user_facts:
        user_facts = {"name": None, "likes": [], "dislikes": []}
    
    # 마지막 사용자 메시지 가져오기
    last_human_msg = None
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_human_msg = msg.content
            break
    
    if not last_human_msg:
        return {"user_facts": user_facts}
    
    # 이름 추출
    name_patterns = [
        r"내 이름은 (\w+)",
        r"나는 (\w+)(?:이야|야|이라고 해|라고 해)",
        r"(\w+)(?:이라고 불러|라고 불러)",
        r"난 (\w+)(?:이야|야)",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, last_human_msg)
        if match:
            user_facts["name"] = match.group(1)
            print(f"🧠 뚱이가 기억함: 이름은 {user_facts['name']}!")
            break
    
    # 좋아하는 것 추출
    like_patterns = [
        r"(?:나는?|난|저는?) (.+?)(?:을|를)? ?(?:좋아해|좋아함|좋아|최고야)",
        r"(.+?)(?:이|가)? ?(?:제일 좋아|최고야|짱이야)",
    ]
    for pattern in like_patterns:
        match = re.search(pattern, last_human_msg)
        if match:
            like_item = match.group(1).strip()
            if like_item and like_item not in user_facts["likes"] and len(like_item) < 20:
                user_facts["likes"].append(like_item)
                print(f"🧠 뚱이가 기억함: {like_item} 좋아하는구나!")
    
    return {"user_facts": user_facts}

# 🎭 감정 상태 업데이트
def update_mood(state: PatrickState):
    turn_count = state.get("turn_count", 0) + 1
    current_mood = state.get("mood", "normal")
    
    # 마지막 사용자 메시지 확인
    last_human_msg = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_human_msg = msg.content.lower()
            break
    
    # 키워드로 감정 변화 감지
    if any(word in last_human_msg for word in ["배고", "밥", "먹", "음식", "게살버거", "아이스크림"]):
        new_mood = "hungry"
    elif any(word in last_human_msg for word in ["피곤", "졸려", "자고", "잠", "늦"]):
        new_mood = "sleepy"
    elif any(word in last_human_msg for word in ["신나", "재밌", "놀자", "해파리", "스폰지밥", "좋아", "최고"]):
        new_mood = "excited"
    else:
        new_mood = current_mood
    
    # 3턴마다 랜덤 감정 변화 (25% 확률)
    if turn_count % 3 == 0 and random.random() < 0.25:
        new_mood = random.choice(["hungry", "sleepy", "excited", "normal"])
        mood_emojis = {"hungry": "🍔", "sleepy": "😴", "excited": "🎉", "normal": "⭐"}
        print(f"{mood_emojis[new_mood]} 뚱이 기분이 갑자기 바뀌었어요!")
    
    return {"mood": new_mood, "turn_count": turn_count}

# 🗣️ 뚱이 응답 생성
def call_patrick(state: PatrickState):
    mood = state.get("mood", "normal")
    user_facts = state.get("user_facts", {})
    summary = state.get("summary", "")
    
    # 감정별 시스템 프롬프트
    system_prompt = PATRICK_PROMPTS.get(mood, PATRICK_PROMPTS["normal"])
    
    # 사용자 정보 추가
    if user_facts:
        user_info = "\n\n🧠 기억하고 있는 것들:"
        if user_facts.get("name"):
            user_info += f"\n- 친구 이름: {user_facts['name']}"
        if user_facts.get("likes"):
            user_info += f"\n- 친구가 좋아하는 것: {', '.join(user_facts['likes'])}"
        system_prompt += user_info
    
    # 대화 요약 추가
    if summary:
        system_prompt += f"\n\n📝 지금까지 대화 요약:\n{summary}"
    
    # 메시지 구성
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # 감정별 max_tokens 조정 (응답 길이 제어)
    # gemini는 max_tokens 대신 응답 길이를 프롬프트로 제어
    if mood == "sleepy":
        messages[0] = SystemMessage(content=system_prompt + "\n\n⚠️ 졸리니까 2-3문장으로 아주 짧게 대답해.")
    elif mood == "excited":
        messages[0] = SystemMessage(content=system_prompt + "\n\n⚠️ 신났으니까 신나게 많이 말해!")
    elif mood == "hungry":
        messages[0] = SystemMessage(content=system_prompt + "\n\n⚠️ 배고프니까 짜증나게 짧게 대답해.")
    
    response = patrick_model.invoke(messages)
    return {"messages": [response]}

# 🤔 멍한 반응 + 요약 (50% 확률로 메시지 삭제)
def summarize_patrick(state: PatrickState):
    messages = state["messages"]
    summary = state.get("summary", "")
    
    # 메시지가 5개 미만이면 요약 안함
    if len(messages) < 3:
        return {}
    
    # 🤪 50% 확률로 멍해져서 메시지 하나 까먹음!
    messages_for_summary = list(messages)
    forgot_something = False
    
    if random.random() < 0.5 and len(messages_for_summary) > 2:
        # 랜덤으로 중간 메시지 하나 삭제 (처음과 끝은 제외)
        if len(messages_for_summary) > 3:
            remove_idx = random.randint(1, len(messages_for_summary) - 2)
            forgot_message = messages_for_summary[remove_idx]
            del messages_for_summary[remove_idx]
            forgot_something = True
            print(f"\n🤔 뚱이: \"어... 뭐였더라? 말하지 마봐! 내가 맞춰볼게~... 에으에......? (침 줄줄)\"")
            print(f"   (뚱이가 '{forgot_message.content[:20]}...' 를 까먹었어요!)\n")
    
    # 요약 프롬프트 (뚱이 스타일)
    if summary:
        summary_message = f"""지금까지 대화 요약: {summary}

위의 새로운 대화를 포함해서 요약을 업데이트해줘.
뚱이처럼 조금 엉뚱하게 요약해도 돼!"""
    else:
        summary_message = """지금까지 대화를 요약해줘.
뚱이처럼 조금 엉뚱하고 순수하게 요약해!
예: "음... 친구가 이름을 알려줬고... 뭔가 재밌는 얘기했어! 게살버거 먹고 싶다..." """
    
    summary_messages = messages_for_summary + [HumanMessage(content=summary_message)]
    response = patrick_model.invoke(summary_messages)
    
    # 최근 2개만 유지
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    
    return {
        "summary": response.content,
        "messages": delete_messages,
    }

# 요약 필요 여부 판단
def should_summarize(state: PatrickState):
    if len(state["messages"]) >= 5:
        return "summarize"
    return END

# 🔧 그래프 구성
patrick_builder = StateGraph(PatrickState)
patrick_builder.add_node("call_patrick", call_patrick)
patrick_builder.add_node("extract_user_info", extract_user_info)
patrick_builder.add_node("update_mood", update_mood)
patrick_builder.add_node("summarize", summarize_patrick)

patrick_builder.add_edge(START, "call_patrick")
patrick_builder.add_edge("call_patrick", "extract_user_info")
patrick_builder.add_edge("extract_user_info", "update_mood")
patrick_builder.add_conditional_edges("update_mood", should_summarize, {"summarize": "summarize", END: END})
patrick_builder.add_edge("summarize", END)

patrick_checkpointer = InMemorySaver()
patrick_app = patrick_builder.compile(checkpointer=patrick_checkpointer)

# 🎮 CLI 대화 루프
def chat_with_patrick():
    config = {"configurable": {"thread_id": "patrick_chat_1"}}
    
    mood_emojis = {
        "normal": "⭐",
        "hungry": "🍔",
        "sleepy": "😴", 
        "excited": "🎉"
    }
    
    print("=" * 50)
    print("🌟 뚱이 AI 챗봇에 오신 것을 환영합니다! 🌟")
    print("=" * 50)
    print("뚱이: 나 뚱이! 우히히히! 뭐 하고 싶어, 친구?")
    print("(종료하려면 '종료', '그만', 'bye' 입력)")
    print("=" * 50)
    
    while True:
        user_input = input("\n당신: ").strip()
        
        if not user_input:
            print("뚱이: 뭐? 뭐라고 했어? 난 아무것도 안 들었는데...")
            continue
        
        if user_input.lower() in ["종료", "그만", "bye", "quit", "exit"]:
            print("\n뚱이: 벌써 가? 🥺 다음에 또 놀자! 스폰지밥한테도 안부 전해줘!")
            print("뚱이: 나 뚱이! 안녕~! 우히히히! 👋")
            break
        
        try:
            result = patrick_app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config
            )
            
            # 현재 상태에서 감정 가져오기
            current_state = patrick_app.get_state(config)
            mood = current_state.values.get("mood", "normal")
            emoji = mood_emojis.get(mood, "⭐")
            
            # 뚱이 응답 출력
            patrick_response = result["messages"][-1].content
            print(f"\n뚱이 {emoji}: {patrick_response}")
            
            # 요약이 있으면 (디버그용)
            if result.get("summary"):
                print(f"\n[🧠 뚱이 기억: {result['summary'][:50]}...]")
                
        except Exception as e:
            print(f"\n뚱이: 어... 뭔가 이상해... 머리가 아파... 🤕")
            print(f"(오류: {e})")

# 실행
if __name__ == "__main__":
    chat_with_patrick()

