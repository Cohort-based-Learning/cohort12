import uuid
import json
from dotenv import load_dotenv
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# =========================================================
# 상태 정의
# =========================================================
class AegisState(TypedDict, total=False):
    frame_id: str
    frame_meta: str

    # --- VLM 결과 ---
    vlm_status: str      # 정상/의심/이상
    vlm_class: str       # 절도/파손/실신/폭행/투기/none
    vlm_report: str

    # --- LLM 결과 ---
    final_label: str     # 정상/이상

    # --- 시스템 결정 ---
    decision: str
    final_report: str


# =========================================================
# 모델
# =========================================================
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


def safe_json(text: str):
    text = text.strip().replace("```json", "").replace("```", "")
    return json.loads(text)


# =========================================================
# 🔵 ACTION TABLE (LLM 의존 제거 → 안정성 ↑)
# =========================================================
ACTION_MAP = {
    "실신": "119 신고",
    "폭행": "보안팀 긴급 출동",
    "절도": "경찰 신고",
    "파손": "시설 관리자 호출",
    "투기": "경고 방송 및 기록",
    "none": "로그 저장"
}


# =========================================================
# 1️⃣ VLM : perception ONLY
# =========================================================
def vlm_perception(state: AegisState):
    print("🔵 [VLM] 인식 단계")

    prompt = f"""
객체와 행동만 사실 그대로 묘사하고 판단하지 마.

JSON:
{{
 "status": "정상|의심|이상",
 "class": "절도|파손|실신|폭행|투기|none",
 "report": "사실 묘사 한 문장"
}}

장면: {state['frame_meta']}
"""

    data = safe_json(model.invoke(prompt).content)

    return {
        "vlm_status": data["status"],
        "vlm_class": data["class"],
        "vlm_report": data["report"]
    }


# =========================================================
# 2️⃣ LLM : reasoning ONLY
# =========================================================
def llm_validation(state: AegisState):
    print("🟣 [LLM] 판단 단계")

    prompt = f"""
다음 정보를 보고 최종 이상 여부만 판단하라.

status={state['vlm_status']}
class={state['vlm_class']}
report={state['vlm_report']}

JSON:
{{ "final_label": "정상|이상" }}
"""

    data = safe_json(model.invoke(prompt).content)

    return {"final_label": data["final_label"]}


# =========================================================
# 3️⃣ 시스템 : deterministic action + 보고서 생성
# =========================================================
def generate_report(state: AegisState):
    print("🟢 [SYSTEM] 액션/보고서 생성")

    decision = ACTION_MAP.get(state["vlm_class"], "로그 저장")

    report = (
        f"[라벨:{state['final_label']} / 분류:{state['vlm_class']}]\n"
        f"언제: 실시간 감지\n"
        f"어디서: 공장 CCTV\n"
        f"무엇을: {state['vlm_report']}\n"
        f"왜: 이상 행위 가능성 탐지\n"
        f"어떻게: {decision}"
    )

    return {
        "decision": decision,
        "final_report": report
    }


# =========================================================
# Graph 구성
# =========================================================
builder = StateGraph(AegisState)

builder.add_node("vlm", vlm_perception)
builder.add_node("llm", llm_validation)
builder.add_node("report", generate_report)

builder.add_edge(START, "vlm")
builder.add_edge("vlm", "llm")
builder.add_edge("llm", "report")
builder.add_edge("report", END)


# =========================================================
# 🚀 실행 + Time Travel 데모
# =========================================================
if __name__ == "__main__":

    config = {"configurable": {"thread_id": "aegis_demo"}}

    with SqliteSaver.from_conn_string("checkpoints.db") as saver:

        graph = builder.compile(checkpointer=saver)

        # =================================================
        # 1. 최초 실행
        # =================================================
        print("\n========== 1️⃣ 최초 실행 ==========")

        result = graph.invoke({
            "frame_id": str(uuid.uuid4())[:8],
            "frame_meta": "야간 공장, 남성 한 명이 바닥에 쓰러져 움직이지 않음"
        }, config)

        print("\n[초기 결과]")
        print(json.dumps(result, indent=2, ensure_ascii=False))


        # =================================================
        # 2. 체크포인트 조회
        # =================================================
        print("\n========== 2️⃣ 히스토리 조회 ==========")

        states = list(graph.get_state_history(config))

        # 안전한 탐색 (노드 기반)
        target_state = next(s for s in states if s.next == ("llm",))

        print("복원 시점:", target_state.next)


        # =================================================
        # 3. Time Travel (오탐 수정)
        # =================================================
        print("\n========== 3️⃣ 과거 수정 ==========")

        new_config = graph.update_state(
            target_state.config,
            values={
                "vlm_status": "정상",
                "vlm_class": "none",
                "vlm_report": "남성이 휴식을 위해 잠시 바닥에 앉아 있음"
            }
        )


        # =================================================
        # 4. 재실행
        # =================================================
        print("\n========== 4️⃣ Fork 재실행 ==========")

        forked = graph.invoke(None, new_config)

        print("\n[수정 후 결과]")
        print(json.dumps(forked, indent=2, ensure_ascii=False))

