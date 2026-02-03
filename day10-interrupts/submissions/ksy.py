import sqlite3
import os
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

# DB 파일 경로 정의
DB_PATH = "travel_fixed.db"

# .env 파일에 설정된 LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2 등을 읽어옵니다.
load_dotenv()

# 1. 여행 상태 정의
class TravelState(TypedDict):
    destination: str
    travelers: int
    hotel: str
    total_price: int
    status: str 

# 2. 예약 툴 정의 (여기를 대폭 수정했습니다!)
def booking_tool(destination: str, travelers: int):
    """
    실제 예약을 진행하는 툴.
    중요: 사용자가 최종 '승인'을 할 때까지 함수가 끝나지 않고 내부에서 계속 돕니다.
    """
    # 초기 제안 값
    current_hotel = f"{destination} 그랜드 하얏트"
    current_price = travelers * 250000
    
    print(f"\n[Tool] {destination} 여행 패키지 생성 중...")

    # --- Tool 내부 루프 시작 (승인/거절 시에만 break) ---
    while True:
        # 툴 실행을 멈추고 사용자에게 확인 요청 (Interrupts in tools)
        user_decision = interrupt({
            "action": "confirm_booking",
            "details": {
                "destination": destination,
                "hotel": current_hotel,
                "travelers": travelers,
                "total_price": current_price
            },
            "message": f"호텔: {current_hotel} / 가격: {current_price}원\n이대로 진행할까요? (approve/edit/reject)"
        })
        
        # resume으로 받은 데이터 분석
        action = user_decision.get("action")
        
        if action == "approve":
            # 루프 종료 및 결과 반환
            return {
                "result": "success", 
                "hotel": current_hotel, 
                "total_price": current_price,
                "msg": "✅ 예약이 확정되었습니다!"
            }
        
        elif action == "edit":
            # 사용자가 수정한 데이터로 변수 업데이트
            print("\n[Tool] 🔄 내용을 수정하고 다시 검토를 요청합니다...")
            if "hotel" in user_decision:
                current_hotel = user_decision["hotel"]
                # 호텔이 바뀌면 가격도 바뀐다고 가정 (+5만원)
                current_price += 50000
            
            # return 하지 않고 while 문 처음으로 돌아가서 다시 interrupt!
            continue
            
        else: # reject
            return {"result": "cancelled", "msg": "❌ 사용자가 취소했습니다."}

# 3. 노드 정의
def validate_travelers_node(state: TravelState):
    num = state["travelers"]
    while True:
        if isinstance(num, int) and num > 0: break
        num = interrupt(f"⚠️ '{num}'명은 불가합니다. 인원(숫자)을 입력하세요.")
    return {"travelers": num}

def process_booking_node(state: TravelState):
    # 툴 실행 (툴 안에서 승인될 때까지 못 빠져나옴)
    res = booking_tool(state["destination"], state["travelers"])
    
    if res["result"] == "success":
        return {"status": "booked", "hotel": res["hotel"], "total_price": res["total_price"]}
    else:
        return {"status": "cancelled"}

# --- 4. 그래프 빌드 및 DB 연결 ---
builder = StateGraph(TravelState)
builder.add_node("validate", validate_travelers_node)
builder.add_node("booking", process_booking_node)

builder.add_edge(START, "validate")
builder.add_edge("validate", "booking")
builder.add_edge("booking", END)

# DB 파일 연결 (영구 저장)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = builder.compile(checkpointer=checkpointer)

# 5. 실행 로직 (재귀를 없애고 while 루프로 변경)
config = {"configurable": {"thread_id": "user_final_fix_1"}}

def run_graph(initial_input=None):
    """
    그래프를 실행하고 인터럽트 발생 시 사용자 입력을 받아 다시 실행하는 루틴입니다.
    GeneratorExit 오류 방지를 위해 generator를 완전히 소모(exhaust)하도록 설계되었습니다.
    """
    current_input = initial_input
    
    while True:
        # stream 실행 (이벤트 순차 수신)
        events = graph.stream(current_input, config, stream_mode="values")
        
        last_event = None
        interrupt_content = None
        
        # GeneratorExit 오류 방지: 루프 중간에 break 하지 않고 끝까지 돌립니다.
        for event in events:
            last_event = event
            if "__interrupt__" in event:
                # 인터럽트 내용만 저장하고 루프는 계속 진행하여 generator를 비웁니다.
                interrupt_content = event["__interrupt__"][0].value
        
        # 루프가 완전히 끝난 후 인터럽트가 있었다면 처리
        if interrupt_content:
            # (1) 인원수 검증 단계 (단순 문자열 메시지 인터럽트)
            if isinstance(interrupt_content, str):
                print(f"\n[AI] {interrupt_content}")
                val = input("답변: ")
                try:
                    # 입력받은 값을 Command(resume=...)에 담아 다음 실행 준비
                    current_input = Command(resume=int(val))
                except ValueError:
                    print("숫자를 입력해주세요.")
                    current_input = Command(resume=0)
            
            # (2) 툴 승인/수정 단계 (딕셔너리 형태의 상세 정보 인터럽트)
            elif isinstance(interrupt_content, dict):
                content = interrupt_content
                print(f"\n──────────────────────────────")
                print(f"[검토 요청] {content['message']}")
                print(f"상세 내용: {content['details']}")
                print(f"──────────────────────────────")
                
                action = input("선택 (approve/edit/reject): ").strip().lower()
                
                if action == "edit":
                    new_hotel = input("새로운 호텔 이름 입력: ")
                    # 수정된 내용과 함께 resume
                    current_input = Command(resume={"action": "edit", "hotel": new_hotel})
                else:
                    # 승인 또는 거절(reject) 상태로 resume
                    current_input = Command(resume={"action": action})
            
            # 인터럽트 처리가 완료되었으므로 while 루프 상단으로 돌아가 다시 stream 시작
            continue
        
        # 더 이상 인터럽트가 없으면(정상 종료) 최종 상태 반환
        return last_event

try:
    # --- 실행부 ---
    print("--- ✈️ AI 여행 예약 ---")

    # 현재 DB 상태 확인
    existing_state = graph.get_state(config)

    # next가 없더라도 values에 데이터가 저장되어 있으면 이어하기 대상으로 간주합니다.
    if existing_state.values:
        print(f"💡 저장된 기록을 불러옵니다. (상태: {existing_state.next or '처리 완료된 세션'})")
        print("💡 이전에 멈춘 지점부터 다시 시작합니다.")
        # 저장된 상태가 있으면 아무 인풋 없이 실행 (체크포인트가 알아서 resume 지점을 찾음)
        final = run_graph(None)
    else:
        print("🆕 저장된 기록이 없습니다. 새로운 세션을 시작합니다.")
        start_input = {"destination": "제주도", "travelers": 0, "status": "searching"}
        final = run_graph(start_input)

    if final:
        print(f"\n--- 최종 결과: {final.get('status')} ---")

finally:
    # 6. 종료 시 정리 작업
    # DB 연결을 명시적으로 닫습니다.
    if 'conn' in locals():
        try:
            conn.commit()  # 강제 종료(^C) 시에도 데이터가 저장되도록 commit 호출
            conn.close()
        except:
            pass
    
    # [선택적 삭제] 프로세스가 완전히 완료(booked 또는 cancelled)된 경우에만 DB를 지웁니다.
    # 이렇게 하면 인터럽트 등으로 중간에 멈췄을 때는 DB가 남아있어 '이어하기'가 가능합니다.
    is_finished = False
    if 'final' in locals() and final:
        status = final.get("status")
        if status in ["booked", "cancelled"]:
            is_finished = True

    if is_finished and os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print(f"\n🗑️ 예약 프로세스가 완료되어 임시 DB 파일({DB_PATH})이 삭제되었습니다.")
        except Exception as e:
            print(f"\n⚠️ DB 파일 삭제 실패: {e}")
    elif os.path.exists(DB_PATH):
        print(f"\n💾 프로세스가 진행 중이므로 DB 파일을 유지합니다. (다음 실행 시 이어하기 가능)")