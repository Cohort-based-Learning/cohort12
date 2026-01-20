# 기본 설정
# pip install -U "langgraph-cli[inmem]"
# python >= 3.11

# 파이썬용 LangGraph 프로젝트 최소 뼈대
# langgraph new path/to/your/app --template new-langgraph-project-python

# 의존성 설치
# cd path/to/your/app
# pip install -e .

# app폴더 안에 .env 파일 생성

# ai 로컬 서버 실행
# langgraph dev

## 통신 테스트 실행
from langgraph_sdk import get_client
import asyncio

client = get_client(url="http://localhost:2024")

async def main():
    async for chunk in client.runs.stream(
            None,  # Threadless run
            "agent", # Name of assistant. Defined in langgraph.json.
            input={
                "messages": [{
                    "role": "human",
                    "content": "2+3이 뭐야??",
                }],
            },
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

asyncio.run(main())