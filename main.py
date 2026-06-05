import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import psycopg2

# 1. 시스템 환경변수 바인딩 명시 로드
load_dotenv()

from pipeline import SimSpeakAIPipeline

app = FastAPI(
    title="SimSpeak AI Character Dual-Track Core Engine",
    version="1.4.0 (Filtered Edition)"
)

# 2. 인메모리 테스트 데이터베이스 세션 딕셔너리 수납 공간
session_storage = {}
pipeline_engine = SimSpeakAIPipeline()

# 3. Pydantic 클라이언트 통신 스키마 가드 매핑
class ChatRequest(BaseModel):
    user_id: str
    character_id: str
    text: str
    is_video_call: bool
    user_audio_url: Optional[str] = None

async def save_to_db_monitoring(user_id: str, char_id: str, raw_json_output: dict):
    """
    Neon PostgreSQL 클라우드 원격 DB 서버에 로그를 쌓는 작업을 
    별도 스레드로 분리하여 메인 비동기 루프의 병목 차단.
    """
    def db_worker():
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            return
            
        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            
            query = """
            INSERT INTO character_chat_logs (user_id, character_id, response_data)
            VALUES (%s, %s, %s);
            """
            cursor.execute(query, (user_id, char_id, json.dumps(raw_json_output, ensure_ascii=False)))
            conn.commit()
            print("[DB Success] Dialog and raw monitoring logs saved successfully.")
        except Exception as e:
            print(f"Warning: PostgreSQL monitoring storage logging failed: {e}")
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    # 동기식 DB 커넥션 연산을 논블로킹으로 분리 실행
    await asyncio.to_thread(db_worker)

@app.on_event("startup")
def verify_database_connection():
    """
    서버 초기 기동 시 외부 Neon Cloud DB 핸드셰이킹 검증 및 뼈대 테이블 구축
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Warning: DATABASE_URL variable missing from configuration.")
        return
        
    print(f"[DB Connect Attempt] Database URL: {db_url}")
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS character_chat_logs (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(100) NOT NULL,
            character_id VARCHAR(50) NOT NULL,
            response_data JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        conn.close()
        print("[DB Success] Verified / created table successfully on postgresql database!")
    except Exception as e:
        print(f"Critical: Database connection failed during system verification: {e}")

@app.post("/chat", summary="실시간 음성 및 텍스트 혼용 챗봇 엔진 엔드포인트")
async def handle_character_chat(payload: ChatRequest):
    try:
        print(f"[User: {payload.user_id}] -> [{payload.character_id}] Initial Input Text: {payload.text}")
        
        # 4. 코어 AI 비동기 파이프라인 엔진 가동 (await 핵심 매핑 완료)
        final_response = await pipeline_engine.run(
            session_db=session_storage,
            user_id=payload.user_id,
            character_id=payload.character_id,
            user_text=payload.text,
            is_video_call=payload.is_video_call,
            user_audio_url=payload.user_audio_url
        )
        
        # 5. 백엔드 가드 모니터링 DB에 로그 비동기식 격리 전송
        await save_to_db_monitoring(payload.user_id, payload.character_id, final_response)
        
        return final_response
        
    except Exception as e:
        print(f"Critical Exception inside router path: {e}")
        raise HTTPException(status_code=500, detail=str(e))
