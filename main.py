import os
import json
import asyncio
import logging
import time
from fastapi import FastAPI, HTTPException,Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAIError
from azure.core.exceptions import AzureError
import psycopg2

# 1. 시스템 환경변수 바인딩 명시 로드
load_dotenv()

from pipeline import SimSpeakAIPipeline

app = FastAPI(
    title="SimSpeak AI Character Dual-Track Core Engine",
    version="1.4.0 (Filtered Edition)"
)
# =================================================================
# ⚙️ [시스템 세팅] API 에러 로깅(Logging) 시스템 구축
# =================================================================
logger = logging.getLogger("api_error_logger")
logger.setLevel(logging.INFO)

# 터미널 창에 깔끔하게 출력하기 위한 포맷 설정
stream_handler = logging.StreamHandler()
log_formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# 로그 데이터를 JSON 형태로 규격화하여 출력하는 헬퍼 함수
def log_external_api_error(provider: str, user_id: str, error_code: str, message: str, status_code: int):
    log_payload = {
        "event": "external_api_failure",
        "timestamp": time.time(),
        "provider": provider,
        "user_id": user_id,
        "error_code": error_code,
        "error_message": message,
        "http_status": status_code
    }
    # 모니터링 시스템 수집을 위해 JSON 문자열로 변환하여 에러 출력
    logger.error(json.dumps(log_payload, ensure_ascii=False))

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

# =================================================================
# 🚨 [전역 에러 핸들러] 외부 API 장애 발생 시 자동 감지 및 로깅
# =================================================================

# 1. OpenAI (LLM, TTS) 에러 감지기
@app.exception_handler(OpenAIError)
async def openai_exception_handler(request: Request, exc: OpenAIError):
    # 유저 ID 추출 시도 (실패 시 unknown 처리)
    user_id = "unknown"
    try:
        body = await request.json()
        user_id = body.get("user_id", "unknown")
    except Exception:
        pass

    status_code = getattr(exc, "status_code", 502)
    message = str(exc)

    # 토큰 한도 초과 및 잔액 부족 판별
    if "insufficient_quota" in message or "rate_limit" in message:
        error_code = "OPENAI_TOKEN_LIMIT_OR_NO_BALANCE"
    else:
        error_code = "OPENAI_SERVICE_ERROR"

    # 시스템에 에러 로그 기록
    log_external_api_error("OpenAI", user_id, error_code, message, status_code)

    # 프론트엔드(유저)에게는 안전하고 규격화된 안내 메시지 반환
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "code": error_code,
            "message": "현재 인공지능 대화 시스템 이용이 일시적으로 제한되었습니다."
        }
    )

# 2. Azure (Speech STT, 발음 평가) 에러 감지기
@app.exception_handler(AzureError)
async def azure_exception_handler(request: Request, exc: AzureError):
    user_id = "unknown"
    try:
        body = await request.json()
        user_id = body.get("user_id", "unknown")
    except Exception:
        pass

    status_code = getattr(exc, "status_code", 502)
    message = str(exc)
    
    # Azure 잔액 부족이나 권한(인증) 실패 스캔
    if "401" in message or "Access Denied" in message:
        error_code = "AZURE_AUTH_OR_BALANCE_FAIL"
    else:
        error_code = "AZURE_SERVICE_ERROR"

    log_external_api_error("Azure", user_id, error_code, message, status_code)

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "code": error_code,
            "message": "음성 인식 시스템에 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
        }
    )

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
