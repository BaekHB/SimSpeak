import os
import json
import logging
import time
import asyncio
import datetime
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# 외부 API 오류 탐지용 예외 클래스들 임포트
from openai import OpenAIError
from azure.core.exceptions import AzureError

# [DB 설정] SQLAlchemy 라이브러리 추가
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# 패키징된 최신 AI 파이프라인 클래스 로드
from pipeline import SimSpeakAIPipeline

# 환경변수 로드 (절대 경로 적용으로 터미널 꼬임 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="SimSpeak Production Pronunciation Core API")

# =================================================================
# ⚙️ [시스템 세팅] API 에러 로깅(Logging) 시스템 구축
# =================================================================
logger = logging.getLogger("api_error_logger")
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
log_formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

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
    logger.error(json.dumps(log_payload, ensure_ascii=False))

# =================================================================
# 🚨 [전역 에러 핸들러] 외부 API 장애 발생 시 자동 감지 및 로깅
# =================================================================

@app.exception_handler(OpenAIError)
async def openai_exception_handler(request: Request, exc: OpenAIError):
    user_id = "unknown"
    try:
        body = await request.json()
        user_id = body.get("user_id", "unknown")
    except Exception:
        pass

    status_code = getattr(exc, "status_code", 502)
    message = str(exc)

    if "insufficient_quota" in message or "rate_limit" in message:
        error_code = "OPENAI_TOKEN_LIMIT_OR_NO_BALANCE"
    else:
        error_code = "OPENAI_SERVICE_ERROR"

    log_external_api_error("OpenAI", user_id, error_code, message, status_code)

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "code": error_code,
            "message": "현재 인공지능 대화 시스템 이용이 일시적으로 제한되었습니다."
        }
    )

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


# =========================================================
# [DB 설정] 환경변수(DATABASE_URL) 로드 및 자동 분기형 DB 엔진 세팅
# =========================================================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./simspeak.db")

print(f"[DB Connect Attempt] Database URL: {DATABASE_URL}")

Base = declarative_base()

try:
    if DATABASE_URL.startswith("sqlite"):
        engine = create_engine(
            DATABASE_URL, 
            connect_args={"check_same_thread": False}
        )
    else:
        engine = create_engine(
            DATABASE_URL, 
            pool_pre_ping=True
        )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as db_err:
    print(f"[DB Error] Failed to create database engine: {db_err}")

# DB에 생성될 chat_logs 테이블 구조 정의
class ChatLogModel(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), index=True, nullable=False)
    character_id = Column(String(50), index=True, nullable=False)
    user_text = Column(Text, nullable=False)
    user_audio_url = Column(Text, nullable=True)         
    ai_text_content = Column(Text, nullable=False)
    ai_audio_url = Column(Text, nullable=True)           
    current_affinity = Column(Integer, default=30)       
    summary_context = Column(Text, nullable=True)        
    stage_id = Column(String(50), nullable=True)            
    
    if DATABASE_URL.startswith("sqlite"):
        from sqlalchemy import JSON
        chat_history_context = Column(JSON, nullable=False)   
        raw_llm_log = Column(JSON, nullable=False)            
    else:
        chat_history_context = Column(JSONB, nullable=False)   
        raw_llm_log = Column(JSONB, nullable=False)            

# [팀원 DB 구조 통합] character_chat_logs 테이블 구조 정의
class CharacterChatLogModel(Base):
    __tablename__ = "character_chat_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), nullable=False)
    character_id = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    if DATABASE_URL.startswith("sqlite"):
        from sqlalchemy import JSON
        response_data = Column(JSON, nullable=False)
    else:
        response_data = Column(JSONB, nullable=False)

# 백엔드가 켜질 때 테이블이 없으면 자동으로 DB에 만들어 주는 안전장치
try:
    Base.metadata.create_all(bind=engine)
    print(f"[DB Success] Verified / created table successfully on {DATABASE_URL.split('://')[0]} database!")
    
    # [DB 자동 마이그레이션] 기존 Neon DB/SQLite 테이블에 stage_id 컬럼이 없을 경우 자동으로 추가
    from sqlalchemy import text
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE chat_logs ADD COLUMN stage_id VARCHAR(50);"))
        print("[DB Success] Automatically added 'stage_id' column to chat_logs table.")
    except Exception as alter_err:
        print(f"[DB Info] Auto ALTER COLUMN 'stage_id' status: {alter_err}")
except Exception as table_err:
    print(f"[DB Error] Table creation/verification failed: {table_err}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# =========================================================


# 2. API 요청 데이터 스키마
class ChatRequest(BaseModel):
    user_id: str
    character_id: str  
    text: str
    is_video_call: bool
    user_audio_url: Optional[str] = None  
    stage_id: Optional[str] = "stage_1"

# AI 파이프라인 인스턴스 전역 생성
pipeline = SimSpeakAIPipeline()


# 4. 깔끔하게 정리된 완성형 엔드포인트
@app.post("/chat")
async def chat_with_character(request: ChatRequest, db: Session = Depends(get_db)):
    char_id = request.character_id.lower()
    user_id = request.user_id
    print(f"[User: {user_id}] -> [{char_id}] Initial Input Text: {request.text}")

    # [DB 연동 스레드 안전화 기동] 람다를 걷어내고 명시적 스레드 워커 실행
    def fetch_last_log():
        return (
            db.query(ChatLogModel)
            .filter(ChatLogModel.user_id == user_id, ChatLogModel.character_id == char_id)
            .order_by(ChatLogModel.id.desc())
            .first()
        )

    last_log = await asyncio.to_thread(fetch_last_log)

    if last_log:
        history = list(last_log.chat_history_context)
        current_affinity = last_log.current_affinity
        current_summary = last_log.summary_context or ""
        print(f"[Memory Load] Restored past memory from DB. (Affinity: {current_affinity}/100, Summary exists: {bool(current_summary)})")
    else:
        history = []
        current_affinity = 30  
        current_summary = ""
        print("[New Chat] First conversation entry created.")

    # pipeline.run() 고도화 사양 딕셔너리 수납 인스턴스 어댑터 생성
    temp_session_db = {
        user_id: {
            char_id: {
                "history": history,
                "current_affinity": current_affinity,
                "summary_context": current_summary
            }
        }
    }

    try:
        # 최종 보정된 pipeline.run 인터페이스 규격에 맞춰 동기화 호출
        ai_result = await pipeline.run(
            session_db=temp_session_db,
            user_id=user_id,
            character_id=char_id,
            user_text=request.text,
            is_video_call=request.is_video_call,
            user_audio_url=request.user_audio_url,
            stage_id=request.stage_id
        )

        updated_data = temp_session_db[user_id][char_id]
        updated_history = updated_data["history"]
        updated_affinity = updated_data["current_affinity"]
        updated_summary = updated_data.get("summary_context", "")

        # 파이프라인에서 분리 통합된 원본 생로그 조립 추출
        raw_usage_log = ai_result.pop("raw_llm_log", {})

        def perform_db_write():
            try:
                new_log = ChatLogModel(
                    user_id=user_id,
                    character_id=char_id,
                    user_text=ai_result.get("user_recognized_text", request.text),   
                    user_audio_url=request.user_audio_url,
                    ai_text_content=ai_result.get("text_content", ""),
                    ai_audio_url=ai_result.get("audio_url", ""),                    
                    current_affinity=updated_affinity,       
                    chat_history_context=updated_history,                            
                    raw_llm_log=raw_usage_log,                                       
                    summary_context=updated_summary,                                 
                    stage_id=request.stage_id
                )
                db.add(new_log)

                # [팀원 DB 구조 통합] character_chat_logs 테이블 적재 데이터 일치 보정
                new_monitoring_log = CharacterChatLogModel(
                    user_id=user_id,
                    character_id=char_id,
                    response_data=ai_result
                )
                db.add(new_monitoring_log)

                db.commit() 
            except Exception as write_err:
                db.rollback()
                raise write_err

        await asyncio.to_thread(perform_db_write)
        print("[DB Success] Dialog and raw monitoring logs saved successfully (both chat_logs and character_chat_logs).")
        
        return ai_result

    except Exception as e:
        print(f"[Error] Pipeline processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))