import os
import json
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# [DB 설정] SQLAlchemy 라이브러리 추가
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# 패키징된 AI 파이프라인 클래스 로드
from pipeline import SimSpeakAIPipeline

# 환경변수 로드 (절대 경로 적용으로 터미널 꼬임 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="SimSpeak Production Pronunciation Core API")


# =========================================================
# [DB 설정] 환경변수(DATABASE_URL) 로드 및 자동 분기형 DB 엔진 세팅
# =========================================================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./simspeak.db")

print(f"[DB Connect Attempt] Database URL: {DATABASE_URL}")

Base = declarative_base()

try:
    # SQLite와 PostgreSQL 환경에 맞춰 최적의 옵션으로 분기 생성
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
    user_audio_url = Column(Text, nullable=True)         # 유저가 업로드한 오디오 URL 기록
    ai_text_content = Column(Text, nullable=False)
    ai_audio_url = Column(Text, nullable=True)           # AI 답변 음성 주소 기록용
    current_affinity = Column(Integer, default=30)       # 영구 저장되는 호감도 스탯
    summary_context = Column(Text, nullable=True)        # [토큰 절약] 오래된 과거 기억 압축 저장소
    
    # 데이터베이스 종류에 맞춰 유연하게 컬럼 타입 세팅 (PostgreSQL일 때는 전용 고성능 JSONB 강제)
    if DATABASE_URL.startswith("sqlite"):
        from sqlalchemy import JSON
        chat_history_context = Column(JSON, nullable=False)   # 대화 히스토리 배열 통째로 저장 (JSON)
        raw_llm_log = Column(JSON, nullable=False)            # 대표님 보고용 토큰 및 원본 생로그 (JSON)
    else:
        chat_history_context = Column(JSONB, nullable=False)   # 대화 히스토리 배열 통째로 저장 (JSONB)
        raw_llm_log = Column(JSONB, nullable=False)            # 대표님 보고용 토큰 및 원본 생로그 (JSONB)

# 백엔드가 켜질 때 테이블이 없으면 자동으로 DB에 만들어 주는 안전장치
try:
    Base.metadata.create_all(bind=engine)
    print(f"[DB Success] Verified / created table successfully on {DATABASE_URL.split('://')[0]} database!")
except Exception as table_err:
    print(f"[DB Error] Table creation/verification failed: {table_err}")

# API 요청이 올 때마다 안전하게 데이터베이스 세션을 열고 닫아주는 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# =========================================================


# 2. API 요청 데이터 스키마 (팀원 코드 유지)
class ChatRequest(BaseModel):
    user_id: str
    character_id: str  
    text: str
    is_video_call: bool
    user_audio_url: Optional[str] = None  # 다른 팀원이 Blob에 저장 후 넘겨줄 오디오 URL 주소

# AI 파이프라인 인스턴스 전역 생성
pipeline = SimSpeakAIPipeline()


# 4. 깔끔하게 정리된 완성형 엔드포인트
@app.post("/chat")
async def chat_with_character(request: ChatRequest, db: Session = Depends(get_db)):
    char_id = request.character_id.lower()
    user_id = request.user_id
    print(f"[User: {user_id}] -> [{char_id}] Initial Input Text: {request.text}")

    # [DB 연동] 로컬 DB에서 해당 유저의 최신 대화 로그 1건 가져와 복구
    last_log = db.query(ChatLogModel)\
        .filter(ChatLogModel.user_id == user_id, ChatLogModel.character_id == char_id)\
        .order_by(ChatLogModel.id.desc())\
        .first()

    if last_log:
        history = list(last_log.chat_history_context)
        current_affinity = last_log.current_affinity
        current_summary = last_log.summary_context or ""
        print(f"[Memory Load] Restored past memory from DB. (Affinity: {current_affinity}/100, Summary exists: {bool(current_summary)})")
    else:
        history = []
        current_affinity = 30  # 최초 대화 시 기본 친밀도
        current_summary = ""
        print("[New Chat] First conversation entry created.")

    # pipeline.run() 인터페이스에 맞게 임시 세션 딕셔너리 구조 생성 (어댑터 패턴)
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
        # 단일 파이프라인 가동 (STT -> LLM -> TTS -> Blob 업로드 일괄 수행)
        ai_result = pipeline.run(
            session_db=temp_session_db,
            user_id=user_id,
            character_id=char_id,
            user_text=request.text,
            is_video_call=request.is_video_call,
            user_audio_url=request.user_audio_url
        )

        # 파이프라인 실행 후 업데이트된 세션 정보 및 친밀도 회수
        updated_data = temp_session_db[user_id][char_id]
        updated_history = updated_data["history"]
        updated_affinity = updated_data["current_affinity"]
        updated_summary = updated_data.get("summary_context", "")

        # 파이프라인에서 조립된 대표님 보고용 GPT 생로그 추출
        raw_usage_log = ai_result.pop("raw_llm_log", {})

        # 대화 기록 및 AI 모니터링 생로그 DB 영구 저장
        new_log = ChatLogModel(
            user_id=user_id,
            character_id=char_id,
            user_text=ai_result.get("user_recognized_text", request.text),   # Whisper가 감지한 텍스트 반영
            user_audio_url=request.user_audio_url,
            ai_text_content=ai_result.get("text_content", ""),
            ai_audio_url=ai_result.get("audio_url", ""),                    # 실제 합성되어 업로드된 TTS 오디오 주소
            current_affinity=updated_affinity,       
            chat_history_context=updated_history,                            # JSON/JSONB 타입으로 대화 배열 통째로 적재
            raw_llm_log=raw_usage_log,                                       # JSON/JSONB 타입으로 토큰 사용량 생로그 저장
            summary_context=updated_summary                                  # 압축 누적된 장기 기억 요약본 저장
        )
        db.add(new_log)
        db.commit() # 데이터베이스 저장 확정!
        print("[DB Success] Dialog and raw monitoring logs saved successfully.")
        
        return ai_result

    except Exception as e:
        db.rollback() # 에러 발생 시 데이터 정합성을 위해 트랜잭션 롤백
        print(f"[Error] Pipeline processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))