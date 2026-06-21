import os
import json
import logging
import asyncio
import datetime
from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from pipeline import SimSpeakAIPipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="SimSpeak Lightning Async Core API")

print("--- [디버그] DB 설정 시작 ---")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./simspeak.db")
print(f"DATABASE_URL: {DATABASE_URL}")
Base = declarative_base()

try:
    print("--- [디버그] 엔진 생성 중 ---")
    if DATABASE_URL.startswith("sqlite"):
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        print("--- [디버그] 엔진 생성 완료 ---")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as db_err:
    print(f"[DB Error] {db_err}")

print("--- [디버그] DB 설정 끝, 앱 시작 ---")

class ChatLogModel(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    character_id = Column(String(50), index=True, nullable=False)
    user_text = Column(Text, nullable=False)
    user_audio_url = Column(Text, nullable=True)         
    ai_text_content = Column(Text, nullable=False)
    ai_audio_url = Column(Text, nullable=True)           
    current_affinity = Column(Integer, default=30)       
    summary_context = Column(Text, nullable=True)        
    stage_id = Column(Integer, nullable=True)        
    # 🚨 [수정 1] DB 모델에 session_id 컬럼 반영 및 기본값 설정
    session_id = Column(String(100), nullable=False, default="default_session")
    
    if DATABASE_URL.startswith("sqlite"):
        from sqlalchemy import JSON
        chat_history_context = Column(JSON, nullable=False)   
        raw_llm_log = Column(JSON, nullable=False)            
    else:
        chat_history_context = Column(JSONB, nullable=False)   
        raw_llm_log = Column(JSONB, nullable=False)            

class CharacterChatLogModel(Base):
    __tablename__ = "character_chat_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    character_id = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))
    if DATABASE_URL.startswith("sqlite"):
        from sqlalchemy import JSON
        response_data = Column(JSON, nullable=False)
    else:
        response_data = Column(JSONB, nullable=False)

class EnglishLevelTestModel(Base):
    __tablename__ = "english_level_tests"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    test_type = Column(String(50), nullable=False, default="PLACEMENT")
    assigned_level = Column(String(10), nullable=False)
    test_score = Column(Integer, nullable=True)
    fluency_score = Column(Integer, nullable=True)
    expression_score = Column(Integer, nullable=True)
    grammar_score = Column(Integer, nullable=True)
    task_completion_score = Column(Integer, nullable=True)
    vocabulary_score = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

try:
    print("DB 테이블 생성 시작...")
    Base.metadata.create_all(bind=engine)
    print("DB 테이블 생성 완료!")
except Exception as e:
    print(f"DB 테이블 생성 실패: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UnifiedChatRequest(BaseModel):
    user_id: int
    character_id: str
    text: str  
    is_video_call: bool
    user_audio_url: Optional[str] = None  
    stage_id: Optional[int] = 1
    # 🚨 [수정 2] 요청 데이터 바디 규격에도 session_id 기본값 설정
    session_id: Optional[str] = "default_session"

class UnifiedLevelTestRequest(BaseModel):
    user_id: int
    character_id: str
    current_question_index: int  # 1 ~ 8
    user_audio_url: Optional[str] = None
    user_text: Optional[str] = None
    accumulated_answers: Optional[list] = []
    is_quit: Optional[bool] = False

pipeline = SimSpeakAIPipeline()

async def background_evaluation_worker(user_id: int, char_id: str, stage_id: int, user_audio_url: str, dialogue_result: dict, session_id: str):
    db = SessionLocal()
    try:
        print(f"▶️ [비동기 병렬 피드백 트랙] 스타트 (오답노트 및 정밀 발음 평가 중...)")
        
        user_recognized_text = dialogue_result.get("user_recognized_text", "")
        feedback_payload = await pipeline.run_only_evaluation_track(
            user_id=user_id,
            character_id=char_id,
            user_text=user_recognized_text,
            stage_id=stage_id,
            user_audio_url=user_audio_url
        )
        
        feedback_json = feedback_payload["system_evaluation"]
        feedback_json["affinity_delta"] = dialogue_result.get("affinity_delta", 0)
        feedback_json["current_total_affinity"] = dialogue_result.get("current_total_affinity", 30)

        print(f"✅ [비동기 병렬 피드백 트랙] 연산 마감 완료!")
        print(f"==================================================================")
        print(json.dumps(feedback_payload, ensure_ascii=False, indent=2))
        print(f"==================================================================")

        # 🚨 [수정 3] INSERT 구문 실행 시 session_id 변수를 명확하게 바인딩하여 NotNullViolation 원천 차단
        new_log = ChatLogModel(
            user_id=user_id, 
            character_id=char_id, 
            user_text=user_recognized_text,
            user_audio_url=user_audio_url if user_audio_url and user_audio_url.strip() else " ", 
            ai_text_content=dialogue_result.get("text_content", ""),
            ai_audio_url=dialogue_result.get("audio_url", ""), 
            current_affinity=dialogue_result.get("current_total_affinity", 30), 
            chat_history_context=dialogue_result.get("history_context", []), 
            raw_llm_log=dialogue_result.get("raw_llm_log", {}),
            summary_context=dialogue_result.get("summary_context", ""), 
            stage_id=stage_id, 
            session_id=session_id  # 매개변수로 넘어온 session_id 바인딩
        )
        
        final_monitoring_data = {
            "text_content": dialogue_result.get("text_content", ""), 
            "action_description": dialogue_result.get("action_description", ""),
            "audio_url": dialogue_result.get("audio_url", ""), 
            "user_recognized_text": user_recognized_text,
            "affinity_delta": dialogue_result.get("affinity_delta", 0), 
            "current_total_affinity": dialogue_result.get("current_total_affinity", 30),
            "system_evaluation": feedback_json
        }
        new_monitoring_log = CharacterChatLogModel(user_id=user_id, character_id=char_id, response_data=final_monitoring_data)
        
        def save_to_db():
            db.add(new_log)
            db.add(new_monitoring_log)
            db.commit()
            
        await asyncio.to_thread(save_to_db)
        print(f"🎉 [Neon DB] 대사방 로그 + 오답노트 정산본 한 통으로 합치기 최종 성공!")

    except Exception as bg_err:
        db.rollback()
        print(f"❌ [백그라운드 피드백 에러 발생]: {bg_err}")
    finally:
        db.close()


@app.post("/api/v1/chat/message")
async def process_chat_simultaneously(request: UnifiedChatRequest, db: Session = Depends(get_db)):
    char_id = request.character_id.lower()
    user_id = request.user_id
    # 들어온 session_id가 없거나 빈 값이면 기본값으로 대체 보정
    req_session_id = request.session_id if request.session_id else "default_session"

    def fetch_last_log():
        return db.query(ChatLogModel).filter(
            ChatLogModel.user_id == user_id, 
            ChatLogModel.character_id == char_id
        ).order_by(ChatLogModel.id.desc()).first()

    last_log = await asyncio.to_thread(fetch_last_log)

    history = list(last_log.chat_history_context) if last_log and last_log.chat_history_context else []
    current_affinity = last_log.current_affinity if last_log else 30
    current_summary = last_log.summary_context or "" if last_log else ""

    temp_session_db = {user_id: {char_id: {"history": history, "current_affinity": current_affinity, "summary_context": current_summary}}}

    try:
        dialogue_result = await pipeline.run_only_dialogue_track(
            session_db=temp_session_db, user_id=user_id, character_id=char_id,
            user_text=request.text, is_video_call=request.is_video_call,
            user_audio_url=request.user_audio_url, stage_id=request.stage_id
        )

        # 비동기 백그라운드 태스크로 연산 및 인서트 작업을 넘길 때 확실하게 session_id를 인자로 주입함
        asyncio.create_task(
            background_evaluation_worker(
                user_id=user_id, char_id=char_id, stage_id=request.stage_id,
                user_audio_url=request.user_audio_url, dialogue_result=dialogue_result,
                session_id=req_session_id
            )
        )

        return {
            "text_content": dialogue_result.get("text_content"),
            "action_description": dialogue_result.get("action_description"),
            "audio_url": dialogue_result.get("audio_url"),
            "user_recognized_text": dialogue_result.get("user_recognized_text"),
            "affinity_delta": dialogue_result.get("affinity_delta"),
            "current_total_affinity": dialogue_result.get("current_total_affinity"),
            "system_notification": dialogue_result.get("system_notification", "")
        }

    except Exception as e:
        print(f"❌ [메인 트랙 치명적 에러]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat/level_test")
async def process_level_test_question(request: UnifiedLevelTestRequest, db: Session = Depends(get_db)):
    try:
        is_finishing = (request.current_question_index == 8) or request.is_quit
        
        result = await pipeline.process_level_test_question(
            user_id=str(request.user_id),
            character_id=request.character_id,
            question_index=request.current_question_index,
            user_audio_url=request.user_audio_url,
            user_text=request.user_text or "",
            is_finishing=is_finishing
        )
        
        if is_finishing:
            accuracy = None
            fluency = None
            if result.get("pronunciation_evaluations"):
                accuracy = result["pronunciation_evaluations"].get("accuracy")
                fluency = result["pronunciation_evaluations"].get("fluency")
                
            if request.accumulated_answers is not None:
                # 마지막 문항이 이미 accumulated_answers에 포함되어 있는지 확인 후 중복 방지
                already_included = any(
                    ans.get("question_index") == request.current_question_index
                    for ans in request.accumulated_answers
                )
                if not already_included:
                    request.accumulated_answers.append({
                        "question_index": request.current_question_index,
                        "text": result.get("user_recognized_text", ""),
                        "accuracy": accuracy,
                        "fluency": fluency
                    })
                
            print(f"▶️ [레벨 테스트 종합 평가] 스타트 ({len(request.accumulated_answers)}개의 문항 분석 중...)")
            final_result = await pipeline.evaluate_holistic_cefr_level(request.accumulated_answers)
            
            new_test = EnglishLevelTestModel(
                user_id=request.user_id,
                test_type="PLACEMENT",
                assigned_level=final_result.get("assigned_level", "A1"),
                test_score=final_result.get("test_score", 0),
                fluency_score=final_result.get("fluency_score", 0),
                expression_score=final_result.get("expression_score", 0),
                grammar_score=final_result.get("grammar_score", 0),
                task_completion_score=final_result.get("task_completion_score", 0),
                vocabulary_score=final_result.get("vocabulary_score", 0)
            )
            db.add(new_test)
            db.commit()
            print(f"✅ [레벨 테스트 종합 평가] 완료 및 DB 저장 성공: {final_result.get('assigned_level')}")
            
            result["is_finished"] = True
            result["final_result"] = final_result
            
        else:
            result["is_finished"] = False
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
