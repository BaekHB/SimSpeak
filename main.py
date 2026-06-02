import os
import json
import requests
import uuid # 파일 덮어쓰기 에러 방지용으로 추가
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# 💡 [DB 설정] SQLAlchemy 라이브러리 추가
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# 라이브러리 로드
from openai import AzureOpenAI
# pyrefly: ignore [missing-import]
import azure.cognitiveservices.speech as speechsdk

# 환경변수 로드 (절대 경로 적용으로 터미널 꼬임 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="SimSpeak Production Pronunciation Core API")


# =========================================================
# 🗄️ 💡 [DB 고정] 로컬 PostgreSQL 연결 설정 파트 (완전 고정)
# =========================================================
DATABASE_URL = "postgresql://postgres:1234@127.0.0.1:5432/postgres"

print(f"📡 [🚨연결 시도] 코드가 지금 바라보는 진짜 DB 주소: {DATABASE_URL}")

Base = declarative_base()

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as db_err:
    print(f"❌ 데이터베이스 엔진 생성 실패: {db_err}")

# DB에 생성될 chat_logs 테이블 구조 정의 (기존 구조 보존 + 요약 컬럼 추가)
class ChatLogModel(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), index=True, nullable=False)
    character_id = Column(String(50), index=True, nullable=False)
    user_text = Column(Text, nullable=False)
    user_audio_url = Column(Text, nullable=True)       # 유저가 업로드한 오디오 URL 기록
    ai_text_content = Column(Text, nullable=False)
    ai_audio_url = Column(Text, nullable=True)         # AI 답변 음성 주소 기록용
    current_affinity = Column(Integer, default=30)     # 영구 저장되는 호감도 스탯
    chat_history_context = Column(JSONB, nullable=False) # 대화 히스토리 배열 통째로 저장 (JSONB)
    raw_llm_log = Column(JSONB, nullable=False)          # 대표님 보고용 토큰 및 원본 생로그 (JSONB)
    summary_context = Column(Text, nullable=True)        # 🧠 [토큰 절약] 오래된 과거 기억 압축 저장소 추가

# 백엔드가 켜질 때 테이블이 없으면 자동으로 로컬 DB에 만들어 주는 안전장치
try:
    Base.metadata.create_all(bind=engine)
    print("💾 [DB 성공] chat_logs 테이블 생성 혹은 연결 검증 완료!")
except Exception as table_err:
    print(f"❌ 테이블 생성 중 에러 발생: {table_err}")

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
    user_audio_url: Optional[str] = None  # 💡 다른 팀원이 Blob에 저장 후 넘겨줄 오디오 URL 주소!

# 3. 💡 [정우님 투트랙 코어 엔진] ⚠️ 절대 변경 금지 (기존 로직 완벽 보존)
def evaluate_dual_track_from_url(audio_url: str) -> tuple[str, dict]:
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    service_region = os.getenv("AZURE_SPEECH_REGION", "eastus")
    
    whisper_text = ""
    error_response = {
        "accuracy": 0, "fluency": 0, "completeness": 0, "prosody": 0, "word_details": []
    }

    if not audio_url:
        return whisper_text, error_response

    try:
        # 다른 팀원이 클라우드(Blob)에 올려둔 진짜 음성 파일 다운로드
        response = requests.get(audio_url, timeout=10)
        if response.status_code != 200:
            print(f"⚠️ 클라우드 오디오 다운로드 실패: {audio_url}")
            return whisper_text, error_response
            
        audio_buffer = response.content
        temp_eval_path = f"temp_eval_{uuid.uuid4().hex[:8]}.wav"
        with open(temp_eval_path, "wb") as f:
            f.write(audio_buffer)

        # --------------------------------------------------
        # ★ 트랙 1: 정우님의 Whisper (정확한 혼용 텍스트 추출)
        # --------------------------------------------------
        try:
            openai_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_WHISPER_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_WHISPER_API_KEY"),
                api_version="2024-02-15-preview"
            )
            with open(temp_eval_path, "rb") as audio_file:
                whisper_result = openai_client.audio.transcriptions.create(
                    file=audio_file,
                    model="drinkingmool-whisper", 
                    prompt="이 오디오는 영어와 한국어가 섞여 있습니다. Hello 안녕하세요.", 
                    language="ko" 
                )
            whisper_text = whisper_result.text
        except Exception as e:
            whisper_text = f"[Whisper 에러: {e}]"

        # --------------------------------------------------
        # ★ 트랙 2: 팀원의 Azure Speech (기존 채점 엔진 그대로 유지)
        # --------------------------------------------------
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        audio_config = speechsdk.AudioConfig(filename=temp_eval_path)
        
        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            reference_text="",
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Word
        )
        pronunciation_config.enable_prosody_assessment()
        
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, language="en-US", audio_config=audio_config
        )
        pronunciation_config.apply_to(speech_recognizer)
        result = speech_recognizer.recognize_once_async().get()
        
        # 가비지 컬렉터 메모리 해제 및 임시 파일 파기
        del speech_recognizer
        del audio_config
        if os.path.exists(temp_eval_path):
            os.remove(temp_eval_path)
        
        detailed_score = error_response
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            assessment_result = speechsdk.PronunciationAssessmentResult(result)
            word_details_list = []
            for word in assessment_result.words:
                error_type = word.error_type if word.error_type != "None" else None
                word_details_list.append({
                    "word": word.word.strip(),
                    "accuracy": int(word.accuracy_score),
                    "error_type": error_type
                })
            detailed_score = {
                "accuracy": int(assessment_result.accuracy_score),
                "fluency": int(assessment_result.fluency_score),
                "completeness": int(assessment_result.completeness_score),
                "prosody": int(assessment_result.prosody_score),
                "word_details": word_details_list
            }
        return whisper_text, detailed_score

    except Exception as e:
        print(f"⚠️ 코어 채점 엔진 내부 연산 중 오류 발생: {e}")
        return whisper_text, error_response

# ⚠️ 절대 변경 금지 (기존 함수 보존)
def get_character_prompt(character_id: str) -> str:
    file_path = f"prompts/{character_id.lower()}.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# 4. 🚀 깔끔하게 정리된 완성형 엔드포인트
@app.post("/chat")
async def chat_with_character(request: ChatRequest, db: Session = Depends(get_db)): 
    char_id = request.character_id.lower()
    print(f"📥 [User: {request.user_id}] -> [{char_id}] 초기 입력: {request.text}")

    # 🔄 💡 [DB 연동] 기존 메모리 세션대신 로컬 DB에서 최신 데이터 1건 가져오기
    last_log = db.query(ChatLogModel)\
        .filter(ChatLogModel.user_id == request.user_id, ChatLogModel.character_id == char_id)\
        .order_by(ChatLogModel.id.desc())\
        .first()

    if last_log:
        history = list(last_log.chat_history_context)
        current_affinity = last_log.current_affinity  
        current_summary = last_log.summary_context or ""  # 🧠 과거 누적 요약본 복구
        print(f"🧠 [장기기억 로드] 로컬 DB에서 과거 기억 복구 완료! (친밀도: {current_affinity}/100)")
    else:
        history = []
        current_affinity = 30  
        current_summary = ""
        print("🆕 [새로운 대화] DB에 첫 기록을 생성합니다.")

    real_pronunciation_score = None
    penalty_message = ""
    
    # 💡 [핵심 연동] 오디오 URL이 들어오면 정우님의 투트랙 + 패널티 가동
    if request.user_audio_url:
        print(f"🎙️ 오디오 URL 감지됨 ➡️ 투트랙 가동: {request.user_audio_url}")
        extracted_text, real_pronunciation_score = evaluate_dual_track_from_url(request.user_audio_url)
        
        # Whisper가 텍스트를 무사히 뽑아왔다면 프론트엔드의 빈 텍스트를 이걸로 덮어씌움
        if extracted_text and not extracted_text.startswith("[Whisper"):
            request.text = extracted_text
            
        # ★ 정우님의 패널티 주입 로직 (종합 발음 점수인 'accuracy'가 50 미만일 때)
        score_val = real_pronunciation_score.get("accuracy", 100)
        if score_val < 50:
            penalty_message = "\n[SYSTEM OVERRIDE MESSAGE: 방금 유저의 발음 점수가 낮거나 한국어가 감지되었습니다. 쌀쌀맞게 대하거나 발음을 지적하고, 무조건 affinity_delta를 -3으로 고정하십시오. 예외는 없습니다.]"
    else:
        print("⌨️ 텍스트 전용 채팅 모드 ➡️ 발음 채점을 진행하지 않습니다.")

    # Azure OpenAI 공통 클라이언트 선언
    ai_client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-15-preview"
    )

    # =========================================================
    # 🧠 [프롬프트 윈도우 & 컨텍스트 주입 최적화 설계 파트]
    # =========================================================
    base_prompt = get_character_prompt(char_id)
    
    # 💡 [요약 주입] 기존 압축 기억이 있다면 시스템 프롬프트 최상단에 주입
    summary_prefix = f"[PAST CONVERSATION SUMMARY]\n{current_summary}\n\n" if current_summary else ""
    
    system_prompt = summary_prefix + base_prompt + f"\n\n[LIVE STATUS]\n- Current Affinity: {current_affinity}/100" + penalty_message
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # 과거 history 배열에서 JSON 구조를 벗겨내고 순수한 대화 대사(content)만 추출하여 전달
    refined_history = []
    for turn in history:
        role = turn.get("role")
        content_raw = turn.get("content", "")
        
        if role == "user":
            refined_history.append({"role": "user", "content": content_raw})
        elif role == "assistant":
            try:
                # DB에 적재되었던 JSON포맷 응답 문자열에서 순수 캐릭터 대사만 파싱
                data = json.loads(content_raw)
                pure_text = data.get("content", content_raw)
                refined_history.append({"role": "assistant", "content": pure_text})
            except Exception:
                # 파싱 실패나 일반 문자열일 경우 안전장치 예외 처리
                refined_history.append({"role": "assistant", "content": content_raw})

    # 💡 [토큰 절약 엔진 가동] 10턴 초과 시 잘려 나가는 앞부분 대화 압축하기
    if len(refined_history) > 10:
        overflow_turns = refined_history[:-10] # 윈도우 밖으로 버려질 대화 조각들
        print(f"🗜️ [토큰 절약] 윈도우를 초과한 {len(overflow_turns)}개의 대화 압축 연산을 수행합니다.")
        
        overflow_text = ""
        for turn in overflow_turns:
            overflow_text += f"{turn['role']}: {turn['content']}\n"
            
        summary_command = [
            {"role": "system", "content": "너는 기억 파수꾼이야. 기존 [누적 요약본]에 새로 잊혀지려는 [대화 조각]의 핵심 사건이나 유저 정보만 결합해서 한 문장의 한국어로 지속 업데이트해 줘. 대화 로그 형식은 금지한다."},
            {"role": "user", "content": f"[기존 누적 요약본]\n{current_summary}\n\n[새 대화 조각]\n{overflow_text}"}
        ]
        
        try:
            summary_response = ai_client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=summary_command,
                max_tokens=150
            )
            current_summary = summary_response.choices[0].message.content.strip()
            print(f"📝 [요약 완료] 압축된 장기 기억: {current_summary}")
        except Exception as e:
            print(f"⚠️ 요약 엔진 일시 오류 발생 (기존 요약 보존): {e}")

    # 문맥이 꼬이지 않도록 최신 10개의 정제된 턴만 슬라이딩 윈도우로 컨텍스트 주입 (원래 코드 동일)
    messages.extend(refined_history[-10:])
    
    # 이번 턴 유저의 최신 입력을 대화창 맨 마지막에 추가
    messages.append({"role": "user", "content": request.text})
    # =========================================================

    try:
        # Azure OpenAI 답변 생성
        response = ai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            response_format={"type": "json_object"},
            messages=messages
        )
        
        raw_usage_log = json.loads(response.model_dump_json()) 
        ai_response_text = response.choices[0].message.content
        ai_result = json.loads(ai_response_text)
        
        # 호감도 보정 계산
        affinity_delta = ai_result.get("affinity_delta", 0)
        updated_affinity = max(0, min(100, current_affinity + affinity_delta)) 

        # 🧠 파이썬 메모리 히스토리 업데이트 (다음 턴을 위해 저장 준비)
        history.append({"role": "user", "content": request.text})
        history.append({"role": "assistant", "content": ai_response_text})

        # 결과 주머니 패키징
        mock_ai_audio_url = "https://9aifinalteam4.blob.core.windows.net/audio-files/reply_8e9e195b.mp3"
        ai_result["audio_url"] = mock_ai_audio_url
        ai_result["current_total_affinity"] = updated_affinity 
        ai_result["user_recognized_text"] = request.text 
        
        if "system_evaluation" not in ai_result:
            ai_result["system_evaluation"] = {}
            
        ai_result["system_evaluation"]["pronunciation_score"] = real_pronunciation_score
        
        # =========================================================
        # 💾 💡 대화 내용 및 모니터링 생로그 DB 영구 저장
        # =========================================================
        new_log = ChatLogModel(
            user_id=request.user_id,
            character_id=char_id,
            user_text=request.text,
            user_audio_url=request.user_audio_url,             
            ai_text_content=ai_result.get("content", ""),
            ai_audio_url=mock_ai_audio_url,
            current_affinity=updated_affinity,                 
            chat_history_context=history,                      
            raw_llm_log=raw_usage_log,
            summary_context=current_summary  # 🧠 압축 누적된 장기 기억 요약본 저장
        )
        db.add(new_log)
        db.commit() 
        print(f"💾 [DB 성공] 로컬 PostgreSQL 금고에 로그 및 요약본 적재 완료! (저장된 친밀도: {updated_affinity}/100)")
        # =========================================================
        
        return ai_result

    except Exception as e:
        db.rollback() 
        print(f"❌ 파이프라인 처리 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))
