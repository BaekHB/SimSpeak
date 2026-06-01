import os
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# 라이브러리 로드
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk

# 환경변수 로드 (절대 경로 적용으로 터미널 꼬임 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="SimSpeak Production Pronunciation Core API")

# 1. 메모리 세션 DB
session_db = {}

# 2. API 요청 데이터 스키마 (★ 다른 팀원이 넘겨줄 URL 변수 확보)
class ChatRequest(BaseModel):
    user_id: str
    character_id: str  
    text: str
    is_video_call: bool
    user_audio_url: Optional[str] = None  # 💡 다른 팀원이 Blob에 저장 후 넘겨줄 오디오 URL 주소!

# 3. 💡 [최종 코어 엔진] 클라우드 URL 오디오 실시간 발음 채점 함수
def evaluate_pronunciation_from_url(audio_url: str) -> dict:
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    service_region = os.getenv("AZURE_SPEECH_REGION", "eastus")
    
    error_response = {
        "accuracy": 0, "fluency": 0, "completeness": 0, "prosody": 0, "word_details": []
    }

    if not audio_url:
        return error_response

    try:
        # 1. 다른 팀원이 클라우드(Blob)에 올려둔 진짜 음성 파일 바이너리를 다운로드
        response = requests.get(audio_url, timeout=10)
        if response.status_code != 200:
            print(f"⚠️ 클라우드에서 오디오 파일을 다운로드하는 데 실패했습니다. URL 확인 필요: {audio_url}")
            return error_response
            
        # 2. 다운로드한 바이너리 데이터를 Azure Speech SDK가 읽을 수 있는 메모리 스트림 구조로 변환
        audio_buffer = response.content
        push_stream = speechsdk.audio.PullAudioInputStreamCallback() 
        # (참고: Azure SDK 표준을 맞추기 위해 파일 스트림 구조 핸들링 처리)
        
        # 파일 점유 문제를 원천 차단하기 위해 임시 파일로 변환하여 안전하게 채점 진행
        temp_eval_path = f"temp_eval_{os.getpid()}.wav"
        with open(temp_eval_path, "wb") as f:
            f.write(audio_buffer)

        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        audio_config = speechsdk.AudioConfig(filename=temp_eval_path)
        
        # 자유 말하기(Aviation/Conversation) 채점 모드 설정
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
        
        # 가비지 컬렉터 메모리 해제 및 임시 파일 즉시 파기 (점유 버그 100% 예방)
        del speech_recognizer
        del audio_config
        if os.path.exists(temp_eval_path):
            os.remove(temp_eval_path)
        
        # 3. 채점 성적표 데이터 파싱 및 조립
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
            return {
                "accuracy": int(assessment_result.accuracy_score),
                "fluency": int(assessment_result.fluency_score),
                "completeness": int(assessment_result.completeness_score),
                "prosody": int(assessment_result.prosody_score),
                "word_details": word_details_list
            }
    except Exception as e:
        print(f"⚠️ 코어 채점 엔진 내부 연산 중 오류 발생: {e}")
    return error_response

def get_character_prompt(character_id: str) -> str:
    file_path = f"prompts/{character_id.lower()}.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 4. 🚀 깔끔하게 정리된 완성형 엔드포인트
@app.post("/chat")
async def chat_with_character(request: ChatRequest):
    char_id = request.character_id.lower()
    print(f"📥 [User: {request.user_id}] -> [{char_id}] 입력문장: {request.text}")

    # 세션 초기화
    if request.user_id not in session_db:
        session_db[request.user_id] = {}
    if char_id not in session_db[request.user_id]:
        session_db[request.user_id][char_id] = {"history": [], "current_affinity": 30}
    user_data = session_db[request.user_id][char_id]

    # 💡 [중점 로직] 다른 팀원이 오디오 URL을 실어서 보냈을 때만 실시간 채점 가동!
    real_pronunciation_score = None
    if request.user_audio_url:
        print(f"🎙️ 다른 팀원이 전송한 오디오 URL 감지됨 ➡️ {request.user_audio_url}")
        real_pronunciation_score = evaluate_pronunciation_from_url(request.user_audio_url)
    else:
        print("⌨️ 텍스트 전용 채팅 모드 ➡️ 발음 채점을 진행하지 않습니다.")

    # 프롬프트 조립 및 기록 바인딩
    base_prompt = get_character_prompt(char_id)
    system_prompt = base_prompt + f"\n\n[LIVE] Affinity: {user_data['current_affinity']}/100"
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(user_data["history"][-10:])
    messages.append({"role": "user", "content": request.text})

    try:
        # Azure OpenAI 답변 생성
        ai_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview"
        )
        response = ai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            response_format={"type": "json_object"},
            messages=messages
        )
        ai_response_text = response.choices[0].message.content
        ai_result = json.loads(ai_response_text)
        
        # 친밀도 수치 갱신 및 히스토리 저장
        affinity_delta = ai_result.get("affinity_delta", 0)
        user_data["history"].append({"role": "user", "content": request.text})
        user_data["history"].append({"role": "assistant", "content": ai_response_text})
        user_data["current_affinity"] = max(0, min(100, user_data["current_affinity"] + affinity_delta))

        # 결과 주머니 패키징
        ai_result["audio_url"] = "https://9aifinalteam4.blob.core.windows.net/audio-files/temporary_test_voice.mp3"
        ai_result["current_total_affinity"] = user_data["current_affinity"]
        
        if "system_evaluation" not in ai_result:
            ai_result["system_evaluation"] = {}
            
        # 🔥 실시간으로 연산된 리얼 발음 점수 주입 (텍스트 모드였다면 None)
        ai_result["system_evaluation"]["pronunciation_score"] = real_pronunciation_score
        
        return ai_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
<<<<<<< HEAD


#업로드창
# 파일 최상단 임포트 부분에 아래 항목들이 있는지 확인하고, 없으면 추가해주세요.
from fastapi import UploadFile, File
import shutil
import os
# 팀원이 만든 발음 평가 모듈 임포트
from pronunciation_engine import get_free_speech_pronunciation_score

# =================================================================
# 🎤 [신규 추가] 유저 음성 업로드 및 발음 평가 테스트용 API
# =================================================================
@app.post("/test-voice")
async def test_voice_pronunciation(
    audio_file: UploadFile = File(..., description="테스트할 유저의 음성 파일(.wav, .mp3)을 업로드하세요.")
):
    print(f"📥 오디오 파일 수신 완료: {audio_file.filename}")
    
    # 1. 프론트(Swagger)에서 올라온 파일을 서버 컴퓨터에 임시로 저장
    temp_file_path = f"temp_{audio_file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
    
    try:
        # 2. 팀원이 만든 발음 평가 엔진에 임시 파일 경로를 던져서 점수 받아오기
        print("🚀 Azure Speech 엔진으로 발음 평가 진행 중...")
        score_result = get_free_speech_pronunciation_score(temp_file_path)
        
        # 3. 평가 완료 후 결과 반환
        return {
            "status": "success",
            "message": "발음 평가가 성공적으로 완료되었습니다.",
            "pronunciation_score": score_result
        }
        
    except Exception as e:
        return {"status": "error", "message": f"발음 평가 중 에러 발생: {str(e)}"}
        
    finally:
        # 4. 메모리 관리 및 용량 확보를 위해 임시 오디오 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"🗑️ 임시 오디오 파일 삭제 완료: {temp_file_path}")
=======
>>>>>>> 8bd6a84f346baee31fba30f1b9f38b7fbfe78087
