import os
import json
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# 라이브러리 로드
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from azure.storage.blob import BlobServiceClient

# .env 환경변수 로드
load_dotenv()

app = FastAPI(title="AI Azure SSML Voice & Chat API Server")

# ---------------------------------------------------------
# 1. 다중 캐릭터 및 유저별 메모리 DB 구조
# ---------------------------------------------------------
session_db = {}

# ---------------------------------------------------------
# 2. 캐릭터별 영국 런던 악센트 목소리(Voice) 매핑
# ---------------------------------------------------------
VOICE_MAP = {
    "liam": "en-GB-RyanNeural",   # 리암: 영국 남자 (런던 악센트)
    "chloe": "en-GB-SoniaNeural", # 클로이: 영국 여자 (런던 악센트)
}

# ---------------------------------------------------------
# 3. API 요청 데이터 스키마 (Request Body)
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str
    character_id: str  # "liam" 또는 "chloe"
    text: str
    is_video_call: bool

# ---------------------------------------------------------
# 4. 프롬프트 파일 동적 로드 헬퍼 함수
# ---------------------------------------------------------
def get_character_prompt(character_id: str) -> str:
    file_path = f"prompts/{character_id.lower()}.txt"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, 
            detail=f"[{character_id}] 프롬프트 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}"
        )

# ---------------------------------------------------------
# 5. 핵심 대화 및 캐릭터 음성 생성 API 엔드포인트
# ---------------------------------------------------------
@app.post("/chat")
async def chat_with_character(request: ChatRequest):
    char_id = request.character_id.lower()
    print(f"📥 [User: {request.user_id}] -> [{char_id}] 입력: {request.text} (is_video_call: {request.is_video_call})")
    
    # [Step 1] 유저 및 캐릭터별 세션 DB 초기화 및 데이터 로드
    if request.user_id not in session_db:
        session_db[request.user_id] = {}
        
    if char_id not in session_db[request.user_id]:
        session_db[request.user_id][char_id] = {
            "history": [],
            "current_affinity": 30  # 초기 기본 호감도
        }
        
    user_data = session_db[request.user_id][char_id]
    
    # [Step 2] 외부 프롬프트 파일 내용 가져오기
    base_prompt = get_character_prompt(char_id)
    
    # 영상통화 여부에 따른 가이드 문구 분기 지침 주입
    call_status_text = (
        "대면 마주보기 모드 (is_video_call: true) - 눈앞 30cm 거리에서 직접 마주 보며 보이스로 대화하는 상황. 물리적 스킨십 및 밀착 연출 가능."
        if request.is_video_call else 
        "비대면 텍스트 채팅 모드 (is_video_call: false) - 서로 떨어져서 스마트폰 문자를 주고받는 상황. 물리적 스킨십 절대 금지, 독립적인 3인칭 행동 묘사 위주."
    )
    
    # 프롬프트 내부의 JSON 중괄호 에러 방지용 텍스트 결합
    dynamic_context = (
        f"\n\n"
        f"[CURRENT BACKEND LIVE PARAMETERS]\n"
        f"- Current Accumulated Affinity Score: {user_data['current_affinity']}/100\n"
        f"- Current Communication State: {call_status_text}\n"
    )
    
    system_prompt = base_prompt + dynamic_context
    
    # 메세지 조립
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(user_data["history"][-10:])  # 최근 10개 턴만 유지
    messages.append({"role": "user", "content": request.text})

    try:
        # [Step 3] Azure OpenAI 호출 (캐릭터 대사 생성)
        print("🚀 1. Azure OpenAI 캐릭터 답변 및 시스템 평가 데이터 생성 중...")
        ai_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview"
        )
        
        response = ai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            response_format={ "type": "json_object" }, 
            messages=messages
        )
        
        ai_response_text = response.choices[0].message.content
        ai_result = json.loads(ai_response_text)
        
        character_reply = ai_result.get("text_content", "")
        affinity_delta = ai_result.get("affinity_delta", 0)
        is_active = ai_result.get("is_active", True) # 방어 규칙 발동 여부 체크

        # [Step 4] 백엔드 상태값 즉시 누적 및 업데이트
        user_data["history"].append({"role": "user", "content": request.text})
        user_data["history"].append({"role": "assistant", "content": ai_response_text})
        user_data["current_affinity"] += affinity_delta
        user_data["current_affinity"] = max(0, min(100, user_data["current_affinity"]))
        
        # [Step 5] 🔥 Azure Speech 고성능 SSML 감정 주입 시스템
        print(f"🚀 2. Azure Speech 오디오 생성 중... (적용 대사: {character_reply})")
        speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("AZURE_SPEECH_KEY"), 
            region=os.getenv("AZURE_SPEECH_REGION")
        )
        
        target_voice = VOICE_MAP.get(char_id, "en-GB-RyanNeural")
        speech_config.speech_synthesis_voice_name = target_voice 
        
        temp_filename = f"reply_{uuid.uuid4().hex[:8]}.mp3"
        audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_filename)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

        # 🎭 상황별 감정(Style) 분기 처리 
        # 욕설이 감지되어 대화가 차단된 상황(is_active가 false)일 때는 엄청 엄격하고 차가운 톤(sad/serious)으로 변경
        if not is_active:
            voice_style = "sad"          # 차갑고 가라앉은 톤
            style_degree = "1.5"         # 감정의 세기 (강하게)
            speaking_rate = "0.9"        # 평소보다 조금 나직하고 느리게 화내기
        else:
            voice_style = "cheerful"     # 평소에는 연애 시뮬레이션에 맞는 다정하고 밝은 톤
            style_degree = "1.0"         
            speaking_rate = "1.0"        # 기본 속도

        # 마크다운 특수문자 에러 방지용 xml 치환 처리
        safe_reply = character_reply.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # 🔥 Azure 서버가 이해할 수 있는 감정 마크업 언어(SSML) 조립
        ssml_string = f"""
        <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-GB'>
            <voice name='{target_voice}'>
                <mstts:express-as style='{voice_style}' styledegree='{style_degree}'>
                    <prosody rate='{speaking_rate}'>
                        {safe_reply}
                    </prosody>
                </mstts:express-as>
            </voice>
        </speak>
        """

        # 일반 텍스트 대신 완성된 SSML 주입하여 오디오 합성
        tts_result = synthesizer.speak_ssml_async(ssml_string).get()

        if tts_result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise HTTPException(status_code=500, detail="Azure SSML 음성 합성 작업에 실패했습니다.")

        # [Step 6] Azure Blob Storage 스토리지 컨테이너 업로드
        print(f"🚀 3. Azure Blob Storage 오디오 파일 업로드 시작: {temp_filename}")
        blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
        container_name = "audio-files"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=temp_filename)
        
        with open(temp_filename, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
            
        audio_url = blob_client.url
        print(f"✅ 오디오 업로드 완료! 반환 URL: {audio_url}")
        
        # [Error 13 방지] 안전하게 임시 파일 삭제
        try:
            os.remove(temp_filename)
        except OSError as e:
            print(f"⚠️ 임시 파일 자동 제거 스킵: {e}")

        # [Step 7] 최종 통합 Response JSON 패키징 및 리턴
        ai_result["audio_url"] = audio_url
        ai_result["current_total_affinity"] = user_data["current_affinity"]
        
        print("🚀 4. 프론트엔드용 최종 규격 Response 반환 준비 완료.")
        return ai_result

    except json.JSONDecodeError:
        print("❌ LLM 응답 JSON 파싱 에러 발생")
        raise HTTPException(status_code=500, detail="LLM 응답 JSON 파싱 에러가 발생했습니다.")
    except Exception as e:
        print(f"❌ 파이프라인 처리 중 치명적인 서버 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))