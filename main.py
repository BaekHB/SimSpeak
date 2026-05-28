import os
import json
import uuid
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# 라이브러리 로드
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient, ContentSettings

# .env 환경변수 로드
load_dotenv()

app = FastAPI(title="AI Azure Hybrid TTS & Chat API Server")

# ---------------------------------------------------------
# 1. 다중 캐릭터 및 유저별 메모리 DB 구조
# ---------------------------------------------------------
session_db = {}

# ---------------------------------------------------------
# 2. 캐릭터별 목소리(Voice) 매핑 (OpenAI + Azure Native 복구 완료)
# ---------------------------------------------------------
VOICE_MAP = {
    "june": "fable",      # 호주 츤데레 남 (OpenAI)
    "yoon": "alloy",      # 호주 쿨뷰티 여 (OpenAI)
    "ian": "echo",        # 미국 능글남 (OpenAI)
    "chloe": "nova",      # 미국 왈가닥 여 (OpenAI)
    "liam": "en-GB-RyanNeural",    # 영국 으른남 (Azure Native)
    "sienna": "en-GB-SoniaNeural"  # 영국 다정녀 (Azure Native)
}

# ---------------------------------------------------------
# 3. API 요청 데이터 스키마 (Request Body)
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str
    character_id: str  
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
    
    # [Step 1] 유저 및 캐릭터별 세션 DB 초기화
    if request.user_id not in session_db:
        session_db[request.user_id] = {}
        
    if char_id not in session_db[request.user_id]:
        session_db[request.user_id][char_id] = {
            "history": [],
            "current_affinity": 30  
        }
        
    user_data = session_db[request.user_id][char_id]
    
    # [Step 2] 프롬프트 조립
    base_prompt = get_character_prompt(char_id)
    call_status_text = (
        "대면 마주보기 모드 (is_video_call: true) - 눈앞 30cm 거리에서 직접 마주 보며 보이스로 대화하는 상황. 물리적 스킨십 및 밀착 연출 가능."
        if request.is_video_call else 
        "비대면 텍스트 채팅 모드 (is_video_call: false) - 서로 떨어져서 스마트폰 문자를 주고받는 상황. 물리적 스킨십 절대 금지, 독립적인 3인칭 행동 묘사 위주."
    )
    
    dynamic_context = (
        f"\n\n"
        f"[CURRENT BACKEND LIVE PARAMETERS]\n"
        f"- Current Accumulated Affinity Score: {user_data['current_affinity']}/100\n"
        f"- Current Communication State: {call_status_text}\n"
        f"\n[SYSTEM OVERRIDE MESSAGE: 방금 유저의 입력에 한국어가 감지되었다면, 공감이나 위로 여부와 무관하게 무조건 affinity_delta를 -3으로 고정하십시오. 예외는 없습니다.]\n"
    )
    
    system_prompt = base_prompt + dynamic_context
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(user_data["history"][-10:]) 
    messages.append({"role": "user", "content": request.text})

    try:
        # [Step 3] Azure OpenAI 호출 (캐릭터 대사 생성)
        print("🚀 1. Azure OpenAI 캐릭터 답변 생성 중...")
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
        is_active = ai_result.get("is_active", True) 

        # 상태 누적 업데이트
        user_data["history"].append({"role": "user", "content": request.text})
        user_data["history"].append({"role": "assistant", "content": ai_response_text})
        user_data["current_affinity"] += affinity_delta
        user_data["current_affinity"] = max(0, min(100, user_data["current_affinity"]))
        
        # [Step 4] 🔥 하이브리드 TTS 오디오 생성 (메모리 다이렉트 처리)
        print(f"🚀 2. 하이브리드 TTS 오디오 생성 중... (적용 대사: {character_reply})")
        target_voice = VOICE_MAP.get(char_id, "echo")
        temp_filename = f"reply_{uuid.uuid4().hex[:8]}.mp3"
        audio_bytes = b""

        # 1. 영국 캐릭터(시엔나, 리암) - Azure Native HTTP API 사용
        if char_id in ["sienna", "liam"]:
            speech_key = os.getenv("AZURE_SPEECH_KEY")
            region = os.getenv("AZURE_SPEECH_REGION", "eastus")
            url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
            headers = {
                "Ocp-Apim-Subscription-Key": speech_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
                "User-Agent": "FastAPI"
            }
            
            # is_active 에 따른 감정 및 속도 분기 (SSML 적용)
            voice_style = "sad" if not is_active else "cheerful"
            speaking_rate = "0.9" if not is_active else "0.85"

            # 특수문자 에러(400) 방지 처리
            safe_reply = character_reply.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") 

            # 엄격한 SSML 문법 조립 (쌍따옴표 및 mstts 네임스페이스 포함)
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-GB">
                <voice name="{target_voice}">
                    <mstts:express-as style="{voice_style}" styledegree="1.0">
                        <prosody rate="{speaking_rate}">{safe_reply}</prosody>
                    </mstts:express-as>
                </voice>
            </speak>
            """
            
            tts_response = requests.post(url, headers=headers, data=ssml.encode('utf-8'))
            
            if tts_response.status_code == 200:
                audio_bytes = tts_response.content
            else:
                print(f"🚨 Azure 에러 코드: {tts_response.status_code}") 
                print(f"🚨 Azure 에러 상세: {tts_response.text}")
                raise Exception("Azure 영국 억양 생성 실패")

        # 2. 호주/미국 캐릭터(준, 윤, 이안, 클로이) - 기존 OpenAI TTS 사용
        else:
            tts_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_TTS_ENDPOINT"),
                api_key=os.getenv("AZURE_TTS_API_KEY"),
                api_version="2024-02-15-preview"
            )
            
            openai_tts = tts_client.audio.speech.create(
                model=os.getenv("AZURE_TTS_DEPLOYMENT_NAME"),
                voice=target_voice,
                input=character_reply,
                speed=0.85
            )
            audio_bytes = openai_tts.content 

        # [Step 5] 🚀 Azure Blob Storage 다이렉트 업로드 (로컬 파일 생성 없이 바로 쏨)
        print(f"☁️ 3. Azure Blob 다이렉트 업로드 (크기: {len(audio_bytes)} bytes)")
        blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
        container_name = "audio-files"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=temp_filename)
        
        # 브라우저 바로 재생을 위한 ContentSettings 적용
        blob_client.upload_blob(
            audio_bytes, 
            overwrite=True,
            content_settings=ContentSettings(
                content_type="audio/mpeg",
                content_disposition="inline"
            )
        )
            
        audio_url = blob_client.url
        print(f"✅ 오디오 업로드 완료! 반환 URL: {audio_url}")
        
        # [Step 6] 최종 통합 Response JSON 패키징
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