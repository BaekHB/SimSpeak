import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import azure.cognitiveservices.speech as speechsdk
from azure.storage.blob import BlobServiceClient

# 환경변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI(title="AI Voice API Server")

# 백엔드가 우리에게 보낼 데이터 양식 (요청 스키마)
class VoiceRequest(BaseModel):
    text: str

# POST 요청을 받을 API 엔드포인트 생성
@app.post("/generate-audio")
async def generate_audio(request: VoiceRequest):
    print(f"📥 백엔드에서 텍스트 도착: {request.text}")
    
    try:
        # 1. Azure Speech로 음성 생성
        speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("AZURE_SPEECH_KEY"), 
            region=os.getenv("AZURE_SPEECH_REGION")
        )
        speech_config.speech_synthesis_voice_name = "en-US-GuyNeural"
        
        temp_filename = f"reply_{uuid.uuid4().hex[:8]}.mp3"
        audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_filename)
        
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        tts_result = synthesizer.speak_text_async(request.text).get()

        if tts_result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise HTTPException(status_code=500, detail="음성 생성 실패")

        # 2. Azure Blob Storage에 업로드
        blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
        container_name = "audio-files"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=temp_filename)
        
        with open(temp_filename, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
            
        audio_url = blob_client.url
        print(f"📤 업로드 완료! URL 반환: {audio_url}")
        
        # 다 쓴 임시 파일 삭제
        # os.remove(temp_filename)

        # 3. 백엔드에 예쁘게 결과 리턴
        return {
            "status": "success",
            "audio_url": audio_url
        }

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))