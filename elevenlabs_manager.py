# elevenlabs_manager.py
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

# .env 파일에 ELEVENLABS_API_KEY 를 추가해야 합니다.
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# ==========================================
# ★ 캐릭터별 ElevenLabs Voice ID 매핑 ★
# (ElevenLabs 대시보드에서 각 캐릭터에 어울리는 Voice ID를 복사해서 넣으세요)
# ==========================================
VOICE_ID_MAP = {
    "chloe": "SDGjghyGZzfnsFUFhSJ0",    
    "liam": "qxc1xfoEt2SROQkllTQK",    
}

# 기본 Voice ID (매핑 실패 시 사용할 디폴트 목소리)
DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL" # Rachel (예시)

async def generate_elevenlabs_audio(character_id: str, text: str, output_filename: str) -> bool:
    """
    ElevenLabs API를 비동기로 호출하여 mp3 파일을 생성합니다.
    """
    if not ELEVENLABS_API_KEY:
        print("❌ [ElevenLabs Error] API 키가 .env에 설정되지 않았습니다.")
        return False

    char_id_lower = character_id.lower()
    voice_id = VOICE_ID_MAP.get(char_id_lower, DEFAULT_VOICE_ID)
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    # eleven_multilingual_v2 모델을 사용하면 한국어 섞인 영어(Konglish)도 자연스럽게 처리됩니다.
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            
            # 받아온 음성 바이너리 데이터를 파일로 저장
            with open(output_filename, "wb") as f:
                f.write(response.content)
            return True
            
        except httpx.HTTPStatusError as exc:
            print(f"❌ [ElevenLabs API Error]: HTTP {exc.response.status_code} - {exc.response.text}")
            return False
        except Exception as e:
            print(f"❌ [ElevenLabs Critical Error]: {e}")
            return False