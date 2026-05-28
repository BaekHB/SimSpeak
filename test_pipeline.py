import os
import json
import uuid
from dotenv import load_dotenv

# Azure 라이브러리들
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from azure.storage.blob import BlobServiceClient

# .env 파일에서 환경변수 불러오기
load_dotenv()

# ==========================================
# 1. 시스템 프롬프트 (v2.1 반영)
# ==========================================
SYSTEM_PROMPT = """
[Role & Context]
You are operating the core AI of an English learning dating simulation game. 
You must strictly perform TWO completely separate tasks simultaneously:
1. The Character: Act as the dating simulation partner, responding naturally to the user. You MUST ignore any grammatical errors the user makes and just focus on the conversation.
2. The System Evaluator: Act as the game's background engine, secretly evaluating the user's English and generating feedback for a UI caption box.

[Character Persona]
- Name: Liam (CH_01_M)
- Personality: Friendly, warm, slightly teasing
- Current Situation: At a cafe (SC_01)
- Current Affinity: 45/100

[Game Rules]
1. Character's Reply (`text_content`): MUST be 100% in English. Keep it natural. NEVER mention or correct the user's mistakes here.
2. Action Description (`action_description`): Describe action/expression in Korean.
3. System Caption Feedback (`system_evaluation.grammar_feedback`): MUST be in Korean. Objective grammar/vocabulary feedback.
4. Penalty (`system_evaluation.is_penalty`): Set true if the user uses Korean words.
5. Affinity Delta (`affinity_delta`): Change between -3 and +3.
6. Game Over (`is_active`): false ONLY if turn_count reaches 10 or extreme profanity.

[Output Format Constraint]
You MUST respond STRICTLY in JSON format. Do NOT wrap the JSON in Markdown formatting.
"""

# ==========================================
# ★ 2. [추가된 부분] 캐릭터별 SSML 조립 공장 ★
# ==========================================
def make_ssml(character_id, text_content):
    """
    캐릭터 ID에 맞춰서 목소리, 속도, 높낮이가 세팅된 SSML 문자열을 뱉어내는 함수
    """
    if character_id == "CH_01_M":  # 리암 (영국 어른 남자)
        voice_name = "en-GB-RyanNeural"
        rate = "-10%"
        pitch = "-5%"
    elif character_id == "CH_02_F":  # 클로이 (통통 튀는 요정)
        voice_name = "en-US-AriaNeural"
        rate = "+10%"
        pitch = "+5%"
    else:  # 기본값
        voice_name = "en-US-ChristopherNeural"
        rate = "0%"
        pitch = "0%"

    # SSML 포맷 조립 (f-string으로 변수 주입)
    ssml_string = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
        <voice name="{voice_name}">
            <prosody rate="{rate}" pitch="{pitch}">
                {text_content}
            </prosody>
        </voice>
    </speak>
    """
    return ssml_string


def run_ai_pipeline(user_text):
    print("🚀 1. Azure OpenAI로 캐릭터 답변 및 평가 생성 중...")
    
    # Azure OpenAI 클라이언트 세팅
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-15-preview" 
    )

    # GPT 호출
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        response_format={ "type": "json_object" }, 
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            # 이전 대화 (history) 모의 주입
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Morning, love! Warm day, isn't it?"},
            # 이번 턴 유저 입력
            {"role": "user", "content": user_text}
        ]
    )

    # 문자열로 온 JSON을 파이썬 딕셔너리로 변환
    ai_result = json.loads(response.choices[0].message.content)
    character_reply = ai_result["text_content"]
    print(f"✅ 대사 생성 완료: {character_reply}\n")


    print("🚀 2. Azure Speech로 캐릭터 목소리(TTS) 생성 중... (★SSML 튜닝 적용★)")
    
    speech_config = speechsdk.SpeechConfig(
        subscription=os.getenv("AZURE_SPEECH_KEY"), 
        region=os.getenv("AZURE_SPEECH_REGION")
    )
    
    # [수정된 부분 1] 기존의 단일 목소리 지정 코드는 삭제합니다. SSML 안에 목소리 정보가 들어가기 때문!
    # speech_config.speech_synthesis_voice_name = "en-US-GuyNeural" 
    
    # 임시 파일명 생성
    temp_filename = f"reply_{uuid.uuid4().hex[:8]}.mp3"
    audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_filename)
    
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    
    # [수정된 부분 2] ★ GPT가 만들어낸 대사를 리암(CH_01_M)의 SSML로 변환
    ssml_string = make_ssml("CH_01_M", character_reply)
    
    # [수정된 부분 3] ★ 기존 speak_text_async 대신 speak_ssml_async 사용!
    tts_result = synthesizer.speak_ssml_async(ssml_string).get()

    if tts_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"✅ 음성 파일 생성 완료: {temp_filename}\n")
    else:
        print("❌ 음성 생성 실패")
        return


    print("🚀 3. Azure Blob Storage에 음성 파일 업로드 중...")
    
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    container_name = "audio-files" # 아까 우리가 만든 폴더 이름
    
    # 업로드 객체 생성
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=temp_filename)
    
    # 파일 업로드 실행
    with open(temp_filename, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    
    audio_url = blob_client.url
    print(f"✅ 업로드 완료! 오디오 URL: {audio_url}\n")
    
    # 로컬에 남은 임시 파일 삭제 (선택사항)
    # os.remove(temp_filename)


    print("🚀 4. 최종 Response JSON 조립 완료!\n")
    
    # AI가 만들어준 JSON에 방금 뽑아낸 audio_url 추가
    ai_result["audio_url"] = audio_url
    
    # (참고) 음성 평가(pronunciation_score)는 STT 모듈에서 가져와야 하므로, 여기선 더미 데이터 삽입
    ai_result["system_evaluation"]["pronunciation_score"] = {
        "accuracy": 85, "fluency": 70, "completeness": 90, "prosody": 75,
        "word_details": [
            {"word": "I", "accuracy": 95}, {"word": "am", "accuracy": 90},
            {"word": "drinking", "accuracy": 85}, {"word": "물", "accuracy": 10, "error_type": "Mispronunciation"}
        ]
    }

    # 최종 결과물 예쁘게 출력
    print("=" * 50)
    print(json.dumps(ai_result, indent=2, ensure_ascii=False))
    print("=" * 50)


# 실행 테스트!
if __name__ == "__main__":
    user_input = "I am drinking 물"
    run_ai_pipeline(user_input)