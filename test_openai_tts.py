import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# ★ 1. 방금 TTS를 찾은 그 리소스의 엔드포인트와 키를 넣습니다!
client = AzureOpenAI(
    azure_endpoint="https://방금만든리소스이름.openai.azure.com/",
    api_key="방금만든리소스의_API_KEY",
    api_version="2024-02-15-preview" 
)

test_text = "Morning, love! Your usual Americano? Coming right up. Don't you worry, we'll get you sorted in no time."

print("🎙️ Azure OpenAI TTS 음성 생성 중...")

# ★ 2. 아까 배포(Deploy)할 때 지어준 이름을 여기에 적으세요! (예: my-tts-hd)
deployment_name = "배포한_이름_입력"

# 3. 오디오 생성! (리암의 찰떡 보이스 = onyx)
response = client.audio.speech.create(
    model=deployment_name, 
    voice="onyx",   # 클로이를 원하시면 "nova" 로 변경!
    input=test_text
)

# 4. mp3 파일로 저장
response.stream_to_file("liam_real_voice.mp3")

print("✅ 파일 생성 완료! 당장 liam_real_voice.mp3 파일을 열어서 들어보세요!")