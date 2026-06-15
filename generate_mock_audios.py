import os
import asyncio
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

answers = [
    "Hello, my name is J. I am very interesting in computer programming, specially Python and AI. In my free time, I like do work out at the gym and watch football games. I think I am a very passion person who always try to learn new thing.",
    "My ideal type is someone with positive energy. I like people who know what they want and work hard for it. Also, having a similar sense of humor is a big plus for me.",
    "I felt my heart flutter when someone remembered a small detail about me. We were just talking, and they handed me my favorite drink. That thoughtfulness really touched me.",
    "I prefer a quiet café for a first date. It has a relaxed vibe, so we can focus on talking and getting to know each other. Fun activities are better for a second date.",
    "To make a good impression, I would dress neatly to feel confident. During the date, I’d make good eye contact and listen carefully. I’d also ask questions to show I'm genuinely interested.",
    "My perfect date is a nice dinner at a cozy restaurant, followed by a quiet walk. I usually work all day, so I just want to unplug and focus entirely on the other person.",
    "I think mutual respect and shared values make a good match. Even if two people have different hobbies, they can be great together if they respect each other and make time to connect.",
    "I disagree that first impressions decide everything. A good start helps, but people are complex. Someone might seem shy at first, but turn out to be very warm and funny later."
]

import azure.cognitiveservices.speech as speechsdk

async def generate_and_upload():
    # Azure Speech TTS Client
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    # 원어민 여성 음성 하나 선택
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
    
    # Blob Client
    storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(storage_conn)
    container_client = blob_service.get_container_client("audio-files")
    
    urls = []
    
    print("[Start] Generating 8 audio files and uploading to Blob...")
    for i, text in enumerate(answers, 1):
        filename = f"user_mock_answer_q{i}.mp3"
        print(f"Generating Q{i}: {filename}")
        
        try:
            # 1. TTS 생성 (Azure Speech)
            audio_config = speechsdk.audio.AudioOutputConfig(filename=filename)
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            
            result = speech_synthesizer.speak_text_async(text).get()
            
            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                raise Exception(f"TTS Failed: {result.reason}")
            
            # 2. Blob 업로드
            blob_client = container_client.get_blob_client(filename)
            with open(filename, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
                
            urls.append(blob_client.url)
            print(f"[Done] Q{i} URL: {blob_client.url}")
            
            # 로컬 파일 삭제 시도 (에러 무시)
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception:
                pass
                
        except Exception as e:
            print(f"[Error] Q{i}: {e}")
            urls.append("")

    print("\n======================================")
    print("[Success] All Blob URLs:")
    print("======================================")
    print("mock_audio_urls = [")
    for u in urls:
        print(f"    '{u}',")
    print("]")

if __name__ == "__main__":
    asyncio.run(generate_and_upload())
