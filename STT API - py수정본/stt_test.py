<<<<<<< HEAD
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

def run_dual_track(
    audio_file_path, 
    openai_key="4E9iixxpREIQezboFfIrh3SfWruRfs5nZAN6ijflKWlD0oWefzy3JQQJ99CEACI8hq2XJ3w3AAAAACOGDSDa", 
    openai_endpoint="https://9ai03-mpouyzd4-switzerlandnorth.services.ai.azure.com/", 
    speech_key="7wPwCa2kS8FBb1bQZFEZTzQweikVVUifRzAStaIKRtkal2f4sEATJQQJ99CEACYeBjFXJ3w3AAAYACOG6gYf", 
    speech_region="eastus"
):
    """
    오디오 파일을 분석하여 텍스트와 발음 점수를 반환하는 함수
    (API 키가 기본값으로 내장되어 있어 파일 경로만 넘겨주면 바로 작동합니다.)
    """
    
    # --------------------------------------------------
    # 트랙 1: Whisper (정확한 텍스트 및 혼용 언어 추출)
    # --------------------------------------------------
    openai_client = AzureOpenAI(
        api_key=openai_key,  
        api_version="2024-02-01", 
        azure_endpoint=openai_endpoint 
    )
    
    try:
        with open(audio_file_path, "rb") as audio_file:
            whisper_result = openai_client.audio.transcriptions.create(
                file=audio_file,
                model="drinkingmool-whisper" 
            )
        final_text = whisper_result.text
    except Exception as e:
        final_text = f"Whisper 에러 발생: {e}"

    # --------------------------------------------------
    # 트랙 2: Azure Speech (대본 없는 발음 평가 전용)
    # --------------------------------------------------
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    audio_config = speechsdk.AudioConfig(filename=audio_file_path) 
    
    pron_config = speechsdk.PronunciationAssessmentConfig(
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Word
    )
    pron_config.reference_text = "" 
    
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, 
        audio_config=audio_config,
        language="en-US" 
    )
    pron_config.apply_to(recognizer)
    
    speech_result = recognizer.recognize_once_async().get()
    
    pron_score = 0
    if speech_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        pronunciation_result = speechsdk.PronunciationAssessmentResult(speech_result)
        pron_score = pronunciation_result.pronunciation_score
        
    # 팀원들의 코드에서 이 변수들을 사용할 수 있도록 뱉어냄
    return final_text, pron_score

# --------------------------------------------------
# 테스트용 블록 (이 파일을 직접 실행할 때만 작동)
# --------------------------------------------------
if __name__ == "__main__":
    print("🎙️ 투트랙 오디오 분석 테스트를 시작합니다...\n")
    
    # 키가 이미 함수에 세팅되어 있으므로 파일 이름만 넣으면 됨
    text_result, score_result = run_dual_track("test2_audio.wav")
    
    print("=== 📊 [백엔드 최종 데이터 결과] ===")
    print(f"✅ 1. 텍스트: {text_result}")
=======
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

def run_dual_track(
    audio_file_path, 
    openai_key="4E9iixxpREIQezboFfIrh3SfWruRfs5nZAN6ijflKWlD0oWefzy3JQQJ99CEACI8hq2XJ3w3AAAAACOGDSDa", 
    openai_endpoint="https://9ai03-mpouyzd4-switzerlandnorth.services.ai.azure.com/", 
    speech_key="7wPwCa2kS8FBb1bQZFEZTzQweikVVUifRzAStaIKRtkal2f4sEATJQQJ99CEACYeBjFXJ3w3AAAYACOG6gYf", 
    speech_region="eastus"
):
    """
    오디오 파일을 분석하여 텍스트와 발음 점수를 반환하는 함수
    (API 키가 기본값으로 내장되어 있어 파일 경로만 넘겨주면 바로 작동합니다.)
    """
    
    # --------------------------------------------------
    # 트랙 1: Whisper (정확한 텍스트 및 혼용 언어 추출)
    # --------------------------------------------------
    openai_client = AzureOpenAI(
        api_key=openai_key,  
        api_version="2024-02-01", 
        azure_endpoint=openai_endpoint 
    )
    
    try:
        with open(audio_file_path, "rb") as audio_file:
            whisper_result = openai_client.audio.transcriptions.create(
                file=audio_file,
                model="drinkingmool-whisper" 
            )
        final_text = whisper_result.text
    except Exception as e:
        final_text = f"Whisper 에러 발생: {e}"

    # --------------------------------------------------
    # 트랙 2: Azure Speech (대본 없는 발음 평가 전용)
    # --------------------------------------------------
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    audio_config = speechsdk.AudioConfig(filename=audio_file_path) 
    
    pron_config = speechsdk.PronunciationAssessmentConfig(
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Word
    )
    pron_config.reference_text = "" 
    
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, 
        audio_config=audio_config,
        language="en-US" 
    )
    pron_config.apply_to(recognizer)
    
    speech_result = recognizer.recognize_once_async().get()
    
    pron_score = 0
    if speech_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        pronunciation_result = speechsdk.PronunciationAssessmentResult(speech_result)
        pron_score = pronunciation_result.pronunciation_score
        
    # 팀원들의 코드에서 이 변수들을 사용할 수 있도록 뱉어냄
    return final_text, pron_score

# --------------------------------------------------
# 테스트용 블록 (이 파일을 직접 실행할 때만 작동)
# --------------------------------------------------
if __name__ == "__main__":
    print("🎙️ 투트랙 오디오 분석 테스트를 시작합니다...\n")
    
    # 키가 이미 함수에 세팅되어 있으므로 파일 이름만 넣으면 됨
    text_result, score_result = run_dual_track("test2_audio.wav")
    
    print("=== 📊 [백엔드 최종 데이터 결과] ===")
    print(f"✅ 1. 텍스트: {text_result}")
>>>>>>> 8bd6a84f346baee31fba30f1b9f38b7fbfe78087
    print(f"✅ 2. 발음 점수: {score_result} / 100")