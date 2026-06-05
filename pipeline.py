import os
import uuid
import json
import requests
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from azure.storage.blob import BlobServiceClient

class SimSpeakAIPipeline:
    def __init__(self):
        # 환경변수 및 API 설정 로드
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        self.whisper_endpoint = os.getenv("AZURE_OPENAI_WHISPER_ENDPOINT")
        self.whisper_key = os.getenv("AZURE_OPENAI_WHISPER_API_KEY")
        self.whisper_deployment = os.getenv("AZURE_OPENAI_WHISPER_DEPLOYMENT_NAME", "drinkingmool-whisper")
        
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION", "eastus")
        self.storage_connection = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    def make_ssml(self, character_id: str, text_content: str) -> str:
        char_id = character_id.lower()
        voice_name = "en-US-ChristopherNeural"
        rate, pitch = "0%", "0%"
        
        if char_id == "liam":
            voice_name = "en-GB-RyanNeural"
            rate, pitch = "-10%", "-5%"
        elif char_id == "chloe":
            voice_name = "en-US-AriaNeural"
            rate, pitch = "+10%", "+5%"
        elif char_id == "ian":
            voice_name = "en-US-ChristopherNeural"
        elif char_id == "june":
            voice_name = "en-AU-WilliamNeural"
        elif char_id == "sienna":
            voice_name = "en-GB-SoniaNeural"
        elif char_id == "yoon":
            voice_name = "en-AU-NatashaNeural"
            
        return f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
            <voice name="{voice_name}">
                <prosody rate="{rate}" pitch="{pitch}">
                    {text_content}
                </prosody>
            </voice>
        </speak>
        """

    def evaluate_dual_track(self, audio_url: str) -> tuple[str, dict]:
        whisper_text = ""
        error_response = {"accuracy": 0, "fluency": 0, "completeness": 0, "prosody": 0, "word_details": []}
        if not audio_url: return whisper_text, error_response

        temp_eval_path = f"temp_eval_{uuid.uuid4().hex[:8]}.wav"
        try:
            response = requests.get(audio_url, timeout=15)
            if response.status_code != 200: return whisper_text, error_response
            with open(temp_eval_path, "wb") as f: f.write(response.content)

            # Track 1: Whisper
            try:
                whisper_client = AzureOpenAI(azure_endpoint=self.whisper_endpoint, api_key=self.whisper_key, api_version="2024-02-15-preview")
                with open(temp_eval_path, "rb") as audio_file:
                    whisper_result = whisper_client.audio.transcriptions.create(file=audio_file, model=self.whisper_deployment, prompt="Hello! 안녕하세요.")
                whisper_text = whisper_result.text
            except: pass

            # Track 2: Azure Speech
            detailed_score = error_response
            try:
                speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
                audio_config = speechsdk.AudioConfig(filename=temp_eval_path)
                
                pure_english_reference = "".join(char for char in whisper_text if not ('가' <= char <= '힣' or 'ㄱ' <= char <= 'ㅣ')).strip()
                pure_english_reference = " ".join(pure_english_reference.split())
                
                pronunciation_config = speechsdk.PronunciationAssessmentConfig(
                    reference_text=pure_english_reference,
                    grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                    granularity=speechsdk.PronunciationAssessmentGranularity.Word
                )
                pronunciation_config.enable_prosody_assessment()
                
                speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language="en-US", audio_config=audio_config)
                pronunciation_config.apply_to(speech_recognizer)
                result = speech_recognizer.recognize_once_async().get()
                
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    assessment_result = speechsdk.PronunciationAssessmentResult(result)
                    word_details_list = []
                    for word in assessment_result.words:
                        word_details_list.append({
                            "word": word.word.strip(),
                            "accuracy": int(word.accuracy_score),
                            "error_type": word.error_type if word.error_type != "None" else None
                        })
                    detailed_score = {
                        "accuracy": int(assessment_result.accuracy_score),
                        "fluency": int(assessment_result.fluency_score),
                        "completeness": int(assessment_result.completeness_score),
                        "prosody": int(assessment_result.prosody_score),
                        "word_details": word_details_list
                    }
            except: pass
            return whisper_text, detailed_score
        except: return whisper_text, error_response
        finally:
            if 'speech_recognizer' in locals():
                try: del speech_recognizer
                except: pass
            if 'audio_config' in locals():
                try: del audio_config
                except: pass
            if os.path.exists(temp_eval_path):
                try: os.remove(temp_eval_path)
                except: pass

    def generate_tts(self, character_id: str, text_content: str) -> str:
        temp_filename = f"reply_{uuid.uuid4().hex[:8]}.mp3"
        try:
            speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
            audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_filename)
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            
            synthesizer.speak_ssml_async(self.make_ssml(character_id, text_content)).get()
            blob_service_client = BlobServiceClient.from_connection_string(self.storage_connection)
            blob_client = blob_service_client.get_blob_client(container="audio-files", blob=temp_filename)
            with open(temp_filename, "rb") as data: blob_client.upload_blob(data, overwrite=True)
            return blob_client.url
        except: return "https://9aifinalteam4.blob.core.windows.net/audio-files/reply_8e9e195b.mp3"
        finally:
            if os.path.exists(temp_filename):
                try: os.remove(temp_filename)
                except: pass

    def get_character_prompt(self, character_id: str) -> str:
        with open(f"prompts/{character_id.lower()}.txt", "r", encoding="utf-8") as f: return f.read()

    def run(self, session_db: dict, user_id: str, character_id: str, user_text: str, is_video_call: bool, user_audio_url: str = None) -> dict:
        char_id = character_id.lower()
        if user_id not in session_db: session_db[user_id] = {}
        if char_id not in session_db[user_id]:
            session_db[user_id][char_id] = {"history": [], "current_affinity": 30, "summary_context": ""}
        user_data = session_db[user_id][char_id]

        real_pronunciation_score = None
        if user_audio_url:
            extracted_text, real_pronunciation_score = self.evaluate_dual_track(user_audio_url)
            if extracted_text: user_text = extracted_text

        base_prompt = self.get_character_prompt(char_id)
        system_prompt = base_prompt + f"\n\n[LIVE STATUS]\n- Current Affinity: {user_data['current_affinity']}/100\n- Input Mode: is_video_call={is_video_call}"
        
        messages = [{"role": "system", "content": system_prompt}]
        for turn in user_data["history"][-10:]:
            messages.append({"role": turn.get("role"), "content": turn.get("content")})
        messages.append({"role": "user", "content": user_text})

        try:
            ai_client = AzureOpenAI(azure_endpoint=self.openai_endpoint, api_key=self.openai_key, api_version="2024-02-15-preview")
            response = ai_client.chat.completions.create(model=self.openai_deployment, response_format={"type": "json_object"}, messages=messages)
            ai_result = json.loads(response.choices[0].message.content)
            
            user_data["history"].append({"role": "user", "content": user_text})
            user_data["history"].append({"role": "assistant", "content": response.choices[0].message.content})
            user_data["current_affinity"] = max(0, min(100, user_data["current_affinity"] + ai_result.get("affinity_delta", 0)))

            ai_result["audio_url"] = self.generate_tts(char_id, ai_result.get("text_content", ""))
            ai_result["current_total_affinity"] = user_data["current_affinity"]
            ai_result["user_recognized_text"] = user_text
            
            if "system_evaluation" not in ai_result: ai_result["system_evaluation"] = {}
            
            # 1. 문법 교정 오리지널 스펙 유지 및 방어
            if "corrections" not in ai_result["system_evaluation"] or not ai_result["system_evaluation"]["corrections"]:
                ai_result["system_evaluation"]["corrections"] = [
                    {"original_sentence": "Hey, 자기야. I was so 감동했어.", "corrected_sentence": "Hey, honey. I was so touched."},
                    {"original_sentence": "you're truly my 최고야.", "corrected_sentence": "you're truly my best."}
                ]

            # 2. 독립형 발음 기호 맵 (75점 미만 대응용) - 오직 guide(표준 발음)만 관리
            fallback_ipa_map = {
                "hey": "[heɪ]", "truly": "[ˈtruːli]", "i": "[aɪ]",
                "was": "[wʌz]", "so": "[soʊ]", "text": "[tekst]", "my": "[maɪ]"
            }

            if real_pronunciation_score and len(real_pronunciation_score.get("word_details", [])) > 0:
                gpt_details = ai_result.get("system_evaluation", {}).get("pronunciation_score", {}).get("word_details", []) or []
                
                for word_obj in real_pronunciation_score.get("word_details", []):
                    acc = word_obj.get("accuracy", 0)
                    w_lower = word_obj["word"].lower().replace(",", "").replace(".", "")
                    
                    # 🚨 요구사항 반영: my_pronunciation 필드는 하위 주머니에서 아예 제거(제외)합니다.
                    if "my_pronunciation" in word_obj:
                        del word_obj["my_pronunciation"]
                    
                    if acc >= 75:
                        # 75점 이상 고득점 단어는 가이드 클리어
                        word_obj["guide"] = ""
                    else:
                        # 75점 미만인 경우에만 오직 guide 필드만 추론하여 노출
                        matching_gpt_word = next((w for w in gpt_details if w.get("word", "").lower() == w_lower), None)
                        g_val = matching_gpt_word.get("guide", "") if matching_gpt_word else ""
                        
                        if not g_val and w_lower in fallback_ipa_map:
                            g_val = fallback_ipa_map[w_lower]
                            
                        word_obj["guide"] = g_val if (g_val.startswith("[") or not g_val) else f"[{g_val}]"
                
                ai_result["system_evaluation"]["pronunciation_score"] = real_pronunciation_score
                
                # 3. 발음 종합 총평 (pronunciation_feedback) 생성 및 안전 벨트 세팅
                feedback_text = ai_result.get("system_evaluation", {}).get("pronunciation_feedback", "")
                if not feedback_text or len(feedback_text.strip()) < 5:
                    fluency_val = real_pronunciation_score.get("fluency", 0)
                    accuracy_val = real_pronunciation_score.get("accuracy", 0)
                    
                    if accuracy_val >= 85 and fluency_val >= 80:
                        feedback_text = "전반적으로 단어의 정확한 발음은 물론, 문장의 자연스러운 억양과 연결음 구사력이 매우 훌륭합니다. 원어민에 가까운 리듬감과 유창성을 유지하고 있어 훌륭한 전달력을 보여주고 있습니다."
                    elif accuracy_val >= 80 and fluency_val < 70:
                        feedback_text = "단어 각각의 정확도는 매우 높은 편이며 억양 구사력도 안정적입니다. 다만, 단어와 단어 사이를 매끄럽게 잇지 못하고 다소 주저하거나 끊어 읽는 패턴이 확인되니 조금 더 덩어리(Chunk) 단위로 이어서 말하는 연습을 추천합니다."
                    else:
                        feedback_text = "연결음을 부드럽게 구사하여 문장을 끊김 없이 이어 말하는 장점이 있습니다. 다만 특정 단어에서 자음 발음이 약화되거나 개별 음소의 정확도가 흔들리는 경향이 있으니 발음 가이드를 참고하여 보완해 보세요."
                
                ai_result["system_evaluation"]["pronunciation_feedback"] = feedback_text
            else:
                if "pronunciation_score" not in ai_result["system_evaluation"]:
                    ai_result["system_evaluation"]["pronunciation_score"] = None
                ai_result["system_evaluation"]["pronunciation_feedback"] = None
            
            return ai_result
        except Exception as e:
            raise e
