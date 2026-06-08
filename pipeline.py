import os
import uuid
import json
import asyncio
import httpx
import aiofiles
from openai import AsyncAzureOpenAI
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

    async def evaluate_dual_track(self, user_id: str, audio_url: str) -> tuple[str, dict]:
        whisper_text = ""
        error_response = {"accuracy": 0, "fluency": 0, "completeness": 0, "prosody": 0, "word_details_json": []}
        if not audio_url: return whisper_text, error_response

        temp_eval_path = f"temp_eval_{uuid.uuid4().hex[:8]}.wav"
        print(f" ⏳ [ASYNC TRACK START] User '{user_id}' - Downloading audio via HTTPX...")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(audio_url, timeout=15.0)
                if response.status_code != 200: return whisper_text, error_response
                
                async with aiofiles.open(temp_eval_path, "wb") as f:
                    await f.write(response.content)

            print(f" 🚀 [ASYNC FLOW] User '{user_id}' - Audio download complete. Running Whisper & Speech Evaluation...")

            # Track 1: Whisper 비동기 클라이언트 호출
            try:
                whisper_client = AsyncAzureOpenAI(
                    azure_endpoint=self.whisper_endpoint, 
                    api_key=self.whisper_key, 
                    api_version="2024-02-15-preview"
                )
                with open(temp_eval_path, "rb") as audio_file:
                    whisper_result = await whisper_client.audio.transcriptions.create(
                        file=audio_file, 
                        model=self.whisper_deployment, 
                        prompt="Hello! 안녕하세요."
                    )
                whisper_text = whisper_result.text
                print(f" 🔍 [ASYNC FLOW] User '{user_id}' - Whisper Text Extracted: '{whisper_text}'")
            except Exception as e:
                print(f" ❌ [WHISPER ERROR] User '{user_id}' - {e}")

            # Track 2: Azure Speech SDK
            detailed_score = error_response
            try:
                def run_speech_assessment():
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
                    return result

                result = await asyncio.to_thread(run_speech_assessment)
                
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
                        "word_details_json": word_details_list
                    }
                    print(f" ✅ [ASYNC FLOW] User '{user_id}' - Pronunciation Score calculated successfully.")
            except Exception as e:
                print(f" ❌ [SPEECH ERROR] User '{user_id}' - {e}")
                
            return whisper_text, detailed_score
        except Exception as e:
            print(f" ❌ [DUAL TRACK CRITICAL ERROR] User '{user_id}' - {e}")
            return whisper_text, error_response
        finally:
            if os.path.exists(temp_eval_path):
                try: os.remove(temp_eval_path)
                except: pass

    async def generate_tts(self, user_id: str, character_id: str, text_content: str) -> str:
        if not text_content or text_content.strip() == "":
            return "https://9aifinalteam4.blob.core.windows.net/audio-files/reply_8e9e195b.mp3"
            
        temp_filename = f"reply_{uuid.uuid4().hex[:8]}.mp3"
        print(f" ⏳ [ASYNC TTS START] User '{user_id}' - Generating Azure TTS in worker thread...")
        try:
            def run_tts_synthesis():
                speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
                audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_filename)
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
                synthesizer.speak_ssml_async(self.make_ssml(character_id, text_content)).get()

            await asyncio.to_thread(run_tts_synthesis)

            def upload_to_blob():
                blob_service_client = BlobServiceClient.from_connection_string(self.storage_connection)
                blob_client = blob_service_client.get_blob_client(container="audio-files", blob=temp_filename)
                with open(temp_filename, "rb") as data: 
                    blob_client.upload_blob(data, overwrite=True)
                return blob_client.url

            blob_url = await asyncio.to_thread(upload_to_blob)
            print(f" ✅ [ASYNC TTS END] User '{user_id}' - TTS file uploaded: {blob_url}")
            return blob_url
        except Exception as e:
            print(f" ❌ [TTS ERROR] User '{user_id}' - {e}. Falling back to default url.")
            return "https://9aifinalteam4.blob.core.windows.net/audio-files/reply_8e9e195b.mp3"
        finally:
            if os.path.exists(temp_filename):
                try: os.remove(temp_filename)
                except: pass

    async def get_character_prompt(self, character_id: str) -> str:
        async with aiofiles.open(f"prompts/{character_id.lower()}.txt", "r", encoding="utf-8") as f: 
            return await f.read()

    async def run(self, session_db: dict, user_id: str, character_id: str, user_text: str, is_video_call: bool, user_audio_url: str = None) -> dict:
        char_id = character_id.lower()
        if user_id not in session_db: session_db[user_id] = {}
        if char_id not in session_db[user_id]:
            session_db[user_id][char_id] = {"history": [], "current_affinity": 30, "summary_context": ""}
        user_data = session_db[user_id][char_id]

        real_pronunciation_evaluations = None
        if user_audio_url:
            extracted_text, real_pronunciation_evaluations = await self.evaluate_dual_track(user_id, user_audio_url)
            if extracted_text: user_text = extracted_text

        base_prompt = await self.get_character_prompt(char_id)
        system_prompt = base_prompt + f"\n\n[LIVE STATUS]\n- Current Affinity: {user_data['current_affinity']}/100\n- Input Mode: is_video_call={is_video_call}"
        
        messages = [{"role": "system", "content": system_prompt}]
        for turn in user_data["history"][-10:]:
            messages.append({"role": turn.get("role"), "content": turn.get("content")})
        messages.append({"role": "user", "content": user_text})

        ai_result = None
        max_retries = 3
        retry_count = 0
        
        # --- [오류 복구 가드 루프 레이어 탑재] ---
        while retry_count < max_retries:
            try:
                print(f" 🧠 [ASYNC LLM CALL] User '{user_id}' - Requesting AsyncAzureOpenAI (Try {retry_count + 1}/{max_retries})...")
                ai_client = AsyncAzureOpenAI(azure_endpoint=self.openai_endpoint, api_key=self.openai_key, api_version="2024-02-15-preview")
                response = await ai_client.chat.completions.create(model=self.openai_deployment, response_format={"type": "json_object"}, messages=messages)
                
                raw_content = response.choices[0].message.content
                if not raw_content or raw_content.strip() == "":
                    raise ValueError("LLM이 텅 빈 응답을 리턴했습니다.")
                    
                parsed_json = json.loads(raw_content)
                
                # 필수 뼈대 필드가 누락되었는지 정밀 검증
                if "text_content" not in parsed_json or "action_description" not in parsed_json:
                    raise KeyError("필수 출력 키값(text_content 또는 action_description)이 누락되었습니다.")
                if "system_evaluation" not in parsed_json:
                    parsed_json["system_evaluation"] = {}
                
                # 검증에 완벽히 통과하면 루프 탈출
                ai_result = parsed_json
                print(f" 🎉 [ASYNC LLM SUCCESS] User '{user_id}' - LLM generation verified on Try {retry_count + 1}.")
                break
                
            except (json.JSONDecodeError, KeyError, ValueError, Exception) as error_ex:
                retry_count += 1
                print(f" ⚠️ [LLM FORMAT ERROR] User '{user_id}' - 포맷 오류 감지 (회차: {retry_count}): {error_ex}")
                
                if retry_count < max_retries:
                    # LLM에게 규격을 다시 지키라고 경고 피드백을 주입하여 보정 재시도 유도
                    messages.append({
                        "role": "system", 
                        "content": "[SYSTEM WARNING] 반환된 출력 포맷이 손상되었거나 지정된 Key가 누락되었습니다. 마크다운을 떼고 명세된 스펙의 순수 JSON 포맷으로만 다시 정확히 답변해 주세요."
                    })
                    await asyncio.sleep(0.5)  # 짧은 간격 지연 후 재요청
                else:
                    print(f" 🚨 [LLM RETRY EXCEEDED] User '{user_id}' - 최대 재시도 횟수를 초과했습니다. 비상 Fallback 데이터셋을 배포합니다.")

        # 최후의 보루: 3회 재시도가 모두 실패할 경우 서버 크래시를 차단하고 안전한 더미 데이터 조립
        if ai_result is None:
            ai_result = {
                "text_content": "Oh, sorry! I got a bit distracted for a second. What were you saying, love?" if char_id == "liam" else "Oh, sorry! I got distracted. What were you saying?",
                "action_description": "어색한 듯 머리를 긁적이며 여유롭게 웃어 보인다.",
                "affinity_delta": 0,
                "system_notification": "",
                "is_active": True,
                "system_evaluation": {
                    "is_penalty": False,
                    "grammar_feedback": "시스템 응답이 지연되어 실시간 문법 피드백을 로드하지 못했습니다.",
                    "corrections_json": [],
                    "pronunciation_evaluations": None,
                    "pronunciation_feedback": None
                }
            }

        # 가상 세션에 대화 히스토리 및 호감도 누적 반영
        user_data["history"].append({"role": "user", "content": user_text})
        user_data["history"].append({"role": "assistant", "content": json.dumps(ai_result, ensure_ascii=False)})
        user_data["current_affinity"] = max(0, min(100, user_data["current_affinity"] + ai_result.get("affinity_delta", 0)))

        # 비동기 TTS 구동 및 바인딩
        ai_result["audio_url"] = await self.generate_tts(user_id, char_id, ai_result.get("text_content", ""))
        ai_result["current_total_affinity"] = user_data["current_affinity"]
        ai_result["user_recognized_text"] = user_text
        
        # --- ERD 기준 매핑에 따른 데이터 Key 최종 정제 보증 레이어 ---
        gpt_eval = ai_result.get("system_evaluation", {})
        if not isinstance(gpt_eval, dict):
            gpt_eval = {}
            ai_result["system_evaluation"] = gpt_eval

        # 1. corrections -> corrections_json 명칭 변경 스위칭
        if "corrections" in gpt_eval:
            gpt_eval["corrections_json"] = gpt_eval.pop("corrections")
        if "corrections_json" not in gpt_eval or not gpt_eval["corrections_json"]:
            gpt_eval["corrections_json"] = []

        fallback_ipa_map = {
            "hey": "[heɪ]", "truly": "[ˈtruːli]", "i": "[aɪ]",
            "was": "[wʌz]", "so": "[soʊ]", "text": "[tekst]", "my": "[maɪ]"
        }

        # 2. pronunciation_score -> pronunciation_evaluations 명칭 스위칭 및 내부 가이드 매핑
        if "pronunciation_score" in gpt_eval:
            gpt_eval.pop("pronunciation_score")

        if real_pronunciation_evaluations and len(real_pronunciation_evaluations.get("word_details_json", [])) > 0:
            for word_obj in real_pronunciation_evaluations.get("word_details_json", []):
                acc = word_obj.get("accuracy", 0)
                w_lower = word_obj["word"].lower().replace(",", "").replace(".", "")
                
                if "my_pronunciation" in word_obj:
                    del word_obj["my_pronunciation"]
                
                if acc >= 75:
                    word_obj["guide"] = ""
                else:
                    if w_lower in fallback_ipa_map:
                        word_obj["guide"] = fallback_ipa_map[w_lower]
                    else:
                        word_obj["guide"] = ""
            
            gpt_eval["pronunciation_evaluations"] = real_pronunciation_evaluations
            
            feedback_text = gpt_eval.get("pronunciation_feedback", "")
            if not feedback_text or len(feedback_text.strip()) < 5:
                fluency_val = real_pronunciation_evaluations.get("fluency", 0)
                accuracy_val = real_pronunciation_evaluations.get("accuracy", 0)
                
                if accuracy_val >= 85 and fluency_val >= 80:
                    feedback_text = "전반적으로 단어의 정확한 발음은 물론, 문장의 자연스러운 억양과 연결음 구사력이 매우 훌륭합니다."
                elif accuracy_val >= 80 and fluency_val < 70:
                    feedback_text = "단어 각각의 정확도는 높은 편이나 단어 사이를 매끄럽게 잇지 못하니 덩어리(Chunk) 단위로 말하는 연습을 추천합니다."
                else:
                    feedback_text = "연결음을 부드럽게 구사하는 장점이 있으나 개별 음소의 정확도가 흔들리는 경향이 있으니 발음 가이드를 참고해 보세요."
            gpt_eval["pronunciation_feedback"] = feedback_text
        else:
            gpt_eval["pronunciation_evaluations"] = None
            gpt_eval["pronunciation_feedback"] = None
        
        return ai_result
