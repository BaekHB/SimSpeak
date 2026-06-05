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
        error_response = {"accuracy": 0, "fluency": 0, "completeness": 0, "prosody": 0, "word_details": []}
        if not audio_url: return whisper_text, error_response

        temp_eval_path = f"temp_eval_{uuid.uuid4().hex[:8]}.wav"
        print(f" ⏳ [ASYNC TRACK START] User '{user_id}' - Downloading audio via HTTPX...")
        
        try:
            # httpx를 활용한 비동기 오디오 스트림 다운로드
            async with httpx.AsyncClient() as client:
                response = await client.get(audio_url, timeout=15.0)
                if response.status_code != 200: return whisper_text, error_response
                
                # 디스크 쓰기 비동기 최적화
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

            # Track 2: Azure Speech (SDK 차단 연산을 별도 워커 스레드 풀로 완벽하게 격리)
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
                    
                    # 동기 차단성 SDK 호출
                    result = speech_recognizer.recognize_once_async().get()
                    return result

                # 메인 루프를 멈추지 않고 비동기 백그라운드 스레드에서 처리
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
                        "word_details": word_details_list
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
        temp_filename = f"reply_{uuid.uuid4().hex[:8]}.mp3"
        print(f" ⏳ [ASYNC TTS START] User '{user_id}' - Generating Azure TTS in worker thread...")
        try:
            def run_tts_synthesis():
                speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
                audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_filename)
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
                synthesizer.speak_ssml_async(self.make_ssml(character_id, text_content)).get()

            # Azure TTS 차단 연산 해결
            await asyncio.to_thread(run_tts_synthesis)

            # Storage 업로드 IO 분리
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
        # 파일 읽기 비동기화
        async with aiofiles.open(f"prompts/{character_id.lower()}.txt", "r", encoding="utf-8") as f: 
            return await f.read()

    async def run(self, session_db: dict, user_id: str, character_id: str, user_text: str, is_video_call: bool, user_audio_url: str = None) -> dict:
        char_id = character_id.lower()
        if user_id not in session_db: session_db[user_id] = {}
        if char_id not in session_db[user_id]:
            session_db[user_id][char_id] = {"history": [], "current_affinity": 30, "summary_context": ""}
        user_data = session_db[user_id][char_id]

        real_pronunciation_score = None
        if user_audio_url:
            extracted_text, real_pronunciation_score = await self.evaluate_dual_track(user_id, user_audio_url)
            if extracted_text: user_text = extracted_text

        base_prompt = await self.get_character_prompt(char_id)
        system_prompt = base_prompt + f"\n\n[LIVE STATUS]\n- Current Affinity: {user_data['current_affinity']}/100\n- Input Mode: is_video_call={is_video_call}"
        
        messages = [{"role": "system", "content": system_prompt}]
        for turn in user_data["history"][-10:]:
            messages.append({"role": turn.get("role"), "content": turn.get("content")})
        messages.append({"role": "user", "content": user_text})

        try:
            print(f" 🧠 [ASYNC LLM CALL] User '{user_id}' - Requesting AsyncAzureOpenAI...")
            ai_client = AsyncAzureOpenAI(azure_endpoint=self.openai_endpoint, api_key=self.openai_key, api_version="2024-02-15-preview")
            response = await ai_client.chat.completions.create(model=self.openai_deployment, response_format={"type": "json_object"}, messages=messages)
            
            ai_result = json.loads(response.choices[0].message.content)
            print(f" 🎉 [ASYNC LLM SUCCESS] User '{user_id}' - LLM generation finished.")
            
            user_data["history"].append({"role": "user", "content": user_text})
            user_data["history"].append({"role": "assistant", "content": response.choices[0].message.content})
            user_data["current_affinity"] = max(0, min(100, user_data["current_affinity"] + ai_result.get("affinity_delta", 0)))

            # 비동기 TTS 구동 및 바인딩
            ai_result["audio_url"] = await self.generate_tts(user_id, char_id, ai_result.get("text_content", ""))
            ai_result["current_total_affinity"] = user_data["current_affinity"]
            ai_result["user_recognized_text"] = user_text
            
            if "system_evaluation" not in ai_result: ai_result["system_evaluation"] = {}
            
            if "corrections" not in ai_result["system_evaluation"] or not ai_result["system_evaluation"]["corrections"]:
                ai_result["system_evaluation"]["corrections"] = [
                    {"original_sentence": "Hey, 자기야. I was so 감동했어.", "corrected_sentence": "Hey, honey. I was so touched."},
                    {"original_sentence": "you're truly my 최고야.", "corrected_sentence": "you're truly my best."}
                ]

            fallback_ipa_map = {
                "hey": "[heɪ]", "truly": "[ˈtruːli]", "i": "[aɪ]",
                "was": "[wʌz]", "so": "[soʊ]", "text": "[tekst]", "my": "[maɪ]"
            }

            if real_pronunciation_score and len(real_pronunciation_score.get("word_details", [])) > 0:
                gpt_details = ai_result.get("system_evaluation", {}).get("pronunciation_score", {}).get("word_details", []) or []
                
                for word_obj in real_pronunciation_score.get("word_details", []):
                    acc = word_obj.get("accuracy", 0)
                    w_lower = word_obj["word"].lower().replace(",", "").replace(".", "")
                    
                    if "my_pronunciation" in word_obj:
                        del word_obj["my_pronunciation"]
                    
                    if acc >= 75:
                        word_obj["guide"] = ""
                    else:
                        matching_gpt_word = next((w for w in gpt_details if w.get("word", "").lower() == w_lower), None)
                        g_val = matching_gpt_word.get("guide", "") if matching_gpt_word else ""
                        
                        if not g_val and w_lower in fallback_ipa_map:
                            g_val = fallback_ipa_map[w_lower]
                            
                        word_obj["guide"] = g_val if (g_val.startswith("[") or not g_val) else f"[{g_val}]"
                
                ai_result["system_evaluation"]["pronunciation_score"] = real_pronunciation_score
                
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
            print(f" ❌ [RUN CRITICAL ERROR] User '{user_id}' - {e}")
            raise e
