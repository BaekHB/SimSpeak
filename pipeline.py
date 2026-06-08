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

    async def call_llm_with_retry(self, ai_client, **kwargs):
        max_retries = 3
        backoff = 2.0
        for attempt in range(max_retries + 1):
            try:
                return await ai_client.chat.completions.create(**kwargs)
            except Exception as e:
                is_rate_limit = False
                if "RateLimitError" in type(e).__name__ or (hasattr(e, "status_code") and e.status_code == 429):
                    is_rate_limit = True
                
                if is_rate_limit and attempt < max_retries:
                    sleep_time = backoff * (2 ** attempt)
                    print(f" ⚠️ [RATE LIMIT] 429 hit. Retrying in {sleep_time}s (Attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(sleep_time)
                else:
                    raise e

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
                        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme
                    )
                    pronunciation_config.phoneme_alphabet = "IPA"
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
                        # 네이티브 IPA 자모들을 하나의 단어 가이드 발음기호로 합성
                        ipa_guide = ""
                        if word.phonemes:
                            ipa_guide = f"[{''.join(p.phoneme for p in word.phonemes)}]"
                            
                        # 단어 평가 75점 미만일 때만 guide 필드에 발음 기호를 주입하고, 그 외에는 공백("") 처리
                        guide = ipa_guide if word.accuracy_score < 75 else ""
                        
                        word_details_list.append({
                            "word": word.word.strip(),
                            "accuracy": int(word.accuracy_score),
                            "error_type": word.error_type if word.error_type != "None" else None,
                            "guide": guide
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

    async def run(self, session_db: dict, user_id: str, character_id: str, user_text: str, is_video_call: bool, user_audio_url: str = None, stage_id: str = "stage_1") -> dict:
        char_id = character_id.lower()
        if user_id not in session_db: session_db[user_id] = {}
        if char_id not in session_db[user_id]:
            session_db[user_id][char_id] = {"history": [], "current_affinity": 30, "summary_context": ""}
        user_data = session_db[user_id][char_id]
        
        if "summary_context" not in user_data:
            user_data["summary_context"] = ""
        current_summary = user_data["summary_context"]

        real_pronunciation_evaluations = None
        if user_audio_url:
            extracted_text, real_pronunciation_evaluations = await self.evaluate_dual_track(user_id, user_audio_url)
            if extracted_text: user_text = extracted_text

        # AsyncAzureOpenAI 인스턴스 미리 생성
        ai_client = AsyncAzureOpenAI(
            azure_endpoint=self.openai_endpoint,
            api_key=self.openai_key,
            api_version="2024-02-15-preview"
        )

        # [토큰 절약 엔진 가동] 10턴 초과 시 압축 로그 작동
        if len(user_data["history"]) > 10:
            overflow_turns = user_data["history"][:-10]
            overflow_text = ""
            for turn in overflow_turns:
                overflow_text += f"{turn['role']}: {turn['content']}\n"
                
            summary_command = [
                {"role": "system", "content": "너는 기억 파수꾼이야. 기존 [누적 요약본]에 새로 잊혀지려는 [대화 조각]의 핵심 사건이나 유저 정보만 결합해서 한 문장의 한국어로 지속 업데이트해 줘. 대화 로그 형식은 금지한다."},
                {"role": "user", "content": f"[기존 누적 요약본]\n{current_summary}\n\n[새 대화 조각]\n{overflow_text}"}
            ]
            
            try:
                summary_response = await self.call_llm_with_retry(
                    ai_client,
                    model=self.openai_deployment,
                    messages=summary_command,
                    max_tokens=150
                )
                current_summary = summary_response.choices[0].message.content.strip()
                user_data["summary_context"] = current_summary
            except Exception as e:
                print(f"[Warning] Summary engine temporary error: {e}")

        base_prompt = await self.get_character_prompt(char_id)
        summary_prefix = f"[PAST CONVERSATION SUMMARY]\n{current_summary}\n\n" if current_summary else ""
        system_prompt = summary_prefix + base_prompt + f"\n\n[LIVE STATUS]\n- Current Affinity: {user_data['current_affinity']}/100\n- Input Mode: is_video_call={is_video_call}"
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # GPT history 클리닝 처리 (JSON 문자열 대신 퓨어 텍스트만 전달하여 챗봇 응답 안정성 확보)
        refined_history = []
        for turn in user_data["history"]:
            role = turn.get("role")
            content_raw = turn.get("content", "")
            
            if role == "user":
                refined_history.append({"role": "user", "content": content_raw})
            elif role == "assistant":
                try:
                    data = json.loads(content_raw)
                    pure_text = data.get("text_content") or data.get("content") or content_raw
                    refined_history.append({"role": "assistant", "content": pure_text})
                except Exception:
                    refined_history.append({"role": "assistant", "content": content_raw})
                    
        for turn in refined_history[-10:]:
            messages.append({"role": turn.get("role"), "content": turn.get("content")})
        messages.append({"role": "user", "content": user_text})

        ai_result = None
        max_retries = 3
        retry_count = 0
        raw_usage_data = {}
        last_response_text = ""

        # --- [오류 복구 가드 루프 레이어 탑재] ---
        while retry_count < max_retries:
            try:
                print(f" 🧠 [ASYNC LLM CALL] User '{user_id}' - Requesting AsyncAzureOpenAI (Try {retry_count + 1}/{max_retries})...")
                response = await self.call_llm_with_retry(
                    ai_client, 
                    model=self.openai_deployment, 
                    response_format={"type": "json_object"}, 
                    messages=messages
                )
                
                last_response_text = response.choices[0].message.content
                if not last_response_text or last_response_text.strip() == "":
                    raise ValueError("LLM이 텅 빈 응답을 리턴했습니다.")
                    
                parsed_json = json.loads(last_response_text)
                
                # 필수 출력 키값(text_content 또는 action_description) 검증
                if "text_content" not in parsed_json or "action_description" not in parsed_json:
                    raise KeyError("필수 출력 키값(text_content 또는 action_description)이 누락되었습니다.")
                if "system_evaluation" not in parsed_json:
                    parsed_json["system_evaluation"] = {}
                
                ai_result = parsed_json
                
                # 대표님 보고용 토큰 및 원본 생로그 추출
                raw_usage_data = {
                    "usage": response.usage.model_dump() if hasattr(response, "usage") and response.usage else {},
                    "model": response.model,
                    "choices": [{"finish_reason": c.finish_reason, "index": c.index} for c in response.choices]
                }
                
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
            raw_usage_data = {"error": "LLM format retry exceeded. Fallback dummy used."}
            last_response_text = json.dumps(ai_result, ensure_ascii=False)

        # [스테이지별 한국어 혼용률 페널티 판정]
        words = user_text.split()
        if not words:
            korean_ratio = 0.0
        else:
            korean_word_count = sum(1 for word in words if any(
                (0xAC00 <= ord(c) <= 0xD7A3) or
                (0x3130 <= ord(c) <= 0x318F) or
                (0x1100 <= ord(c) <= 0x11FF)
                for c in word
            ))
            korean_ratio = korean_word_count / len(words)

        stage_clean = str(stage_id).lower().strip().replace(" ", "_")
        threshold = 0.30  # 기본값
        if stage_clean in ["stage_1", "stage_2"]:
            threshold = 0.30
        elif stage_clean in ["stage_3", "stage_4", "stage_5", "stage_6", "bonus_1", "bonus_stage_1"]:
            threshold = 0.20
        elif stage_clean in ["stage_7", "stage_8", "bonus_2", "bonus_stage_2"]:
            threshold = 0.10

        if korean_ratio >= threshold:
            print(f"[Stage Penalty] Triggered! Korean ratio: {korean_ratio:.2f} >= threshold: {threshold:.2f} in stage_id: {stage_id}")
            ai_result["affinity_delta"] = -1
            if "system_evaluation" not in ai_result:
                ai_result["system_evaluation"] = {}
            ai_result["system_evaluation"]["is_penalty"] = True
        
        user_data["history"].append({"role": "user", "content": user_text})
        user_data["history"].append({"role": "assistant", "content": last_response_text})
        user_data["current_affinity"] = max(0, min(100, user_data["current_affinity"] + ai_result.get("affinity_delta", 0)))

        if "system_evaluation" not in ai_result: ai_result["system_evaluation"] = {}
        
        # 1. corrections -> corrections_json 명칭 스위칭 보정
        if "corrections" in ai_result["system_evaluation"]:
            ai_result["system_evaluation"]["corrections_json"] = ai_result["system_evaluation"].pop("corrections")
        
        # (1) corrections_json 리스트가 비어 있거나 없을 시 디폴트 원어민 교정 문장 fallback 주입
        if "corrections_json" not in ai_result["system_evaluation"] or not ai_result["system_evaluation"]["corrections_json"]:
            ai_result["system_evaluation"]["corrections_json"] = [
                {
                    "original_sentence": user_text,
                    "corrected_sentence": user_text
                }
            ]

        # 비동기 TTS 구동 및 바인딩 (보이스 모드일 때는 AI 답변과 문법 교정문(corrected_sentence) 가이드를 병렬로 가동하여 대기 시간 최적화)
        if user_audio_url:
            ai_audio_task = self.generate_tts(user_id, char_id, ai_result.get("text_content", ""))
            
            # corrections_json의 각 corrected_sentence에 대한 TTS 생성 태스크 리스트 구축
            corrections_list = ai_result["system_evaluation"]["corrections_json"]
            tts_tasks = []
            for corr in corrections_list:
                sentence = corr.get("corrected_sentence", "")
                if sentence:
                    tts_tasks.append(self.generate_tts(user_id, char_id, sentence))
                else:
                    async def dummy_none(): return None
                    tts_tasks.append(dummy_none())
            
            # 병렬 합성 가동
            results = await asyncio.gather(ai_audio_task, *tts_tasks)
            
            ai_result["audio_url"] = results[0]
            corrected_urls = results[1:]
            for corr, url in zip(corrections_list, corrected_urls):
                corr["corrected_audio_url"] = url
        else:
            ai_result["audio_url"] = await self.generate_tts(user_id, char_id, ai_result.get("text_content", ""))
            # 텍스트 모드에서는 가이드 음성을 생성하지 않고 None으로 채웁니다.
            corrections_list = ai_result["system_evaluation"]["corrections_json"]
            for corr in corrections_list:
                corr["corrected_audio_url"] = None

        ai_result["current_total_affinity"] = user_data["current_affinity"]
        ai_result["user_recognized_text"] = user_text

        fallback_ipa_map = {
            "hey": "[heɪ]", "truly": "[ˈtruːli]", "i": "[aɪ]",
            "was": "[wʌz]", "so": "[soʊ]", "text": "[tekst]", "my": "[maɪ]"
        }

        if real_pronunciation_evaluations and len(real_pronunciation_evaluations.get("word_details_json", [])) > 0:
            # 2. GPT가 작성한 기존 pronunciation_score 필드가 있다면 제거하고 pronunciation_evaluations로 통합 적용
            gpt_details = []
            if "pronunciation_score" in ai_result["system_evaluation"]:
                gpt_details = ai_result["system_evaluation"].pop("pronunciation_score", {}).get("word_details", []) or []
            
            for word_obj in real_pronunciation_evaluations.get("word_details_json", []):
                acc = word_obj.get("accuracy", 0)
                w_lower = word_obj["word"].lower().replace(",", "").replace(".", "")
                
                if "my_pronunciation" in word_obj:
                    del word_obj["my_pronunciation"]
                
                if acc >= 75:
                    word_obj["guide"] = ""
                else:
                    # 이미 evaluate_dual_track에서 주입된 네이티브 IPA 가이드가 있는지 우선 확인
                    g_val = word_obj.get("guide", "")
                    
                    if not g_val:
                        matching_gpt_word = next((w for w in gpt_details if w.get("word", "").lower() == w_lower), None)
                        g_val = matching_gpt_word.get("guide", "") if matching_gpt_word else ""
                    
                    if not g_val and w_lower in fallback_ipa_map:
                        g_val = fallback_ipa_map[w_lower]
                        
                    word_obj["guide"] = g_val if (g_val.startswith("[") or not g_val) else f"[{g_val}]"
            
            ai_result["system_evaluation"]["pronunciation_evaluations"] = real_pronunciation_evaluations
            
            feedback_text = ai_result.get("system_evaluation", {}).get("pronunciation_feedback", "")
            if not feedback_text or len(feedback_text.strip()) < 5:
                fluency_val = real_pronunciation_evaluations.get("fluency", 0)
                accuracy_val = real_pronunciation_evaluations.get("accuracy", 0)
                
                if accuracy_val >= 85 and fluency_val >= 80:
                    feedback_text = "전반적으로 단어의 정확한 발음은 물론, 문장의 자연스러운 억양과 연결음 구사력이 매우 훌륭합니다. 원어민에 가까운 리듬감과 유창성을 유지하고 있어 훌륭한 전달력을 보여주고 있습니다."
                elif accuracy_val >= 80 and fluency_val < 70:
                    feedback_text = "단어 각각의 정확도는 매우 높은 편이며 억양 구사력도 안정적입니다. 다만, 단어와 단어 사이를 매끄럽게 잇지 못하고 다소 주저하거나 끊어 읽는 패턴이 확인되니 조금 더 덩어리(Chunk) 단위로 이어서 말하는 연습을 추천합니다."
                else:
                    feedback_text = "연결음을 부드럽게 구사하여 문장을 끊김 없이 이어 말하는 장점이 있습니다. 다만 특정 단어에서 자음 발음이 약화되거나 개별 음소의 정확도가 흔들리는 경향이 있으니 발음 가이드를 참고하여 보완해 보세요."
            
            ai_result["system_evaluation"]["pronunciation_feedback"] = feedback_text
        else:
            if "pronunciation_score" in ai_result["system_evaluation"]:
                ai_result["system_evaluation"].pop("pronunciation_score")
            ai_result["system_evaluation"]["pronunciation_evaluations"] = None
            ai_result["system_evaluation"]["pronunciation_feedback"] = None
        
        ai_result["raw_llm_log"] = raw_usage_data
        return ai_result
