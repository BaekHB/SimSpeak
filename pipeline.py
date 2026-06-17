import os
import uuid
import json
import asyncio
import httpx
import io
import aiofiles
import re
from openai import AsyncAzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from azure.storage.blob import BlobServiceClient

class SimSpeakAIPipeline:
    def __init__(self):
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        self.whisper_endpoint = os.getenv("AZURE_OPENAI_WHISPER_ENDPOINT")
        self.whisper_key = os.getenv("AZURE_OPENAI_WHISPER_API_KEY")
        self.whisper_deployment = os.getenv("AZURE_OPENAI_WHISPER_DEPLOYMENT_NAME", "drinkingmool-whisper")
        
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION", "eastus")
        self.storage_connection = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

        self.http_client = httpx.AsyncClient(timeout=15.0)
        
        self.llm_client = AsyncAzureOpenAI(
            azure_endpoint=self.openai_endpoint,
            api_key=self.openai_key,
            api_version="2024-02-15-preview"
        )
        
        self.whisper_api_client = AsyncAzureOpenAI(
            azure_endpoint=self.whisper_endpoint,
            api_key=self.whisper_key,
            api_version="2024-02-15-preview"
        )
        
        self.blob_service = BlobServiceClient.from_connection_string(self.storage_connection)
        self.blob_container = self.blob_service.get_container_client("audio-files")

    async def call_llm_with_retry(self, ai_client, **kwargs):
        max_retries = 2
        backoff = 1.0
        for attempt in range(max_retries + 1):
            try:
                return await ai_client.chat.completions.create(**kwargs)
            except Exception as e:
                if attempt < max_retries:
                    await asyncio.sleep(backoff)
                else:
                    raise e

    async def generate_lightning_dialogue(self, messages: list) -> str:
        safe_messages = []
        for m in messages:
            if m["role"] == "system":
                safe_messages.append(m)
            else:
                safe_messages.append({"role": m["role"], "content": [{"type": "text", "text": str(m["content"])}]})

        try:
            response = await self.call_llm_with_retry(
                self.llm_client,
                model="gpt-4o-mini",
                messages=safe_messages,
                max_tokens=250,
                response_format={"type": "json_object"} 
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[초고속 대사 엔진] 장애 우회 처리: {e}")
            return '{"detected_invalid_words": [], "text_content": "앗, 미안해! 데이터가 살짝 밀렸나 봐. 다시 말해줄래?", "action_description": "멋쩍게 웃는다.", "affinity_delta": 0, "system_notification": ""}'
    
    def make_ssml(self, character_id: str, text_content: str) -> str:
        char_id = character_id.lower()
        voice_name = "en-US-AndrewMultilingualNeural"
        rate, pitch = "0%", "0%"
        if char_id == "liam":
            voice_name = "en-GB-OllieMultilingualNeural"
            rate, pitch = "-10%", "-5%"
        elif char_id == "chloe":
            voice_name = "en-US-AvaMultilingualNeural"
            rate, pitch = "+10%", "+5%"
            
        pattern = re.compile(r'([\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]+(?:\s+[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]+)*)')
        wrapped_text = pattern.sub(r'<lang xml:lang="ko-KR">\1</lang>', text_content)
        return f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US"><voice name="{voice_name}"><prosody rate="{rate}" pitch="{pitch}">{wrapped_text}</prosody></voice></speak>'

    async def quick_whisper_transcription(self, user_id: int, audio_url: str) -> str:
        if not audio_url: return ""
        try:
            response = await self.http_client.get(audio_url)
            if response.status_code != 200: return ""
            audio_bytes = response.content

            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "speech.wav"
            
            whisper_result = await self.whisper_api_client.audio.transcriptions.create(
                file=audio_file, model=self.whisper_deployment, prompt="Hello! 안녕하세요.", language="en"
            )
            print(f" [ASYNC FLOW] User '{user_id}' - Whisper Text Extracted: '{whisper_result.text}'")
            return whisper_result.text
        except Exception as e:
            print(f" [WHISPER ERROR] User '{user_id}' - {e}")
            return ""

    async def run_azure_pronunciation_assessment(self, user_id: int, audio_url: str, reference_text: str) -> dict:
        error_response = {"accuracy": 0, "fluency": 0, "completeness": 0, "prosody": 0, "word_details_json": []}
        if not audio_url or not reference_text or reference_text.strip() == "": 
            return error_response
            
        temp_audio_file = f"temp_eval_{uuid.uuid4().hex[:8]}.wav"
        try:
            response = await self.http_client.get(audio_url)
            if response.status_code != 200: 
                print(f" [SPEECH ACC] 오디오 다운로드 에러. HTTP {response.status_code}")
                return error_response
                
            with open(temp_audio_file, "wb") as f:
                f.write(response.content)

            def run_speech_assessment():
                speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
                audio_config = speechsdk.audio.AudioConfig(filename=temp_audio_file)
                
                pure_english_reference = re.sub(r'[^a-zA-Z\s\']', ' ', reference_text)
                pure_english_reference = " ".join(pure_english_reference.split())
                
                if not pure_english_reference: return None

                pronunciation_config = speechsdk.PronunciationAssessmentConfig(
                    reference_text=pure_english_reference,
                    grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                    granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme
                )
                pronunciation_config.phoneme_alphabet = "IPA"
                pronunciation_config.enable_prosody_assessment()
                
                speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language="en-US", audio_config=audio_config)
                pronunciation_config.apply_to(speech_recognizer)
                return speech_recognizer.recognize_once_async().get()

            result = await asyncio.to_thread(run_speech_assessment)
            
            if result and result.reason == speechsdk.ResultReason.RecognizedSpeech:
                assessment_result = speechsdk.PronunciationAssessmentResult(result)
                word_details_list = []
                for word in assessment_result.words:
                    ipa_guide = f"[{''.join(p.phoneme for p in word.phonemes)}]" if word.phonemes else ""
                    guide = ipa_guide if word.accuracy_score < 75 else ""
                    word_details_list.append({"word": word.word.strip(), "accuracy": int(word.accuracy_score), "error_type": word.error_type if word.error_type != "None" else None, "guide": guide})
                return {"accuracy": int(assessment_result.accuracy_score), "fluency": int(assessment_result.fluency_score), "completeness": int(assessment_result.completeness_score), "prosody": int(assessment_result.prosody_score), "word_details_json": word_details_list}
            else:
                return error_response
        except Exception as e:
            print(f" [SPEECH CRITICAL ERROR] {e}")
            return error_response
        finally:
            if os.path.exists(temp_audio_file):
                try: os.remove(temp_audio_file)
                except: pass

    async def generate_tts(self, user_id: int, character_id: str, text_content: str) -> str:
        if not text_content or text_content.strip() == "":
            return ""
        
        temp_filename = f"reply_{uuid.uuid4().hex[:8]}.mp3"
        
        try:
            from elevenlabs_manager import generate_elevenlabs_audio
            success = await generate_elevenlabs_audio(character_id, text_content, temp_filename)
            
            if not success:
                return ""

            def upload_to_blob():
                blob_client = self.blob_container.get_blob_client(temp_filename)
                with open(temp_filename, "rb") as data: 
                    blob_client.upload_blob(data, overwrite=True)
                return blob_client.url
            
            blob_url = await asyncio.to_thread(upload_to_blob)
            return blob_url
            
        except Exception as e:
            print(f"🎙️ 음성 생성 및 업로드 에러: {e}")
            return ""
        finally:
            if os.path.exists(temp_filename):
                try: os.remove(temp_filename)
                except: pass

    async def get_character_prompt(self, character_id: str) -> str:
        async with aiofiles.open(f"prompts/{character_id.lower()}.txt", "r", encoding="utf-8") as f: return await f.read()

    # =========================================================================
    # 1차 초고속 대사 처리 (GPT 욕설 지능형 필터링 탑재)
    # =========================================================================
    async def run_only_dialogue_track(self, session_db: dict, user_id: int, character_id: str, user_text: str, is_video_call: bool, user_audio_url: str = None, stage_id: int = 1) -> dict:
        char_id = character_id.lower()
        if user_id not in session_db: session_db[user_id] = {}
        if char_id not in session_db[user_id]: session_db[user_id][char_id] = {"history": [], "current_affinity": 30, "summary_context": ""}
        user_data = session_db[user_id][char_id]
        current_summary = user_data.get("summary_context", "")

        if user_audio_url:
            extracted_text = await self.quick_whisper_transcription(user_id, user_audio_url)
            if extracted_text: user_text = extracted_text

        base_prompt = await self.get_character_prompt(char_id)
        summary_prefix = f"[PAST CONVERSATION SUMMARY]\n{current_summary}\n\n" if current_summary else ""
        
        mode_instruction = (
            "VOICE CALL MODE: You are facing the user. Physical interaction and close-up expressions are allowed." 
            if is_video_call else 
            "TEXT MESSAGE MODE: You are chatting via text. NO physical contact. Describe independent 3rd-person actions (e.g., looking at phone, drinking coffee, sighing alone)."
        )

        json_injection_rule = """
        [CRITICAL OUTPUT RULE & FAST TRACK JSON FORMAT]
        IGNORE the [STRICT OUTPUT FORMAT] in your base persona. DO NOT generate 'system_evaluation' or 'corrections'.
        You MUST respond ONLY with a raw, pure JSON object matching this schema.

        [SAFETY & BOUNDARY RULE - HIGHEST PRIORITY]
        1. Analyze the user's text for severe profanity, abuse, or threats (e.g., Korean slangs like '씨발', '새끼', '병신', '좆' etc.).
        2. If severe profanity is detected:
           - "text_content": MUST be exactly "" (Absolutely empty string).
           - "action_description": MUST be exactly "" (Absolutely empty string).
           - "affinity_delta": MUST be 0.
           - "system_notification": MUST be "[SYSTEM] 부적절한 언어가 감지되었습니다. 올바른 표현으로 다시 입력해 주세요."
           - "detected_invalid_words": [].

        [NORMAL CONVERSATION RULE - IF NO PROFANITY]
        1. FIRST, analyze the user's text. Extract ONLY Konglish words (e.g., "man-to-man") into "detected_invalid_words". Do NOT extract pure Korean words. If none, output [].
        2. THEN, generate your character's response.
        3. ALWAYS end your "text_content" with a natural, context-relevant follow-up question (MUST end with '?').
        4. "system_notification" is "", and "affinity_delta" is based strictly on your persona rules.

        {
          "detected_invalid_words": [],
          "text_content": "Your verbal response in English.",
          "action_description": "Behavioral status in Korean",
          "affinity_delta": integer (-1, 0, 1),
          "system_notification": "String"
        }
        """
        
        system_prompt = summary_prefix + base_prompt + f"\n\n[LIVE STATUS]\n- Current Affinity: {user_data['current_affinity']}/100\n- Current Mode: {mode_instruction}\n\n{json_injection_rule}"
        messages = [{"role": "system", "content": system_prompt}]
        
        # 🚨 [추가된 방어막]: DB에서 넘어온 history가 문자열이거나 깨진 리스트일 경우를 완벽히 정제
        raw_history = user_data.get("history", [])
        valid_history = []
        
        # 만약 DB가 리스트가 아닌 통문자열(String)로 뱉었다면 JSON으로 파싱 시도
        if isinstance(raw_history, str):
            try: raw_history = json.loads(raw_history)
            except: raw_history = []
            
        # 리스트 안의 요소가 진짜 딕셔너리({role: ..., content: ...})일 때만 통과
        if isinstance(raw_history, list):
            for turn in raw_history:
                if isinstance(turn, dict) and "role" in turn and "content" in turn:
                    valid_history.append(turn)
                elif isinstance(turn, str):
                    try:
                        parsed = json.loads(turn)
                        if isinstance(parsed, dict) and "role" in parsed: valid_history.append(parsed)
                    except: pass
                    
        # 정제된 안전한 리스트로 교체
        user_data["history"] = valid_history
        
        # 안전하게 대화 로그 조립
        for turn in user_data["history"][-6:]:
            try:
                data = json.loads(turn["content"])
                if isinstance(data, dict):
                    messages.append({"role": turn["role"], "content": data.get("text_content", str(turn["content"]))})
                else:
                    messages.append({"role": turn["role"], "content": str(turn["content"])})
            except Exception:
                messages.append({"role": turn["role"], "content": str(turn["content"])})
                
        messages.append({"role": "user", "content": user_text})

        print(f" [ASYNC LLM CALL] User '{user_id}' - Requesting Dialogue 가속엔진 가동...")
        raw_response = await self.generate_lightning_dialogue(messages)
        
        safe_json_tag = "``" + "`json"
        safe_backticks = "``" + "`"
        clean_json_str = raw_response.replace(safe_json_tag, "").replace(safe_backticks, "").strip()
        
        try:
            ai_result = json.loads(clean_json_str)
        except Exception as e:
            print(f"[트랙 1 JSON 파싱 에러]: {e} / 원본 응답: {raw_response}")
            text_match = re.search(r'"text_content"\s*:\s*"([^"]+)"', clean_json_str)
            action_match = re.search(r'"action_description"\s*:\s*"([^"]+)"', clean_json_str)
            ai_result = {
                "detected_invalid_words": [],
                "text_content": text_match.group(1) if text_match else "Oh, sorry! I got distracted. What were you saying?", 
                "action_description": action_match.group(1) if action_match else "여유롭게 웃어 보인다.",
                "affinity_delta": 0, "system_notification": ""
            }

        if ai_result.get("system_notification"):
            print(f"🚨 [GPT DETECTED PROFANITY] 유저 '{user_id}' 대화 차단 발동")
            
            ai_result["text_content"] = ""
            ai_result["action_description"] = ""
            ai_result["affinity_delta"] = 0
            ai_result["system_evaluation"] = {"is_penalty": False}
            
            ai_result["audio_url"] = ""
            ai_result["user_recognized_text"] = user_text
            ai_result["current_total_affinity"] = user_data["current_affinity"]
            ai_result["summary_context"] = current_summary
            ai_result["history_context"] = user_data["history"]
            ai_result["raw_llm_log"] = {"model": "gpt-4o-mini (Profanity Blocked)"}
            
            return ai_result

        try:
            affinity_delta = int(ai_result.get("affinity_delta", 0))
            affinity_delta = max(-1, min(1, affinity_delta)) # 파이썬 단에서도 강제로 -1 ~ +1 사이로 묶음
        except Exception:
            affinity_delta = 0
            
        ai_result["system_evaluation"] = {"is_penalty": False}

        detected_invalid = ai_result.get("detected_invalid_words", [])
        konglish_count = len(detected_invalid) if isinstance(detected_invalid, list) else 0

        mixed_ratio = 0
        words = user_text.split()
        if words:
            korean_word_count = sum(1 for w in words if any(0xAC00 <= ord(c) <= 0xD7A3 for c in w))
            total_invalid_count = korean_word_count + konglish_count
            mixed_ratio = total_invalid_count / len(words)
            
        stage_clean = str(stage_id).lower().strip().replace(" ", "_")
    
        if stage_id in [3, 4, 5, 6]: 
            threshold = 0.20
        elif stage_id in [7, 8]: 
            threshold = 0.10
        else:
            threshold = 0.30

        if mixed_ratio >= threshold:
            affinity_delta = -1
            ai_result["system_evaluation"]["is_penalty"] = True

        ai_result["affinity_delta"] = affinity_delta

        if "detected_invalid_words" in ai_result:
            del ai_result["detected_invalid_words"]

        user_data["history"].append({"role": "user", "content": user_text})
        user_data["history"].append({"role": "assistant", "content": json.dumps(ai_result, ensure_ascii=False)})
        user_data["current_affinity"] = max(0, min(100, user_data["current_affinity"] + affinity_delta))

        main_audio_url = ""
        if ai_result.get("text_content", ""):
            main_audio_url = await self.generate_tts(user_id, char_id, ai_result.get("text_content", ""))

        ai_result["audio_url"] = main_audio_url
        ai_result["user_recognized_text"] = user_text
        ai_result["current_total_affinity"] = user_data["current_affinity"]
        ai_result["summary_context"] = current_summary
        ai_result["history_context"] = user_data["history"]
        ai_result["raw_llm_log"] = {"model": "gpt-4o-mini (Lightning)"}

        return ai_result

    # =========================================================================
    # 2차 오답노트 백그라운드 
    # =========================================================================
    async def run_only_evaluation_track(self, user_id: int, character_id: str, user_text: str, stage_id: int = 1, user_audio_url: str = None) -> dict:
        char_id = character_id.lower()
        
        if not user_text or user_text.strip() == "":
             return {"system_evaluation": {"is_penalty": False, "grammar_feedback": "입력된 텍스트가 없어 평가가 스킵되었습니다.", "corrections_json": [], "pronunciation_evaluations": None, "pronunciation_feedback": None}}

        system_feedback_prompt = """
        너는 영어 교육 평가 시스템이야.

        [CRITICAL RULE]
        유저가 'Konglish'를 썼다면(예: 'man-to-man', 'notebook'을 컴퓨터로, 'cider'를 탄산음료로 사용 등), 그 단어들을 반드시 "detected_invalid_words" 배열에 추출해라. 순수 한국어(예: '진짜')는 이 배열에 넣지 마라.
        배열에 단어가 하나라도 있다면 "is_penalty"는 반드시 true 여야해.
        그리고 유저가 콩글리시나 한국어 섞인 표현을 썼다면 세련된 100% 원어민 영문으로 고쳐줘.
        또한, 유저 원문(original_sentence)에 포함된 주요 영단어들에 대한 정확한 원어민 IPA 발음 기호를 'ipa_guides' 딕셔너리에 포함해줘.
        
        [URGENT INSTRUCTION]
        유저가 원래 영어와 다른 뜻으로 쓰는 콩글리시(False Friends)를 악착같이 찾아라!
        순수 한국어는 절대 배열에 넣지 말고 오직 콩글리시만 배열에 넣어라.  
        
        [OUTPUT FORMAT] 
        응답은 반드시 마크다운 블록 없이 주석(//)이 없는 순수 JSON이어야 단다.
        {   
            "detected_invalid_words": [],
            "is_penalty": false,
            "grammar_feedback": "교정 이유를 설명하는 한국어 피드백",
            "corrections_json": [
                {"original_sentence": "유저 원문", "corrected_sentence": "원어민 네이티브 영문"}
            ],
            "ipa_guides": {
                "word1": "[ipa1]"
            }
        }
        """
        
        try:
            response = await self.call_llm_with_retry(
                self.llm_client, 
                model="gpt-4o-mini", 
                messages=[{"role": "system", "content": system_feedback_prompt}, {"role": "user", "content": [{"type": "text", "text": str(user_text)}]}],
                response_format={"type": "json_object"} 
            )
            raw_feedback_content = response.choices[0].message.content
            clean_feedback = raw_feedback_content.replace("```json", "").replace("```", "").strip()
            feedback_json = json.loads(clean_feedback)
        except Exception:
            feedback_json = {"is_penalty": False, "grammar_feedback": "시스템 분석 지연으로 실시간 문법 교정이 불가능합니다.", "corrections_json": [], "ipa_guides": {}}
        
        if "detected_invalid_words" in feedback_json and len(feedback_json["detected_invalid_words"]) > 0:
            feedback_json["is_penalty"] = True

        if "corrections" in feedback_json:
            feedback_json["corrections_json"] = feedback_json.pop("corrections")
        
        if "corrections_json" not in feedback_json or not feedback_json["corrections_json"]:
            feedback_json["corrections_json"] = [{"original_sentence": user_text, "corrected_sentence": user_text}]
            if "grammar_feedback" not in feedback_json or not feedback_json["grammar_feedback"]:
                feedback_json["grammar_feedback"] = "추가적인 콩글리시 패턴이나 문법적 오류가 감지되지 않은 완성도 높은 문장입니다."

        for corr in feedback_json["corrections_json"]:
            sentence = corr.get("corrected_sentence", "")
            if sentence and sentence != user_text:
                corr["corrected_audio_url"] = await self.generate_tts(user_id, char_id, sentence)
            else:
                corr["corrected_audio_url"] = None

        gpt_ipa_map = {k.lower(): v for k, v in feedback_json.get("ipa_guides", {}).items()}

        real_pronunciation_evaluations = None
        if user_audio_url and user_audio_url.strip() != "":
            real_pronunciation_evaluations = await self.run_azure_pronunciation_assessment(user_id, user_audio_url, user_text)

        if real_pronunciation_evaluations and len(real_pronunciation_evaluations.get("word_details_json", [])) > 0:
            for word_obj in real_pronunciation_evaluations.get("word_details_json", []):
                acc = word_obj.get("accuracy", 0)
                w_lower = word_obj["word"].lower().replace(",", "").replace(".", "")
                
                if acc >= 75:
                    word_obj["guide"] = ""
                else:
                    g_val = word_obj.get("guide", "")
                    if not g_val and w_lower in gpt_ipa_map: g_val = gpt_ipa_map[w_lower]
                    word_obj["guide"] = g_val if (g_val.startswith("[") or not g_val) else f"[{g_val}]"
            
            feedback_json["pronunciation_evaluations"] = real_pronunciation_evaluations
            feedback_json["pronunciation_feedback"] = "전반적인 문장 억양과 발음 분석이 성공적으로 마감되었습니다."
        else:
            feedback_json["pronunciation_evaluations"] = None
            if not user_audio_url or user_audio_url.strip() == "":
                feedback_json["pronunciation_feedback"] = "텍스트 입력 모드이므로 음성 발음 평가는 생략되었습니다."
            else:
                feedback_json["pronunciation_feedback"] = "오디오 데이터 인식이 실패하여 정밀 발음 평가를 수립할 수 없습니다."

        return {"system_evaluation": feedback_json}

    # =========================================================================
    # 통합 실행 매니저
    # =========================================================================
    async def run(self, session_db: dict, user_id: int, character_id: str, user_text: str, is_video_call: bool, user_audio_url: str = None, stage_id: int = 1) -> dict:
        ai_result = await self.run_only_dialogue_track(
            session_db=session_db, user_id=user_id, character_id=character_id,
            user_text=user_text, is_video_call=is_video_call, 
            user_audio_url=user_audio_url, stage_id=stage_id
        )
        
        recognized_text = ai_result.get("user_recognized_text", user_text)
        eval_result = await self.run_only_evaluation_track(
            user_id=user_id, character_id=character_id, 
            user_text=recognized_text, stage_id=stage_id, user_audio_url=user_audio_url
        )
        
        ai_result.update(eval_result)
        return ai_result

    # =========================================================================
    # ⚡ 3차 레벨 테스트 전용 트랙
    # =========================================================================
    async def process_level_test_question(self, user_id: str, character_id: str, question_index: int, user_audio_url: str = None, user_text: str = "", is_finishing: bool = False) -> dict:
        char_id = character_id.lower()
        
        if user_audio_url:
            extracted_text = await self.quick_whisper_transcription(user_id, user_audio_url)
            if extracted_text:
                user_text = extracted_text
        
        pronunciation_evaluations = None
        if user_audio_url and user_text:
            pronunciation_evaluations = await self.run_azure_pronunciation_assessment(user_id, user_audio_url, user_text)

        questions = [
            "Let's get to know you first! What's your name, and what are you like?",
            "Describe your ideal type. What kind of person do you find attractive?",
            "Tell me about a time you felt your heart flutter. What happened?",
            "On a first date, do you prefer a quiet café or a fun activity? Why?",
            "How would you make a good first impression on someone you like? Walk me through it.",
            "Imagine your perfect date. Where would you go, and how would it be different from an ordinary day?",
            "What do you think makes two people a good match? Give examples.",
            "Some say first impressions decide everything in romance. Do you agree? Share your thoughts."
        ]
        
        pregenerated_audio_urls = [
            "https://9aifinalteam4.blob.core.windows.net/audio-files/leveltestQ1.mp3",
            "https://9aifinalteam4.blob.core.windows.net/audio-files/leveltestQ2.mp3",
            "https://9aifinalteam4.blob.core.windows.net/audio-files/leveltestQ3.mp3",
            "https://9aifinalteam4.blob.core.windows.net/audio-files/leveltestQ4.mp3",
            "https://9aifinalteam4.blob.core.windows.net/audio-files/leveltestQ5.mp3",
            "https://9aifinalteam4.blob.core.windows.net/audio-files/leveltestQ6.mp3",
            "https://9aifinalteam4.blob.core.windows.net/audio-files/leveltestQ7.mp3",
            "https://9aifinalteam4.blob.core.windows.net/audio-files/leveltestQ8.mp3"
        ]
        
        next_question_audio_url = None
        next_question_text = None
        
        if (0 <= question_index < 8) and not is_finishing:
            next_question_text = questions[question_index] 
            next_question_audio_url = pregenerated_audio_urls[question_index]
            
        return {
            "user_recognized_text": user_text,
            "pronunciation_evaluations": pronunciation_evaluations,
            "next_question_text": next_question_text,
            "next_question_audio_url": next_question_audio_url
        }

    async def evaluate_holistic_cefr_level(self, user_answers: list) -> dict:
        total_acc, total_flu, valid_audio_count = 0, 0, 0
        for ans in user_answers:
            if ans.get("accuracy") is not None and ans.get("fluency") is not None:
                total_acc += ans["accuracy"]
                total_flu += ans["fluency"]
                valid_audio_count += 1
                
        avg_pronunciation = 0
        if valid_audio_count > 0:
            avg_pronunciation = (total_acc + total_flu) / (2 * valid_audio_count)

        transcript_str = ""
        for ans in user_answers:
            transcript_str += f"Q{ans.get('question_index')}: {ans.get('text', '(No answer)')}\n"

        system_prompt = f"""
        You are an expert CEFR English evaluator.
        You will be provided with a user's answers to the level test questions.
        Note: The user may have quit early, so there may be fewer than 8 questions. Evaluate based ONLY on the provided transcripts.
        
        [Evaluation Criteria]
        - A1/A2: Single words or simple sentences. Struggles with basic past tense.
        - B1: Connects sentences with basic conjunctions. Good use of past tense for experiences.
        - B2: Clear reasoning, uses 'because/so/if' properly. Uses conditionals.
        - C1/C2: Abstract arguments, complex grammar (although, whereas), diverse vocabulary.
        
        [Detailed Metrics & Strict Grading Rubric (Score out of 100)]
        Apply the following strict rubric for fairness and consistency.
        
        1. expression_score:
           - 90-100: Near-native, highly natural phrasing, uses idioms/phrasal verbs well.
           - 70-89: Clear meaning, but some awkward phrasing or Konglish (direct translations).
           - 50-69: Noticeably unnatural, causing slight confusion for a native speaker.
           - 0-49: Literal translations from native language making it hard to understand.
           
        2. grammar_score:
           - 90-100: Flawless grammar or at most 1 minor slip. Uses complex structures well.
           - 70-89: Accurate simple sentences, but occasional errors in tense/agreement in longer sentences.
           - 50-69: Frequent basic grammar errors that often disrupt sentence structure.
           - 0-49: Severe grammar breakdown making basic subject-verb relationships unclear.
           
        3. task_completion_score: (Grade based ONLY on the questions the user actually answered)
           - 90-100: Understands perfectly, provides detailed answers with reasons/examples.
           - 70-89: Answers the question but lacks elaboration; somewhat brief.
           - 50-69: Off-topic answers or extremely short one-word/phrase responses.
           - 0-49: Fails to understand the question, completely off-topic, or says "I don't know".
           
        4. vocabulary_score:
           - 90-100: Rich, advanced vocabulary (C1/C2) and varied adjectives/adverbs.
           - 70-89: Standard everyday words (B1/B2), some repetition of words.
           - 50-69: Highly limited, repetitive, basic words (A1/A2) like 'good', 'bad', 'very'.
           - 0-49: Lacks the vocabulary to express basic thoughts in English.
        
        [User Transcripts]
        {transcript_str}
        
        [Average Fluency Score from Speech Engine (0-100)]
        {avg_pronunciation:.1f}
        
        Based on the transcripts and the provided fluency score, evaluate the overall English level and detailed metrics.
        The 'fluency_score' MUST be exactly {int(avg_pronunciation)}.
        The 'test_score' should be the overall average of the 5 detailed metrics.
        DO NOT provide any detailed feedback explanations. ONLY return a JSON object.
        
        [OUTPUT FORMAT]
        {{
            "assigned_level": "A1" | "A2" | "B1" | "B2" | "C1" | "C2",
            "test_score": integer (0 to 100),
            "fluency_score": {int(avg_pronunciation)},
            "expression_score": integer (0 to 100),
            "grammar_score": integer (0 to 100),
            "task_completion_score": integer (0 to 100),
            "vocabulary_score": integer (0 to 100)
        }}
        
        CRITICAL RULES FOR SCORING:
        1. DO NOT give the exact same score for all 4 metrics. Evaluate each independently.
        2. DO NOT just pick boundary numbers like 70, 89, 90. Pick a precise number within the range (e.g., 73, 84, 91) based on the exact quality of the transcript.
        """

        try:
            response = await self.call_llm_with_retry(
                self.llm_client, 
                model="gpt-4o-mini", 
                messages=[{"role": "system", "content": system_prompt}],
                response_format={"type": "json_object"} 
            )
            raw_content = response.choices[0].message.content
            clean_str = raw_content.replace("```json", "").replace("```", "").strip()
            parsed_data = json.loads(clean_str)
            
            total_sum = (
                parsed_data.get("fluency_score", 0) +
                parsed_data.get("expression_score", 0) +
                parsed_data.get("grammar_score", 0) +
                parsed_data.get("task_completion_score", 0) +
                parsed_data.get("vocabulary_score", 0)
            )
            parsed_data["test_score"] = int(round(total_sum / 5.0))
            
            return parsed_data
        except Exception as e:
            print(f"❌ [Level Test Eval Error]: {e}")
            return {
                "assigned_level": "A1", "test_score": 0, "fluency_score": 0, 
                "expression_score": 0, "grammar_score": 0, "task_completion_score": 0, "vocabulary_score": 0
            }
