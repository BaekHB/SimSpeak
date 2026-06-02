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
        """
        캐릭터 ID에 따라 목소리, 속도, 높낮이가 튜닝된 SSML 양식을 생성합니다.
        """
        char_id = character_id.lower()
        
        # 기본값 설정
        voice_name = "en-US-ChristopherNeural"
        rate = "0%"
        pitch = "0%"
        
        if char_id == "liam":          # 리암 (영국 어른 남자)
            voice_name = "en-GB-RyanNeural"
            rate = "-10%"
            pitch = "-5%"
        elif char_id == "chloe":       # 클로이 (미국 왈가닥 요정)
            voice_name = "en-US-AriaNeural"
            rate = "+10%"
            pitch = "+5%"
        elif char_id == "ian":         # 이안 (미국 서부 서핑 강사)
            voice_name = "en-US-ChristopherNeural"
            rate = "0%"
            pitch = "0%"
        elif char_id == "june":        # 준 (호주 츤데레 남선배)
            voice_name = "en-AU-WilliamNeural"
            rate = "0%"
            pitch = "0%"
        elif char_id == "sienna":      # 시에나 (영국 러블리 파티시에)
            voice_name = "en-GB-SoniaNeural"
            rate = "0%"
            pitch = "0%"
        elif char_id == "yoon":        # 윤 (호주 츤데레 여선배)
            voice_name = "en-AU-NatashaNeural"
            rate = "0%"
            pitch = "0%"
            
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

    def evaluate_dual_track(self, audio_url: str) -> tuple[str, dict]:
        """
        오디오 URL을 다운로드하여 Whisper(한영 추출)와 Azure Speech(발음 평가)를 동시에 진행하는 투트랙 코어 엔진
        """
        whisper_text = ""
        error_response = {
            "accuracy": 0, "fluency": 0, "completeness": 0, "prosody": 0, "word_details": []
        }

        if not audio_url:
            return whisper_text, error_response

        # 임시 저장 파일명 설정 (UUID 적용하여 충돌 방지)
        temp_eval_path = f"temp_eval_{uuid.uuid4().hex[:8]}.wav"
        
        try:
            # 1. 클라우드에서 오디오 바이너리 다운로드
            response = requests.get(audio_url, timeout=15)
            if response.status_code != 200:
                print(f"Warning: Audio download failed: {audio_url}")
                return whisper_text, error_response
                
            with open(temp_eval_path, "wb") as f:
                f.write(response.content)

            # --------------------------------------------------
            # ★ 트랙 1: Azure OpenAI Whisper (한영 혼용 텍스트 추출)
            # --------------------------------------------------
            try:
                whisper_client = AzureOpenAI(
                    azure_endpoint=self.whisper_endpoint,
                    api_key=self.whisper_key,
                    api_version="2024-02-15-preview"
                )
                with open(temp_eval_path, "rb") as audio_file:
                    whisper_result = whisper_client.audio.transcriptions.create(
                        file=audio_file,
                        model=self.whisper_deployment,
                        prompt="Hello! 안녕하세요."
                    )
                whisper_text = whisper_result.text
            except Exception as e:
                whisper_text = f"[Whisper 에러: {e}]"
                print(f"Warning: Whisper STT failed: {e}")

            # --------------------------------------------------
            # ★ 트랙 2: Azure Speech (영어 발음 정밀 평가)
            # --------------------------------------------------
            detailed_score = error_response
            try:
                speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
                audio_config = speechsdk.AudioConfig(filename=temp_eval_path)
                
                pronunciation_config = speechsdk.PronunciationAssessmentConfig(
                    reference_text="",
                    grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                    granularity=speechsdk.PronunciationAssessmentGranularity.Word
                )
                pronunciation_config.enable_prosody_assessment()
                
                speech_recognizer = speechsdk.SpeechRecognizer(
                    speech_config=speech_config, language="en-US", audio_config=audio_config
                )
                pronunciation_config.apply_to(speech_recognizer)
                result = speech_recognizer.recognize_once_async().get()
                
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    assessment_result = speechsdk.PronunciationAssessmentResult(result)
                    word_details_list = []
                    for word in assessment_result.words:
                        error_type = word.error_type if word.error_type != "None" else None
                        word_details_list.append({
                            "word": word.word.strip(),
                            "accuracy": int(word.accuracy_score),
                            "error_type": error_type
                        })
                    detailed_score = {
                        "accuracy": int(assessment_result.accuracy_score),
                        "fluency": int(assessment_result.fluency_score),
                        "completeness": int(assessment_result.completeness_score),
                        "prosody": int(assessment_result.prosody_score),
                        "word_details": word_details_list
                    }
                else:
                    print(f"Warning: Pronunciation assessment failed or no match (Reason: {result.reason})")
            except Exception as e:
                print(f"Warning: Azure Speech Pronunciation Assessment failed: {e}")
                
            return whisper_text, detailed_score

        except Exception as e:
            print(f"Warning: Core scoring engine error: {e}")
            return whisper_text, error_response
            
        finally:
            # 자원 명시적 해제하여 파일 잠금 해제
            if 'speech_recognizer' in locals():
                del speech_recognizer
            if 'audio_config' in locals():
                del audio_config
                
            # 에러 발생 여부와 상관없이 무조건 임시 파일 삭제하여 점유 버그 100% 차단
            if os.path.exists(temp_eval_path):
                try:
                    os.remove(temp_eval_path)
                except Exception as e:
                    print(f"Warning: Temp file deletion failed: {e}")

    def generate_tts(self, character_id: str, text_content: str) -> str:
        """
        캐릭터의 답변을 TTS 오디오로 생성하여 Azure Blob Storage에 업로드하고 링크를 반환합니다.
        """
        temp_filename = f"reply_{uuid.uuid4().hex[:8]}.mp3"
        
        try:
            # 1. Azure Speech SDK TTS 설정
            speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
            audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_filename)
            
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            
            # 2. SSML 변환 후 음성 합성
            ssml_string = self.make_ssml(character_id, text_content)
            tts_result = synthesizer.speak_ssml_async(ssml_string).get()
            
            if tts_result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                print(f"Warning: TTS synthesis failed (Reason: {tts_result.reason})")
                return "https://9aifinalteam4.blob.core.windows.net/audio-files/reply_8e9e195b.mp3" # 실패 시 기존 기본값 반환
                
            # 3. Azure Blob Storage 업로드
            blob_service_client = BlobServiceClient.from_connection_string(self.storage_connection)
            container_name = "audio-files"
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=temp_filename)
            
            with open(temp_filename, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
                
            print(f"Success: TTS generated and uploaded: {blob_client.url}")
            return blob_client.url

        except Exception as e:
            print(f"Warning: TTS pipeline error: {e}")
            return "https://9aifinalteam4.blob.core.windows.net/audio-files/reply_8e9e195b.mp3" # 에러 시 기본값 대체
            
        finally:
            # 자원 명시적 해제하여 파일 잠금 해제
            if 'synthesizer' in locals():
                del synthesizer
            if 'audio_config' in locals():
                del audio_config
                
            # 로컬 임시 오디오 파일 삭제
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception as e:
                    print(f"Warning: Temp audio file deletion failed: {e}")

    def get_character_prompt(self, character_id: str) -> str:
        """
        캐릭터 ID에 따라 prompts 폴더 내 텍스트 설정을 로드합니다.
        """
        file_path = f"prompts/{character_id.lower()}.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def run(self, session_db: dict, user_id: str, character_id: str, user_text: str, is_video_call: bool, user_audio_url: str = None) -> dict:
        """
        STT 입력 ➡️ 과거 기억 조회 ➡️ LLM 추론 ➡️ TTS 출력 생성까지 전체 AI 파이프라인을 조율하여 최종 JSON 데이터를 완성합니다.
        """
        char_id = character_id.lower()
        print(f"[User: {user_id}] -> [{char_id}] Initial input: {user_text}")

        # 1. 과거 기억 조회 및 세션 초기화
        if user_id not in session_db:
            session_db[user_id] = {}
        if char_id not in session_db[user_id]:
            session_db[user_id][char_id] = {"history": [], "current_affinity": 30, "summary_context": ""}
        user_data = session_db[user_id][char_id]
        
        # summary_context 키 누락 방지 방어 코드
        if "summary_context" not in user_data:
            user_data["summary_context"] = ""
            
        current_summary = user_data["summary_context"]

        real_pronunciation_score = None
        penalty_message = ""
        
        # 2. STT (Whisper) + 발음 평가 (Azure Speech) 투트랙 작동
        if user_audio_url:
            print(f"Audio URL detected -> Starting dual-track: {user_audio_url}")
            extracted_text, real_pronunciation_score = self.evaluate_dual_track(user_audio_url)
            
            # Whisper 분석 성공 시 유저 텍스트 덮어쓰기
            if extracted_text and not extracted_text.startswith("[Whisper"):
                user_text = extracted_text
                
            # =========================================================
            # ✨ 💡 [음성 모드 한국어 남용 방어 레이어 고도화]
            # =========================================================
            # Whisper가 음성에서 복원해 낸 텍스트 문장에 한국어 자음/모음/완성형 글자가 들어있는지 추적
            has_korean_in_audio = any('가' <= char <= '힣' or 'ㄱ' <= char <= 'ㅣ' for char in user_text)
            score_val = real_pronunciation_score.get("accuracy", 100)
            
            # [수정 포인트]: 영어 종합 발음 정확도가 50점 미만이거나, 음성 안에 한국어가 단 한 글자라도 포착된 경우 패널티 작동!
            if score_val < 50 or has_korean_in_audio:
                print(f"🚨 [Audio Guardrail Activated] Korean in Audio: {has_korean_in_audio} | Score: {score_val}")
                penalty_message = "\n[SYSTEM OVERRIDE MESSAGE: 방금 유저가 보낸 음성에서 한국어가 감지되었거나 영어 발음이 불량합니다. 유저의 한국어 사용/나쁜 발음을 반드시 따끔하게 지적하고, 이번 턴의 affinity_delta를 무조건 -3으로 고정하여 출력하십시오. 예외는 없습니다.]"
            # =========================================================
        else:
            print("Text-only mode -> Skipping pronunciation assessment.")

        # 3. LLM 추론 (Azure OpenAI GPT-4o)
        base_prompt = self.get_character_prompt(char_id)
        
        # [요약 주입] 기존 압축 기억이 있다면 시스템 프롬프트 최상단에 주입
        summary_prefix = f"[PAST CONVERSATION SUMMARY]\n{current_summary}\n\n" if current_summary else ""
        system_prompt = summary_prefix + base_prompt + f"\n\n[LIVE STATUS]\n- Current Affinity: {user_data['current_affinity']}/100" + penalty_message
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # 과거 history 배열에서 JSON 구조를 벗겨내고 순수한 대화 대사(content)만 추출하여 전달
        refined_history = []
        for turn in user_data["history"]:
            role = turn.get("role")
            content_raw = turn.get("content", "")
            
            if role == "user":
                refined_history.append({"role": "user", "content": content_raw})
            elif role == "assistant":
                try:
                    # DB에 적재되었던 JSON포맷 응답 문자열에서 순수 캐릭터 대사만 파싱
                    data = json.loads(content_raw)
                    pure_text = data.get("text_content") or data.get("content") or content_raw
                    refined_history.append({"role": "assistant", "content": pure_text})
                except Exception:
                    # 파싱 실패나 일반 문자열일 경우 안전장치 예외 처리
                    refined_history.append({"role": "assistant", "content": content_raw})

        # Azure OpenAI 공통 클라이언트 선언
        ai_client = AzureOpenAI(
            azure_endpoint=self.openai_endpoint,
            api_key=self.openai_key,
            api_version="2024-02-15-preview"
        )

        # [토큰 절약 엔진 가동] 10턴 초과 시 잘려 나가는 앞부분 대화 압축하기
        if len(refined_history) > 10:
            overflow_turns = refined_history[:-10] # 윈도우 밖으로 버려질 대화 조각들
            print(f"[Token Saving Engine] Compacting {len(overflow_turns)} overflow turns into long-term summary.")
            
            overflow_text = ""
            for turn in overflow_turns:
                overflow_text += f"{turn['role']}: {turn['content']}\n"
                
            summary_command = [
                {"role": "system", "content": "너는 기억 파수꾼이야. 기존 [누적 요약본]에 새로 잊혀지려는 [대화 조각]의 핵심 사건이나 유저 정보만 결합해서 한 문장의 한국어로 지속 업데이트해 줘. 대화 로그 형식은 금지한다."},
                {"role": "user", "content": f"[기존 누적 요약본]\n{current_summary}\n\n[새 대화 조각]\n{overflow_text}"}
            ]
            
            try:
                summary_response = ai_client.chat.completions.create(
                    model=self.openai_deployment,
                    messages=summary_command,
                    max_tokens=150
                )
                current_summary = summary_response.choices[0].message.content.strip()
                user_data["summary_context"] = current_summary
                print(f"[Token Saving Engine] Compaction complete. Long-term summary: {current_summary}")
            except Exception as e:
                print(f"[Warning] Summary engine temporary error (retaining past summary): {e}")

        # 문맥이 꼬이지 않도록 최신 10개의 정제된 턴만 슬라이딩 윈도우로 컨텍스트 주입
        messages.extend(refined_history[-10:])
        messages.append({"role": "user", "content": user_text})

        try:
            response = ai_client.chat.completions.create(
                model=self.openai_deployment,
                response_format={"type": "json_object"},
                messages=messages
            )
            raw_usage_log = json.loads(response.model_dump_json())
            ai_response_text = response.choices[0].message.content
            ai_result = json.loads(ai_response_text)
            ai_result["raw_llm_log"] = raw_usage_log
            
            # 4. 과거 기억(세션) 업데이트 및 호감도 계산
            affinity_delta = ai_result.get("affinity_delta", 0)
            user_data["history"].append({"role": "user", "content": user_text})
            user_data["history"].append({"role": "assistant", "content": ai_response_text})
            user_data["current_affinity"] = max(0, min(100, user_data["current_affinity"] + affinity_delta))

            # 5. TTS 출력 생성 (답변 대사를 음성 합성하여 업로드)
            character_reply = ai_result.get("text_content") or ai_result.get("content") or ""
            tts_audio_url = self.generate_tts(char_id, character_reply)
            
            # 6. JSON 응답 최종 데이터 주머니(Response Body) 패키징
            ai_result["audio_url"] = tts_audio_url
            ai_result["current_total_affinity"] = user_data["current_affinity"]
            ai_result["user_recognized_text"] = user_text
            
            if "system_evaluation" not in ai_result:
                ai_result["system_evaluation"] = {}
                
            ai_result["system_evaluation"]["pronunciation_score"] = real_pronunciation_score
            
            return ai_result

        except Exception as e:
            print(f"Warning: GPT inference and pipeline error: {e}")
            raise e