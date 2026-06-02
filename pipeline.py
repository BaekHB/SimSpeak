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
        오디오 URL을 다운로드하여 Whisper(원본 한영 추출)와 Azure Speech(발음 평가)를 진행하는 투트랙 코어 엔진
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
            # ★ 트랙 1: Azure OpenAI Whisper (원본 한영 혼용 텍스트 전체 추출)
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
                print(f"[Sync Track 1] Whisper Master Text Extracted: '{whisper_text}'")
            except Exception as e:
                whisper_text = f"[Whisper 에러: {e}]"
                print(f"Warning: Whisper STT failed: {e}")

            # --------------------------------------------------
            # ★ 트랙 2: Azure Speech (한글 단어 오인식 배제 후 순수 영어만 채점)
            # --------------------------------------------------
            detailed_score = error_response
            try:
                speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
                audio_config = speechsdk.AudioConfig(filename=temp_eval_path)
                
                # ✨ [유일한 백엔드 보조 기능]: 발음 채점기가 미쳐 날뛰지 않게 한글 글자만 일회성 정제
                pure_english_reference = "".join(
                    char for char in whisper_text 
                    if not ('가' <= char <= '힣' or 'ㄱ' <= char <= 'ㅣ')
                ).strip()
                pure_english_reference = " ".join(pure_english_reference.split())
                
                pronunciation_config = speechsdk.PronunciationAssessmentConfig(
                    reference_text=pure_english_reference,
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
            if 'speech_recognizer' in locals():
                del speech_recognizer
            if 'audio_config' in locals():
                del audio_config
                
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
            speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
            audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_filename)
            
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            
            ssml_string = self.make_ssml(character_id, text_content)
            tts_result = synthesizer.speak_ssml_async(ssml_string).get()
            
            if tts_result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                print(f"Warning: TTS synthesis failed (Reason: {tts_result.reason})")
                return "https://9aifinalteam4.blob.core.windows.net/audio-files/reply_8e9e195b.mp3"
                
            blob_service_client = BlobServiceClient.from_connection_string(self.storage_connection)
            container_name = "audio-files"
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=temp_filename)
            
            with open(temp_filename, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
                
            print(f"Success: TTS generated and uploaded: {blob_client.url}")
            return blob_client.url

        except Exception as e:
            print(f"Warning: TTS pipeline error: {e}")
            return "https://9aifinalteam4.blob.core.windows.net/audio-files/reply_8e9e195b.mp3"
            
        finally:
            if 'synthesizer' in locals():
                del synthesizer
            if 'audio_config' in locals():
                del audio_config
                
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
        STT 입력 ➡️ 과거 기억 조회 ➡️ LLM 추론 ➡️ TTS 출력 생성까지 전체 AI 파이프라인 조율
        """
        char_id = character_id.lower()
        print(f"[User: {user_id}] -> [{char_id}] Initial input: {user_text}")

        # 1. 과거 기억 조회 및 세션 초기화
        if user_id not in session_db:
            session_db[user_id] = {}
        if char_id not in session_db[user_id]:
            session_db[user_id][char_id] = {"history": [], "current_affinity": 30, "summary_context": ""}
        user_data = session_db[user_id][char_id]
        
        if "summary_context" not in user_data:
            user_data["summary_context"] = ""
            
        current_summary = user_data["summary_context"]

        real_pronunciation_score = None
        
        # 2. STT (Whisper) + 발음 평가 (Azure Speech) 투트랙 작동
        if user_audio_url:
            print(f"Audio URL detected -> Starting dual-track: {user_audio_url}")
            extracted_text, real_pronunciation_score = self.evaluate_dual_track(user_audio_url)
            
            # Whisper 분석 성공 시 유저 텍스트 덮어쓰기
            if extracted_text and not extracted_text.startswith("[Whisper"):
                user_text = extracted_text
        else:
            print("Text-only mode -> Skipping pronunciation assessment.")

        # 3. LLM 추론 (Azure OpenAI GPT-4o)
        # 파이프라인 단의 어떠한 인위적인 제약 조건문도 추가하지 않고, 
        # 오직 정우님이 prompts/chloe.txt 파일에 명시해 둔 룰북 그대로 GPT에게 상속합니다.
        base_prompt = self.get_character_prompt(char_id)
        
        summary_prefix = f"[PAST CONVERSATION SUMMARY]\n{current_summary}\n\n" if current_summary else ""
        system_prompt = summary_prefix + base_prompt + f"\n\n[LIVE STATUS]\n- Current Affinity: {user_data['current_affinity']}/100\n- Input Mode: is_video_call={is_video_call}"
        
        messages = [{"role": "system", "content": system_prompt}]
        
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

        # Azure OpenAI 공통 클라이언트 선언
        ai_client = AzureOpenAI(
            azure_endpoint=self.openai_endpoint,
            api_key=self.openai_key,
            api_version="2024-02-15-preview"
        )

        # [토큰 절약 엔진 가동] 10턴 초과 시 압축 로그 작동
        if len(refined_history) > 10:
            overflow_turns = refined_history[:-10]
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
            except Exception as e:
                print(f"[Warning] Summary engine temporary error: {e}")

        # 문맥 동기화 컨텍스트 주입
        messages.extend(refined_history[-10:])
        messages.append({"role": "user", "content": user_text})

        try:
            # 프롬프트의 JSON Output Format Specification 정의서 그대로 GPT가 순수하게 파싱하도록 유도
            response = ai_client.chat.completions.create(
                model=self.openai_deployment,
                response_format={"type": "json_object"},
                messages=messages
            )
            
            ai_response_text = response.choices[0].message.content
            ai_result = json.loads(ai_response_text)
            
            # 4. 과거 기억(세션) 업데이트 및 호감도 반영
            # 프롬프트 규칙에 의해 완벽히 조율되어 나온 affinity_delta 값을 그대로 승계하여 합산합니다. (파이프라인 침해 차단)
            affinity_delta = ai_result.get("affinity_delta", 0)
            user_data["history"].append({"role": "user", "content": user_text})
            user_data["history"].append({"role": "assistant", "content": ai_response_text})
            user_data["current_affinity"] = max(0, min(100, user_data["current_affinity"] + affinity_delta))

            # 5. TTS 출력 생성 (답변 대사를 음성 합성하여 업로드)
            character_reply = ai_result.get("text_content") or ai_result.get("content") or ""
            tts_audio_url = self.generate_tts(char_id, character_reply)
            
            # 6. JSON 응답 최종 데이터 주머니 패키징
            ai_result["audio_url"] = tts_audio_url
            ai_result["current_total_affinity"] = user_data["current_affinity"]
            ai_result["user_recognized_text"] = user_text
            
            # 발음 평가 리포트 구조 보존
            if "system_evaluation" not in ai_result:
                ai_result["system_evaluation"] = {}
            ai_result["system_evaluation"]["pronunciation_score"] = real_pronunciation_score
            
            return ai_result

        except Exception as e:
            print(f"Warning: GPT inference and pipeline error: {e}")
            raise e
