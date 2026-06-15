import asyncio
import httpx

async def run_test():
    url = "http://127.0.0.1:8000/api/v1/chat/level_test"
    user_id = "2"
    char_id = "chloe"
    sample_audio_url = "https://9aifinalteam4.blob.core.windows.net/audio-files/3.wav"

    accumulated_answers = []

    print("=========================================================")
    print("🚀 [대화형 레벨 테스트 시뮬레이터]")
    print("=========================================================")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(1, 9):
            print(f"\n================ [ Q{i} ] ================")
            print("선택지를 입력하세요:")
            print("1. 직접 텍스트로 영어 답변 타이핑하기")
            print(f"2. 하드코딩된 오디오 전송하기 ({sample_audio_url})")
            print("3. 지금 바로 그만두고 채점 결과 보기 (Quit)")
            
            choice = input("입력 (1/2/3): ").strip()
            
            payload = {
                "user_id": user_id,
                "character_id": char_id,
                "current_question_index": i,
                "accumulated_answers": accumulated_answers,
                "is_quit": False
            }
            
            if choice == "3":
                print("\n🛑 중간 종료 요청을 보냅니다. 즉각 채점이 시작됩니다...")
                payload["is_quit"] = True
                payload["user_text"] = "I quit."
            elif choice == "1":
                user_text = input("\n💬 영어 답변을 타이핑하세요: ")
                payload["user_text"] = user_text
            else:
                audio_url_input = input(f"\n▶️ 오디오 URL을 붙여넣으세요 (엔터 치면 기본값 3.wav 전송): ").strip()
                if not audio_url_input:
                    audio_url_input = sample_audio_url
                payload["user_audio_url"] = audio_url_input
            
            response = await client.post(url, json=payload)
            data = response.json()
            
            extracted_text = data.get('user_recognized_text', '')
            pronunciation = data.get('pronunciation_evaluations', {})
            
            print(f"\n✅ [Q{i} 백엔드 인식 결과]: {extracted_text}")
            if pronunciation:
                print(f"📊 [발음 점수]: 정확도 {pronunciation.get('accuracy')}점 / 유창성 {pronunciation.get('fluency')}점")
            
            if data.get('is_finished'):
                print(f"\n🛑 [채점 완료됨] -> 즉시 동기식 채점 결과가 도착했습니다!")
                final_res = data.get("final_result", {})
                print(f"🏆 최종 등급: {final_res.get('assigned_level')}")
                print(f"💯 종합 점수: {final_res.get('test_score')}점")
                print(f" - 유창성: {final_res.get('fluency_score')}점")
                print(f" - 표현력: {final_res.get('expression_score')}점")
                print(f" - 문법 정확도: {final_res.get('grammar_score')}점")
                print(f" - 과제 수행도: {final_res.get('task_completion_score')}점")
                print(f" - 어휘력: {final_res.get('vocabulary_score')}점")
                break
                
            print(f"💬 [다음 질문]: {data.get('next_question_text')}")
            print(f"🎧 [TTS 주소]: {data.get('next_question_audio_url')}")
            
            accumulated_answers.append({
                "question_index": i,
                "text": extracted_text,
                "accuracy": pronunciation.get('accuracy', 0) if pronunciation else 0,
                "fluency": pronunciation.get('fluency', 0) if pronunciation else 0
            })

if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        print("\n종료되었습니다.")
