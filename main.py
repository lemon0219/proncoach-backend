import json
import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import azure.cognitiveservices.speech as speechsdk

# Render 대시보드의 환경 변수 (코드에 키를 직접 적지 마세요)
SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "")

app = FastAPI(title="PronCoach API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def health():
    return {"status": "ok", "service": "proncoach"}


@app.post("/api/v1/assess")
async def assess(attempt: int = Form(1), audio: UploadFile = File(...)):
    """사용자 녹음을 받아 -> 1) 무슨 말인지 스스로 인식 -> 2) 인식된 문장 기준 발음 분석."""
    if not SPEECH_KEY or not SPEECH_REGION:
        raise HTTPException(500, "AZURE_SPEECH_KEY / AZURE_SPEECH_REGION 환경 변수가 설정되지 않았습니다.")

    wav_bytes = await audio.read()
    print(f"[진단] 받은 녹음: {len(wav_bytes)} bytes, 약 {max(len(wav_bytes)-44, 0)/32000:.1f}초", flush=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_bytes)
        wav_path = tmp.name

    try:
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)

        # --- 1) 음성 인식 (단어 단위 시간 정보 포함) ---
        speech_config.request_word_level_timestamps()
        audio_config = speechsdk.AudioConfig(filename=wav_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        stt_result = recognizer.recognize_once()

        # --- 디버깅: 실패 이유를 로그에 남김 ---
        print(f"[진단] STT 결과: reason={stt_result.reason}", flush=True)
        if stt_result.cancellation_details is not None:
            print(f"[진단] STT 취소 상세: {stt_result.cancellation_details.reason} / {stt_result.cancellation_details.error_details}", flush=True)

        if stt_result.reason != speechsdk.ResultReason.RecognizedSpeech:
            raise HTTPException(400, f"음성을 인식하지 못했습니다 (이유: {stt_result.reason}). 녹음이 1초 이상인지, 충분히 크게 말했는지 확인해주세요.")

        stt_json = json.loads(stt_result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult))
        recognized = stt_json["NBest"][0].get("DisplayText", "")

        # --- 2) 인식된 문장을 기준으로 발음 분석 ---
        scores, words = {}, []
        try:
            pron_config = speechsdk.PronunciationAssessmentConfig(
                reference_text=recognized,
                grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                granularity=speechsdk.PronunciationAssessmentGranularity.Word,
                enable_miscue=False,
            )
            speech_config2 = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
            audio_config2 = speechsdk.AudioConfig(filename=wav_path)
            recognizer2 = speechsdk.SpeechRecognizer(speech_config=speech_config2, audio_config=audio_config2)
            pron_config.apply_to(recognizer2)
            pa_result = recognizer2.recognize_once()
            pa_json = json.loads(pa_result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult))
            nbest = pa_json["NBest"][0]
            overall = nbest["PronunciationAssessment"]
            scores = {
                "pronunciation": round(overall.get("PronScore", 0), 1),
                "accuracy": round(overall.get("AccuracyScore", 0), 1),
                "fluency": round(overall.get("FluencyScore", 0), 1),
                "prosody": round(overall.get("ProsodyScore", 0), 1),
                "completeness": round(overall.get("CompletenessScore", 0), 1),
            }
            for w in nbest.get("Words", []):
                pa = w.get("PronunciationAssessment", {})
                words.append({
                    "word": w["Word"],
                    "accuracy": round(pa.get("AccuracyScore", 0), 1),
                    "errorType": w.get("ErrorType"),
                    "offsetMs": round(w.get("Offset", 0) / 10000),
                    "durationMs": round(w.get("Duration", 0) / 10000),
                })
        except Exception:
            # 발음 분석이 실패해도 인식 결과(단어+시간)는 반환
            for w in stt_json["NBest"][0].get("Words", []):
                words.append({
                    "word": w["Word"], "accuracy": None, "errorType": None,
                    "offsetMs": round(w.get("Offset", 0) / 10000),
                    "durationMs": round(w.get("Duration", 0) / 10000),
                })

        onsets = [w["offsetMs"] for w in words]
        beats = [b - a for a, b in zip(onsets, onsets[1:])] if len(onsets) > 1 else []

        return {
            "attempt": attempt,
            "recognizedText": recognized,
            "scores": scores,
            "words": words,
            "beatIntervalsMs": beats,
        }
    finally:
        os.unlink(wav_path)
