import json
import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import azure.cognitiveservices.speech as speechsdk

# Render 대시보드에서 환경 변수로 설정할 값들 (코드에 키를 직접 적지 마세요!)
SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "")

app = FastAPI(title="PronCoach API")

# 앱(클라이언트)에서 접근할 수 있게 열어두는 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 연습 문장 (난이도 1~5) ----------
SENTENCES = {
    1: [
        "I like cats.",
        "She has a dog.",
        "It is sunny today.",
    ],
    2: [
        "I want a cup of coffee.",
        "He goes to school by bus.",
        "We watch movies on weekends.",
    ],
    3: [
        "Could you tell me where the station is?",
        "I have been studying English for five years.",
        "The weather was much colder than we expected.",
    ],
    4: [
        "Would you mind if I borrowed this book for a few days?",
        "The project was delayed because of unexpected technical issues.",
        "She suggested that we should leave earlier to avoid traffic.",
    ],
    5: [
        "Had I known about the meeting, I would have prepared more thoroughly.",
        "The proposal, which was submitted last week, has yet to be reviewed.",
        "Despite the challenges we faced, the team managed to deliver on time.",
    ],
}


@app.get("/")
def health():
    return {"status": "ok", "service": "proncoach"}


@app.get("/api/v1/sentences")
def get_sentences(level: int = 1):
    """연습 문장 목록. 난이도 1~5 중 하나를 ?level=3 처럼 보내면 됩니다."""
    level = level if level in SENTENCES else 1
    return {"level": level, "sentences": SENTENCES[level]}


@app.post("/api/v1/assessment")
async def assess(
    sentence: str = Form(...),          # 사용자가 읽은 문장 (영어 원문)
    attempt: int = Form(1),             # 1차/2차 시도 구분용
    audio: UploadFile = File(...),      # 앱에서 보내는 녹음 파일 (wav, 16kHz)
):
    if not SPEECH_KEY or not SPEECH_REGION:
        raise HTTPException(500, "AZURE_SPEECH_KEY / AZURE_SPEECH_REGION 환경 변수가 Render에 설정되지 않았습니다.")

    # 업로드된 녹음을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        wav_path = tmp.name

    try:
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        pron_config = speechsdk.PronunciationAssessmentConfig(
            reference_text=sentence,
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Word,
            enable_miscue=True,  # 생략(Omission)/끼워넣기(Insertion)까지 잡아냄
        )

        audio_config = speechsdk.AudioConfig(filename=wav_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        pron_config.apply_to(recognizer)

        result = recognizer.recognize_once()
        if result.reason != speechsdk.ResultReason.RecognizedSpeech:
            raise HTTPException(400, "음성을 인식하지 못했습니다. 조금 더 크게, 문장 그대로 읽어주세요.")

        # Azure가 보내준 분석 결과(JSON) 파싱
        payload = json.loads(result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult))
        nbest = payload["NBest"][0]
        overall = nbest["PronunciationAssessment"]

        # 단어별 결과 정리 — 앱의 워드 매트릭스(취소선/강세 표시)에 쓰는 데이터
        words = []
        for w in nbest.get("Words", []):
            pa = w.get("PronunciationAssessment", {})
            words.append({
                "word": w["Word"],
                "accuracy": round(pa.get("AccuracyScore", 0), 1),
                "errorType": w.get("ErrorType"),       # None / Omission / Insertion / Mispronunciation
                "offsetMs": round(w.get("Offset", 0) / 10000),    # 100ns 단위 -> ms
                "durationMs": round(w.get("Duration", 0) / 10000),
            })

        # 엇박(리듬): 단어 시작 시간 간격 목록
        onsets = [w["offsetMs"] for w in words]
        beats = [b - a for a, b in zip(onsets, onsets[1:])] if len(onsets) > 1 else []

        return {
            "attempt": attempt,
            "scores": {
                "pronunciation": round(overall.get("PronScore", 0), 1),
                "accuracy": round(overall.get("AccuracyScore", 0), 1),
                "fluency": round(overall.get("FluencyScore", 0), 1),
                "prosody": round(overall.get("ProsodyScore", 0), 1),
                "completeness": round(overall.get("CompletenessScore", 0), 1),
            },
            "words": words,
            "beatIntervalsMs": beats,
        }
    finally:
        os.unlink(wav_path)  # 임시 파일 삭제
