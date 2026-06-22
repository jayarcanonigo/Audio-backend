from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile, threading, os, re, time, uuid, json
from queue import Queue

app = FastAPI(title="Radio Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# CORE STATE
# =======================
model = None
sessions = {}
lock = threading.Lock()

job_queue = Queue(maxsize=1)


# =======================
# MODEL LOAD
# =======================
@app.on_event("startup")
def load_model():
    global model
    model = WhisperModel("small", device="cpu", compute_type="int8")
    threading.Thread(target=worker, daemon=True).start()
    print("Whisper loaded")


# =======================
# UTIL
# =======================
def format_time(sec):
    return f"{int(sec//3600):02}:{int(sec%3600//60):02}:{int(sec%60):02}"


def normalize(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


# =======================
# ADS LOGIC (UNCHANGED)
# =======================
AD_MERGE_GAP = 5

AD_KEYWORDS = [
    "sponsored","brought to you","hatid ng","hatid sa inyo",
    "inihahatid ng","ipinagmamalaki ng","ang programang ito ay hatid",
    "time check","time now","oras natin","oras na"
]


def check_custom_keywords(text, keywords):
    if not keywords:
        return False
    text = text.lower()
    return any(k.lower() in text for k in keywords if k)


def is_advertisement(text):
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    score = sum(2 for x in AD_KEYWORDS if x in text)

    if re.search(r"(09\d{9}|\+639\d{9})", text):
        score += 2
    if re.search(r"(php|peso|pesos)\s*\d+", text):
        score += 2
    if any(x in text for x in ["facebook","fb","www",".com",".ph"]):
        score += 2

    return score >= 2


def merge_advertisement(session, seg):
    t = session["transcript"]

    if not t:
        t.append(seg)
        return

    p = t[-1]
    gap = seg["start"] - p["end"]

    if p["advertisement"] and seg["advertisement"] and gap <= AD_MERGE_GAP:
        p["end"] = seg["end"]
        p["end_time"] = seg["end_time"]
        p["text"] += " " + seg["text"]
        p["text_tokens"] = normalize(p["text"])
        return

    t.append(seg)


# =======================
# LOGS
# =======================
def add_log(sid, msg, start=None, end=None, ad=False):
    if sid not in sessions:
        return

    with lock:
        sessions[sid]["logs"].append({
            "id": time.time_ns(),
            "time": time.strftime("%H:%M:%S"),
            "message": msg,
            "start_time": start,
            "end_time": end,
            "advertisement": ad
        })

        sessions[sid]["logs"] = sessions[sid]["logs"][-100:]


# =======================
# TRANSCRIBE (NO CHUNK)
# =======================
def transcribe_audio(path, sid):
    s = sessions.get(sid)
    if not s:
        return

    try:
        s["progress"].update({
            "status": "transcribing",
            "message": "Transcribing...",
            "started_at": time.time()
        })

        add_log(sid, "Transcription started")

        segments, info = model.transcribe(
            path,
            beam_size=5,
            vad_filter=True,
            task="transcribe"
        )

        add_log(sid, f"Language: {info.language}")

        count = 0

        for seg in segments:

            if s.get("stop"):
                s["progress"].update({"status": "stopped"})
                add_log(sid, "Stopped")
                return

            text = (seg.text or "").strip()
            if not text:
                continue

            count += 1

            start_t = format_time(seg.start)
            end_t = format_time(seg.end)

            ad = is_advertisement(text) or check_custom_keywords(
                text, s.get("keywords", [])
            )

            data = {
                "index": count - 1,
                "start": seg.start,
                "end": seg.end,
                "start_time": start_t,
                "end_time": end_t,
                "text": text,
                "text_tokens": normalize(text),
                "advertisement": ad
            }

            with lock:
                merge_advertisement(s, data)

            if count % 5 == 0:
                s["progress"].update({
                    "current_segment": count,
                    "processed_time": end_t
                })

            add_log(sid, text, start_t, end_t, ad)

        s["progress"].update({
            "status": "completed",
            "total_segments": count
        })

        add_log(sid, f"Completed {count} segments")

    finally:
        try:
            os.remove(path)
        except:
            pass


# =======================
# WORKER (IMPORTANT)
# =======================
def worker():
    while True:
        path, sid = job_queue.get()
        try:
            transcribe_audio(path, sid)
        finally:
            job_queue.task_done()


# =======================
# API
# =======================
@app.post("/upload")
async def upload(file: UploadFile = File(...), keywords: str = Form("")):

    sid = str(uuid.uuid4())

    try:
        kw = [x.strip() for x in json.loads(keywords) if x.strip()]
    except:
        kw = []

    sessions[sid] = {
        "stop": False,
        "progress": {"status": "starting"},
        "logs": [],
        "transcript": [],
        "keywords": kw
    }

    # STREAM FILE (IMPORTANT FIX)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)
        path = tmp.name

    add_log(sid, "File uploaded")

    job_queue.put((path, sid))

    return {"session_id": sid, "status": "started"}


@app.get("/status/{sid}")
def status(sid: str):
    return sessions.get(sid, {}).get("progress", {})


@app.get("/logs/{sid}")
def logs(sid: str):
    return sessions.get(sid, {}).get("logs", [])


@app.get("/transcript/{sid}")
def transcript(sid: str):
    return sessions.get(sid, {}).get("transcript", [])


@app.post("/stop/{sid}")
def stop(sid: str):
    if sid in sessions:
        sessions[sid]["stop"] = True
    return {"status": "stopped"}

@app.post("/reset/{sid}")
def reset(sid: str):

    if sid not in sessions:
        return {"status": "not_found"}

    with lock:
        sessions.pop(sid, None)

    return {"status": "reset", "session_id": sid}
