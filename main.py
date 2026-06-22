from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile, threading, os, re, time, uuid, json

app = FastAPI(title="Radio Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
lock = threading.Lock()
sessions = {}


def create_progress():
    return {
        "status": "idle",
        "message": "",
        "current_segment": 0,
        "processed_time": "00:00:00",
        "total_segments": 0,
        "processing_time_seconds": 0,
        "started_at": None,
        "error": None,
    }


@app.on_event("startup")
def load_model():
    global model
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    print("Whisper loaded")


def format_time(sec):
    return f"{int(sec//3600):02}:{int(sec%3600//60):02}:{int(sec%60):02}"


def normalize(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


AD_MERGE_GAP = 5


AD_KEYWORDS = [
    "sponsored",
    "brought to you",
    "hatid ng",
    "hatid sa inyo",
    "inihahatid ng",
    "ipinagmamalaki ng",
    "ang programang ito ay hatid",
    "time check",
    "time check brought to you by",
    "the time is",
    "time now",
    "oras natin",
    "ang oras ay",
    "oras na",
    "hatid na time check",
    "ang time check ay hatid ng",
    "oras hatid ng",
]


def check_custom_keywords(text, keywords):

    if not keywords:
        return False

    text = text.lower()

    return any(k.strip().lower() in text for k in keywords if k.strip())


def is_advertisement(text):

    text = re.sub(r"[^a-z0-9\s]", "", text.lower())

    text = " ".join(text.split())

    score = sum(2 for x in AD_KEYWORDS if x in text)

    if re.search(r"(09\d{9}|\+639\d{9})", text):
        score += 2

    if re.search(r"(php|peso|pesos)\s*\d+", text):
        score += 2

    if any(x in text for x in ["facebook", "fb", "www", ".com", ".ph"]):
        score += 2

    return score >= 2


def merge_advertisement(session, new_segment):

    transcript = session["transcript"]

    if not transcript:
        transcript.append(new_segment)
        return

    previous = transcript[-1]

    gap = new_segment["start"] - previous["end"]

    if (
        previous["advertisement"]
        and new_segment["advertisement"]
        and gap <= AD_MERGE_GAP
    ):

        previous["end"] = new_segment["end"]

        previous["end_time"] = new_segment["end_time"]

        previous["text"] += " " + new_segment["text"]

        previous["text_tokens"] = normalize(previous["text"])

        return

    transcript.append(new_segment)


def add_log(sid, msg, start=None, end=None, advertisement=False):

    if sid not in sessions:
        return

    item = {
        "id": time.time_ns(),
        "time": time.strftime("%H:%M:%S"),
        "message": msg,
        "start_time": start,
        "end_time": end,
        "advertisement": advertisement,
    }

    with lock:

        sessions[sid]["logs"].append(item)

        if len(sessions[sid]["logs"]) > 500:
            sessions[sid]["logs"] = sessions[sid]["logs"][-500:]


def transcribe_audio(path, sid):

    session = sessions[sid]

    try:

        start = time.time()

        session["progress"].update(
            {
                "status": "transcribing",
                "message": "Transcribing...",
                "started_at": start,
                "error": None,
            }
        )

        add_log(sid, "Transcription started")

        segments, info = model.transcribe(
            path, beam_size=5, vad_filter=True, language=None, task="transcribe"
        )

        add_log(sid, f"Language: {info.language}")

        count = 0

        for seg in segments:

            if session["stop"]:

                session["progress"].update(
                    {"status": "stopped", "message": "Stopped", "error": None}
                )

                add_log(sid, "Processing stopped")

                return

            text = seg.text.strip()

            text = seg.text.strip()

            if not text:
                continue

            count += 1

            start_t = format_time(seg.start)
            end_t = format_time(seg.end)

            ad = is_advertisement(text) or check_custom_keywords(
                text, session.get("keywords", [])
            )

            data = {
                "index": count - 1,
                "start": float(seg.start),
                "end": float(seg.end),
                "start_time": start_t,
                "end_time": end_t,
                "text": text,
                "text_tokens": normalize(text),
                "advertisement": ad,
            }

            with lock:
                merge_advertisement(session, data)

            session["progress"].update(
                {
                    "current_segment": count,
                    "processed_time": end_t,
                    "message": f"Processed segment {count}",
                }
            )

            add_log(sid, text, start_t, end_t, ad)

        duration = round(time.time() - start, 2)

        session["progress"].update(
            {
                "status": "completed",
                "message": "Completed",
                "total_segments": count,
                "processing_time_seconds": duration,
            }
        )

        add_log(sid, f"Completed {count} segments")

    except Exception as e:

        session["progress"].update({"status": "error", "error": str(e)})

        add_log(sid, f"ERROR: {e}")

    finally:

        try:
            os.remove(path)

        except:
            pass


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/upload")
async def upload(file: UploadFile = File(...), keywords: str = Form("")):

    sid = str(uuid.uuid4())

    try:
        keyword_list = [x.strip() for x in json.loads(keywords) if x.strip()]

    except:
        keyword_list = []

    sessions[sid] = {
        "stop": False,
        "progress": create_progress(),
        "logs": [],
        "transcript": [],
        "keywords": keyword_list,
    }

    content = await file.read()

    if not content:
        return {"error": "Empty file"}

    ext = os.path.splitext(file.filename)[1] or ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:

        tmp.write(content)
        path = tmp.name

    sessions[sid]["progress"].update({"status": "starting", "message": "Uploading..."})

    add_log(sid, "File uploaded")

    if keyword_list:

        add_log(sid, "Filters: " + ", ".join(keyword_list))

    threading.Thread(target=transcribe_audio, args=(path, sid), daemon=True).start()

    return {"status": "started", "session_id": sid}


@app.get("/status/{sid}")
def status(sid: str):

    if sid not in sessions:
        return {"error": "not found"}

    return sessions[sid]["progress"]


@app.get("/logs/{sid}")
def logs(sid: str):

    if sid not in sessions:
        return []

    return sessions[sid]["logs"][-100:]


@app.get("/transcript/{sid}")
def transcript(sid: str):

    if sid not in sessions:
        return []

    return sessions[sid]["transcript"]


@app.post("/stop/{sid}")
def stop(sid: str):

    if sid not in sessions:
        return {"status": "not_found"}

    sessions[sid]["stop"] = True

    sessions[sid]["progress"].update({"status": "stopped", "message": "Stopped"})

    add_log(sid, "Processing stopped")

    return {"status": "stopped", "session_id": sid}


@app.post("/reset/{sid}")
def reset(sid: str):

    if sid not in sessions:
        return {"status": "not_found"}

    with lock:
        sessions.pop(sid, None)

    return {"status": "reset", "session_id": sid}
