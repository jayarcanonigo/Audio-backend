from fastapi import FastAPI,UploadFile,File,Form
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from queue import Queue
import tempfile,threading,os,re,time,uuid,json,subprocess,shutil
import platform

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

model=None
sessions={}
lock=threading.Lock()
job_queue=Queue(maxsize=1)

FFMPEG=shutil.which("ffmpeg") or (r"C:\ffmpeg\bin\ffmpeg.EXE" if platform.system()=="Windows" else None)

print("FFMPEG:",FFMPEG)

if not FFMPEG:
    raise Exception("FFmpeg not found")


@app.on_event("startup")
def load():
    global model
    print("Loading whisper...")
    model=WhisperModel("small",device="cpu",compute_type="int8")
    print("Whisper ready")
    threading.Thread(target=worker,daemon=True).start()


def fmt(s):
    return f"{int(s//3600):02}:{int(s%3600//60):02}:{int(s%60):02}"


def norm(t):
    return re.sub(r"[^\w\s]"," ",t.lower()).split()


AD_KEYS=[
    "sponsored",
    "brought to you",
    "hatid ng",
    "hatid sa inyo",
    "time check",
    "oras na"
]


def check_kw(t,k):
    return any(x.lower() in t.lower() for x in k if x)


def is_ad(t):
    t=re.sub(r"[^a-z0-9\s]","",t.lower())
    score=sum(2 for x in AD_KEYS if x in t)

    if re.search(r"(09\d{9}|\+639\d{9})",t):
        score+=2

    if any(x in t for x in ["fb","facebook",".com",".ph"]):
        score+=2

    return score>=2


def log(sid,msg,start=None,end=None,ad=False):
    if sid not in sessions:
        return

    with lock:
        sessions[sid]["logs"].append({
            "id":time.time_ns(),
            "time":time.strftime("%H:%M:%S"),
            "message":msg,
            "start_time":start,
            "end_time":end,
            "advertisement":ad
        })


def split_audio(path,sec=300):

    print("SPLIT FILE:",path)

    out=tempfile.mkdtemp()
    pat=os.path.join(out,"chunk_%03d.wav")

    cmd=[
        FFMPEG,
        "-y",
        "-i",
        path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "segment",
        "-segment_time",
        str(sec),
        "-reset_timestamps",
        "1",
        pat
    ]

    print("RUN:",cmd)

    r=subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if r.returncode!=0:
        print(r.stderr)
        raise Exception("ffmpeg failed")

    files=[
        os.path.join(out,x)
        for x in sorted(os.listdir(out))
        if x.endswith(".wav")
    ]

    print("CHUNKS:",files)

    return files


def merge(s,seg):
    tr=s["transcript"]

    if len(tr)<2:
        return

    p=tr[-2]

    if p["advertisement"] and seg["advertisement"] and seg["start"]-p["end"]<=5:
        p["end"]=seg["end"]
        p["text"]+=" "+seg["text"]
        p["text_tokens"]=norm(p["text"])


def transcribe(path,sid):

    s=sessions.get(sid)

    if not s:
        return

    try:

        s["progress"]={"status":"transcribing"}

        log(sid,"started")


        chunks=split_audio(path,300)

        log(sid,f"chunks {len(chunks)}")


        count=0


        for chunk in chunks:

            if s["stop"]:
                return


            segments,info=model.transcribe(
                chunk,
                beam_size=5,
                vad_filter=True,
                task="transcribe"
            )


            log(sid,f"language {info.language}")


            for seg in segments:

                text=(seg.text or "").strip()

                if not text:
                    continue


                count+=1

                ad=is_ad(text) or check_kw(text,s["keywords"])


                data={
                    "index":count-1,
                    "start":seg.start,
                    "end":seg.end,
                    "start_time":fmt(seg.start),
                    "end_time":fmt(seg.end),
                    "text":text,
                    "text_tokens":norm(text),
                    "advertisement":ad
                }


                with lock:
                    s["transcript"].append(data)
                    merge(s,data)


                log(
                    sid,
                    text,
                    fmt(seg.start),
                    fmt(seg.end),
                    ad
                )


            os.remove(chunk)


        s["progress"]={
            "status":"completed",
            "total":count
        }

        log(sid,f"done {count}")


    except Exception as e:

        print("ERROR:",e)

        s["progress"]={
            "status":"error",
            "message":str(e)
        }

        log(sid,f"ERROR {e}")


    finally:

        try:
            os.remove(path)
        except:
            pass



def worker():

    print("WORKER READY")

    while True:

        path,sid=job_queue.get()

        print("WORKER START:",path)

        try:
            transcribe(path,sid)

        finally:
            job_queue.task_done()



@app.post("/upload")
async def upload(
    file:UploadFile=File(...),
    keywords:str=Form("")
):

    sid=str(uuid.uuid4())


    try:
        kw=json.loads(keywords)
    except:
        kw=[]


    sessions[sid]={
        "stop":False,
        "progress":{"status":"starting"},
        "logs":[],
        "transcript":[],
        "keywords":kw
    }


    with tempfile.NamedTemporaryFile(delete=False,suffix=".audio") as f:

        while data:=await file.read(1024*1024):
            f.write(data)

        path=f.name


    log(sid,"uploaded")

    job_queue.put((path,sid))


    return {
        "session_id":sid,
        "status":"started"
    }



@app.get("/status/{sid}")
def status(sid):
    return sessions.get(sid,{}).get("progress",{})


@app.get("/logs/{sid}")
def logs(sid):
    return sessions.get(sid,{}).get("logs",[])


@app.get("/transcript/{sid}")
def transcript(sid):
    return sessions.get(sid,{}).get("transcript",[])


@app.post("/stop/{sid}")
def stop(sid):

    if sid in sessions:
        sessions[sid]["stop"]=True

    return {"status":"stopped"}


@app.post("/reset/{sid}")
def reset(sid):

    sessions.pop(sid,None)

    return {"status":"reset"}
