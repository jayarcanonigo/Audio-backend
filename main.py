from fastapi import FastAPI,UploadFile,File,Form
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile,threading,os,re,time,uuid,json
from queue import Queue

app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"],allow_credentials=True)

model=None
sessions={}
lock=threading.Lock()
job_queue=Queue(maxsize=1)

@app.on_event("startup")
def load():
    global model
    model=WhisperModel("small",device="cpu",compute_type="int8")
    threading.Thread(target=worker,daemon=True).start()

def fmt(s):return f"{int(s//3600):02}:{int(s%3600//60):02}:{int(s%60):02}"
def norm(t):return re.sub(r"[^\w\s]"," ",t.lower()).split()

AD_KEYS=["sponsored","brought to you","hatid ng","hatid sa inyo","time check","oras na"]

def check_kw(t,kws):return any(k.lower() in t.lower() for k in kws if k)

def is_ad(t):
    t=re.sub(r"[^a-z0-9\s]","",t.lower())
    s=sum(2 for k in AD_KEYS if k in t)
    if re.search(r"(09\d{9}|\+639\d{9})",t):s+=2
    if any(x in t for x in["fb","facebook",".com",".ph"]):s+=2
    return s>=2

def merge(s,seg):
    tr=s["transcript"]
    if len(tr)<2:return
    p=tr[-2]
    if p["advertisement"] and seg["advertisement"] and seg["start"]-p["end"]<=5:
        p["end"]=seg["end"]
        p["text"]+=" "+seg["text"]
        p["text_tokens"]=norm(p["text"])

def log(sid,msg,start=None,end=None,ad=False):
    if sid not in sessions:return
    with lock:
        sessions[sid]["logs"].append({
            "id":time.time_ns(),
            "time":time.strftime("%H:%M:%S"),
            "message":msg,
            "start_time":start,
            "end_time":end,
            "advertisement":ad
        })
        sessions[sid]["logs"]=sessions[sid]["logs"][-100:]

def transcribe(path,sid):
    s=sessions.get(sid)
    if not s:return

    try:
        s["progress"]={"status":"transcribing"}
        log(sid,"started")

        segments,info=model.transcribe(path,beam_size=5,vad_filter=True,task="transcribe")
        log(sid,f"lang {info.language}")

        count=0

        for seg in segments:
            if s.get("stop"):
                s["progress"]["status"]="stopped"
                return

            text=(seg.text or "").strip()
            if not text:continue

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
                s["transcript"].append(data)   # FIX: always push segment
                merge(s,data)

            log(sid,text,fmt(seg.start),fmt(seg.end),ad)

        s["progress"]={"status":"completed","total":count}
        log(sid,f"done {count}")

    finally:
        try:os.remove(path)
        except:pass

def worker():
    while True:
        path,sid=job_queue.get()
        try:transcribe(path,sid)
        finally:job_queue.task_done()

@app.post("/upload")
async def upload(file:UploadFile=File(...),keywords:str=Form("")):
    sid=str(uuid.uuid4())

    try:kw=[x.strip() for x in json.loads(keywords) if x.strip()]
    except:kw=[]

    sessions[sid]={
        "stop":False,
        "progress":{"status":"starting"},
        "logs":[],
        "transcript":[],
        "keywords":kw
    }

    with tempfile.NamedTemporaryFile(delete=False,suffix=".audio") as f:
        while chunk:=await file.read(1024*1024):
            f.write(chunk)
        path=f.name

    log(sid,"uploaded")
    job_queue.put((path,sid))

    return {"session_id":sid,"status":"started"}

@app.get("/status/{sid}")
def status(sid):return sessions.get(sid,{}).get("progress",{})

@app.get("/logs/{sid}")
def logs(sid):return sessions.get(sid,{}).get("logs",[])

@app.get("/transcript/{sid}")
def transcript(sid):return sessions.get(sid,{}).get("transcript",[])

@app.post("/stop/{sid}")
def stop(sid):
    if sid in sessions:sessions[sid]["stop"]=True
    return {"status":"stopped"}

@app.post("/reset/{sid}")
def reset(sid):
    if sid not in sessions:return {"status":"not_found"}
    with lock:sessions.pop(sid,None)
    return {"status":"reset"}
