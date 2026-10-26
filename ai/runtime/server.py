import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import tempfile
import time
import uuid
from fastapi import FastAPI,File,Form,HTTPException,UploadFile
from fastapi.responses import FileResponse
from core import ROOT,FaceEngine,VoiceEngine,list_users,delete_user

DB=ROOT/'data'/'faces-v2.sqlite3'

@asynccontextmanager
async def lifespan(app):
    DB.parent.mkdir(parents=True,exist_ok=True)
    app.state.face=FaceEngine(ROOT/'models')
    app.state.voice=VoiceEngine(ROOT/'models'/'whisper-base')
    # Serialize inference/enrollment to keep memory and CPU use bounded.
    app.state.busy=asyncio.Lock()
    yield

app=FastAPI(title='PBL4 AI · file input',version='2.0',lifespan=lifespan)

@app.get('/',include_in_schema=False)
def index():
    return FileResponse(ROOT/'ui.html')

@app.get('/health')
def health():
    return dict(status='ready',face_model='SFace-2021dec',voice_model='Whisper-base',
        maximum_users=4,input='uploaded_files_only',liveness='not_implemented',unlock_allowed=False)

@app.get('/users')
def users():
    return list_users(DB)

@app.delete('/users/{slot}')
def remove(slot:int):
    try:
        return delete_user(DB,slot)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc

async def save_file(upload,directory,index,limit):
    try:
        data=await upload.read(limit+1)
        if not data or len(data)>limit:
            raise ValueError('Empty file or upload size exceeded')
        # Never use client-provided filenames as a filesystem path.
        path=Path(directory)/str(index)
        path.write_bytes(data)
        return path
    finally:
        await upload.close()

async def run_job(function,session_id):
    if len(session_id)>100:
        raise HTTPException(422,'session_id too long')
    started=time.perf_counter()
    try:
        async with app.state.busy:
            result=await asyncio.to_thread(function)
        return {**result,'session_id':session_id or str(uuid.uuid4()),
                'request_id':str(uuid.uuid4()),'elapsed_ms':round((time.perf_counter()-started)*1000)}
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    except Exception as exc:
        # API doesn't leak device paths or internals; full debugging belongs in local development.
        raise HTTPException(503,'AI processing failed') from exc

@app.post('/face/enroll')
async def enroll(slot:int=Form(...),name:str=Form(...),replace:bool=Form(False),
                 images:list[UploadFile]=File(...),session_id:str=Form('')):
    if slot not in [1,2,3,4] or not name.strip() or not 3<=len(images)<=30:
        raise HTTPException(422,'Slot 1–4, name and 3–30 images required')
    try:
        with tempfile.TemporaryDirectory(prefix='pbl4-enroll-') as directory:
            paths=[await save_file(upload,directory,i,10*1024*1024) for i,upload in enumerate(images)]
            return await run_job(lambda:app.state.face.enroll(DB,slot,name,paths,replace),session_id)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc

@app.post('/face/recognize')
async def recognize(image:UploadFile=File(...),threshold:float=Form(...),
                    margin:float=Form(0.05),session_id:str=Form('')):
    try:
        with tempfile.TemporaryDirectory(prefix='pbl4-face-') as directory:
            path=await save_file(image,directory,0,10*1024*1024)
            return await run_job(lambda:app.state.face.recognize(DB,path,threshold,margin),session_id)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc

@app.post('/voice/transcribe')
async def voice(audio:UploadFile=File(...),session_id:str=Form('')):
    try:
        with tempfile.TemporaryDirectory(prefix='pbl4-voice-') as directory:
            path=await save_file(audio,directory,0,20*1024*1024)
            return await run_job(lambda:app.state.voice.transcribe(path),session_id)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
