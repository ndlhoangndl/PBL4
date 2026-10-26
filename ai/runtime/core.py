"""File-only engines. No camera/microphone capture or lock control."""
from pathlib import Path
from contextlib import contextmanager
import json
import re
import sqlite3
import threading
import unicodedata

ROOT=Path(__file__).resolve().parent
DIGITS=dict(zip('không một hai ba bốn năm sáu bảy tám chín'.split(),'0123456789'))

def normalize_otp(text):
    tokens=re.sub(r'[,.;:!?]',' ',unicodedata.normalize('NFC',text).lower()).split()
    parts=[]
    for token in tokens:
        if token in DIGITS:
            parts.append(DIGITS[token])
        elif re.fullmatch(r'[0-9]+',token):
            parts.append(token)
        else:
            return None
    result=''.join(parts)
    return result if re.fullmatch(r'[0-9]{6}',result) else None

@contextmanager
def database(path):
    db=sqlite3.connect(path)
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('CREATE TABLE IF NOT EXISTS users (slot INTEGER PRIMARY KEY CHECK(slot BETWEEN 1 AND 4), name TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS templates (id INTEGER PRIMARY KEY, slot INTEGER NOT NULL REFERENCES users(slot) ON DELETE CASCADE, vector TEXT NOT NULL, model TEXT NOT NULL)')
    try:
        with db:
            yield db
    finally:
        db.close()

class FaceEngine:
    def __init__(self,models):
        import cv2
        import numpy as np
        self.cv=cv2
        self.np=np
        self.lock=threading.RLock()
        for filename in ['face_detection_yunet_2023mar.onnx','face_recognition_sface_2021dec.onnx']:
            if not (models/filename).is_file():
                raise ValueError('Missing model '+filename)
        self.det=cv2.FaceDetectorYN.create(str(models/'face_detection_yunet_2023mar.onnx'),'',(320,320),0.9)
        self.rec=cv2.FaceRecognizerSF.create(str(models/'face_recognition_sface_2021dec.onnx'),'')

    def embed(self,path):
        if not path.is_file() or path.stat().st_size>10*1024*1024:
            raise ValueError('Image missing or exceeds 10 MB')
        image=self.cv.imdecode(self.np.frombuffer(path.read_bytes(),dtype=self.np.uint8),self.cv.IMREAD_COLOR)
        if image is None:
            raise ValueError('Invalid image')
        h,w=image.shape[:2]
        if max(h,w)>1920:
            scale=1920/max(h,w)
            image=self.cv.resize(image,(round(w*scale),round(h*scale)))
        with self.lock:
            self.det.setInputSize((image.shape[1],image.shape[0]))
            _,faces=self.det.detect(image)
            if faces is None or len(faces)!=1:
                raise ValueError('Exactly one face required')
            if min(faces[0][2:4])<80:
                raise ValueError('Face too small')
            crop=self.rec.alignCrop(image,faces[0])
            vector=self.rec.feature(crop).flatten().copy()
        norm=self.np.linalg.norm(vector)
        if not self.np.isfinite(norm) or norm<=0:
            raise ValueError('Invalid face feature')
        return vector/norm

    def enroll(self,db_path,slot,name,images,replace=False):
        if slot not in [1,2,3,4] or not name.strip() or not 3<=len(images)<=30:
            raise ValueError('Slot 1–4, nonempty name and 3–30 images required')
        vectors=[self.embed(path).tolist() for path in images]
        with database(db_path) as db:
            db.execute('BEGIN IMMEDIATE')
            exists=db.execute('SELECT 1 FROM users WHERE slot=?',(slot,)).fetchone()
            if exists and not replace:
                raise ValueError('Slot occupied; explicitly enable replace')
            if exists:
                db.execute('UPDATE users SET name=? WHERE slot=?',(name.strip(),slot))
                db.execute('DELETE FROM templates WHERE slot=?',(slot,))
            else:
                db.execute('INSERT INTO users VALUES (?,?)',(slot,name.strip()))
            db.executemany('INSERT INTO templates(slot,vector,model) VALUES (?,?,?)',
                [(slot,json.dumps(v),'SFace-2021dec') for v in vectors])
        return dict(status='enrolled',slot=slot,samples=len(vectors),unlock_allowed=False)

    def recognize(self,db_path,image,threshold,margin):
        if not 0<=threshold<=1 or not 0<=margin<=1:
            raise ValueError('Threshold and margin must be 0–1')
        query=self.embed(image)
        grouped={}
        with database(db_path) as db:
            rows=db.execute('SELECT u.slot,u.name,t.vector,t.model FROM users u JOIN templates t ON u.slot=t.slot').fetchall()
        for slot,name,vector,model in rows:
            if model!='SFace-2021dec':
                raise ValueError('Template model mismatch; re-enroll')
            stored=self.np.array(json.loads(vector),dtype=self.np.float32)
            if stored.shape!=query.shape or not self.np.isfinite(stored).all():
                raise ValueError('Invalid stored face template')
            similarity=float(self.np.dot(query,stored))
            if slot not in grouped or similarity>grouped[slot]['similarity']:
                grouped[slot]=dict(slot=slot,name=name,similarity=similarity)
        ranking=sorted(grouped.values(),key=lambda item:item['similarity'],reverse=True)
        best=ranking[0] if ranking else None
        gap=best['similarity']-ranking[1]['similarity'] if len(ranking)>1 else None
        matched=best is not None and best['similarity']>=threshold and (gap is None or gap>=margin)
        return dict(status='candidate' if matched else 'unknown',candidate=best if matched else None,
            ranking=ranking,threshold=threshold,minimum_margin=margin,score_gap=gap,
            liveness='not_implemented',unlock_allowed=False)

def list_users(db_path):
    with database(db_path) as db:
        return [dict(slot=s,name=n,samples=c) for s,n,c in db.execute(
            'SELECT u.slot,u.name,COUNT(t.id) FROM users u LEFT JOIN templates t ON u.slot=t.slot GROUP BY u.slot ORDER BY u.slot')]

def delete_user(db_path,slot):
    if slot not in [1,2,3,4]:
        raise ValueError('Invalid slot')
    with database(db_path) as db:
        count=db.execute('DELETE FROM users WHERE slot=?',(slot,)).rowcount
    return dict(status='deleted' if count else 'not_found',slot=slot)

class VoiceEngine:
    def __init__(self,model_path):
        from faster_whisper import WhisperModel
        if not (model_path/'model.bin').is_file():
            raise ValueError('Missing Whisper model')
        self.model=WhisperModel(str(model_path),device='cpu',compute_type='int8',local_files_only=True)
        self.lock=threading.Lock()

    def transcribe(self,path):
        from faster_whisper.audio import decode_audio
        if not path.is_file() or path.stat().st_size>20*1024*1024:
            raise ValueError('Audio missing or exceeds 20 MB')
        try:
            waveform=decode_audio(str(path),sampling_rate=16000)
        except Exception as exc:
            raise ValueError('Cannot decode audio; provide a valid recording') from exc
        if not 0.3<=len(waveform)/16000<=20:
            raise ValueError('Audio must be 0.3–20 seconds')
        with self.lock:
            segments,_=self.model.transcribe(waveform,language='vi',beam_size=5,
                vad_filter=True,condition_on_previous_text=False,temperature=0)
            text=' '.join(segment.text.strip() for segment in segments)
        code=normalize_otp(text)
        return dict(status='decoded' if code else 'retry',transcript=text,digits=code,
            otp_verified=False,unlock_allowed=False)
