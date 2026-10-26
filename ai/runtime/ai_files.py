import argparse
import json
from pathlib import Path
from core import ROOT,FaceEngine,VoiceEngine,list_users,delete_user,normalize_otp

def main():
    parser=argparse.ArgumentParser(description='File-only AI for 4 registered people')
    parser.add_argument('--models',type=Path,default=ROOT/'models')
    parser.add_argument('--db',type=Path,default=ROOT/'data'/'faces-v2.sqlite3')
    sub=parser.add_subparsers(dest='command',required=True)
    enroll=sub.add_parser('enroll')
    enroll.add_argument('--slot',type=int,choices=range(1,5),required=True)
    enroll.add_argument('--name',required=True)
    enroll.add_argument('--images',type=Path,nargs='+',required=True)
    enroll.add_argument('--replace',action='store_true')
    face=sub.add_parser('recognize')
    face.add_argument('--image',type=Path,required=True)
    face.add_argument('--threshold',type=float,required=True)
    face.add_argument('--margin',type=float,default=0.05)
    voice=sub.add_parser('voice')
    voice.add_argument('--audio',type=Path,required=True)
    sub.add_parser('list')
    remove=sub.add_parser('delete')
    remove.add_argument('--slot',type=int,choices=range(1,5),required=True)
    normal=sub.add_parser('normalize')
    normal.add_argument('--text',required=True)
    args=parser.parse_args()
    args.db.parent.mkdir(parents=True,exist_ok=True)
    try:
        if args.command=='list':
            result=list_users(args.db)
        elif args.command=='delete':
            result=delete_user(args.db,args.slot)
        elif args.command=='normalize':
            result=dict(digits=normalize_otp(args.text),unlock_allowed=False)
        elif args.command=='voice':
            result=VoiceEngine(args.models/'whisper-base').transcribe(args.audio)
        else:
            engine=FaceEngine(args.models)
            if args.command=='enroll':
                result=engine.enroll(args.db,args.slot,args.name,args.images,args.replace)
            else:
                result=engine.recognize(args.db,args.image,args.threshold,args.margin)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except Exception as exc:
        print(json.dumps(dict(status='error',message=str(exc),unlock_allowed=False),ensure_ascii=False))
        return 1

if __name__=='__main__':
    raise SystemExit(main())
