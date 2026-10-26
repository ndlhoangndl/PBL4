"""Launch the existing AI unchanged; workaround OpenCV Unicode model paths on Windows."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ai-dir',type=Path,default=Path(__file__).resolve().parents[1]/'ai'/'runtime')
    parser.add_argument('--model-bundle',type=Path,default=Path(__file__).resolve().parents[1]/'ai')
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--test-db',type=Path,help='Optional isolated smoke-test database; never production fallback')
    args=parser.parse_args()
    root=args.ai_dir.resolve()
    if not (root/'server.py').is_file():
        raise SystemExit('Missing SmartDoor/ai/runtime/server.py; supply --ai-dir.')
    model_bundle=args.model_bundle.resolve()
    manifest_path=model_bundle/'models-manifest.json'
    if not manifest_path.is_file():
        raise SystemExit('Missing model bundle manifest: '+str(manifest_path))
    model_root=model_bundle/'models'
    # Native OpenCV on Windows may reject non-ASCII filenames. Copy exact manifest bytes.
    if os.name=='nt' and not str(model_bundle).isascii():
        cache=Path(tempfile.gettempdir())/'Pbl4SmartDoor-model-cache'
        if not str(cache).isascii():
            raise SystemExit('TEMP must use an ASCII path, or move the AI package to an ASCII path.')
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        for entry in manifest['files']:
            relative=Path(entry['path'])
            source=model_bundle/relative
            if hashlib.sha256(source.read_bytes()).hexdigest()!=entry['sha256']:
                raise SystemExit('Manifest checksum mismatch: '+entry['path'])
            destination=cache/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest()!=entry['sha256']:
                temporary=destination.with_suffix(destination.suffix+'.copying')
                shutil.copyfile(source,temporary)
                temporary.replace(destination)
        shutil.copyfile(manifest_path,cache/'models-manifest.json')
        model_root=cache/'models'
        print('Verified model bundle from SmartDoor/ai in ASCII cache; no model training or replacement.')
    else:
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        for entry in manifest['files']:
            source=model_bundle/Path(entry['path'])
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=entry['sha256']:
                raise SystemExit('Model bundle checksum mismatch: '+entry['path'])
    sys.path.insert(0,str(root))
    server=importlib.import_module('server')
    face_factory=server.FaceEngine
    voice_factory=server.VoiceEngine
    server.FaceEngine=lambda _:face_factory(model_root)
    server.VoiceEngine=lambda _:voice_factory(model_root/'whisper-base')
    if args.test_db:
        server.DB=args.test_db.resolve()
    import uvicorn
    uvicorn.run(server.app,host='127.0.0.1',port=args.port)


if __name__=='__main__':
    main()
