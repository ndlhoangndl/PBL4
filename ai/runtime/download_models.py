"""Download original model files and record exact revisions, sizes and SHA256."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT=Path(__file__).resolve().parent
WHISPER_REV='ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66'

def read_json(url):
    request=urllib.request.Request(url, headers={'User-Agent':'PBL4-model-collector'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)

def download(item):
    target=ROOT/item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    expected=item.get('sha256')
    if target.is_file() and expected and hashlib.sha256(target.read_bytes()).hexdigest()==expected:
        print('Verified '+item['path'], flush=True)
        return item
    temp=target.with_suffix(target.suffix+'.partial')
    print('Downloading '+item['path'], flush=True)
    try:
        request=urllib.request.Request(item['url'], headers={'User-Agent':'PBL4-model-collector'})
        with urllib.request.urlopen(request, timeout=120) as response, temp.open('wb') as output:
            shutil.copyfileobj(response, output)
        size=temp.stat().st_size
        if size<item.get('minimum_bytes', 1):
            raise ValueError('Download too small: '+item['path'])
        checksum=hashlib.sha256(temp.read_bytes()).hexdigest()
        if expected and checksum!=expected:
            raise ValueError('SHA256 mismatch: '+item['path'])
        temp.replace(target)
        result={**item, 'sha256':checksum, 'size_bytes':size}
        print('Ready '+item['path']+' '+str(size)+' bytes', flush=True)
        return result
    finally:
        temp.unlink(missing_ok=True)

def main():
    manifest=ROOT/'models-manifest.json'
    if manifest.exists():
        items=json.loads(manifest.read_text(encoding='utf-8'))['files']
    else:
        opencv_rev=read_json('https://api.github.com/repos/opencv/opencv_zoo/commits/main')['sha']
        items=[]
        for folder, filename in [('face_detection_yunet','face_detection_yunet_2023mar.onnx'),
                                 ('face_recognition_sface','face_recognition_sface_2021dec.onnx')]:
            prefix=f'models/{folder}/'
            items.append(dict(path='models/'+filename,
                url=f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{opencv_rev}/{prefix}{filename}',
                revision=opencv_rev, minimum_bytes=100000))
            for doc in ['LICENSE','README.md']:
                items.append(dict(path=f'licenses/{folder}/{doc}',
                    url=f'https://raw.githubusercontent.com/opencv/opencv_zoo/{opencv_rev}/{prefix}{doc}',revision=opencv_rev))
        for filename in ['config.json','model.bin','tokenizer.json','vocabulary.txt','README.md']:
            items.append(dict(path='models/whisper-base/'+filename,
                url=f'https://huggingface.co/Systran/faster-whisper-base/resolve/{WHISPER_REV}/{filename}',
                revision=WHISPER_REV, minimum_bytes=100000000 if filename=='model.bin' else 1))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(download, items))
    manifest.write_text(json.dumps(dict(files=results), ensure_ascii=False, indent=2), encoding='utf-8')
    print('All model files verified. Manifest written.', flush=True)

if __name__=='__main__':
    main()
