import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
manifest=json.loads((root/'models-manifest.json').read_text(encoding='utf-8'))
for item in manifest['files']:
    path=root/item['path']
    if not path.is_file():
        raise SystemExit('Missing '+item['path']+'; run download_models.py')
    if path.stat().st_size!=item['size_bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        raise SystemExit('Invalid checksum: '+item['path'])
print('PASS: all collected model/source files match SHA256.')
