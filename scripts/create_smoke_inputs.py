"""Synthetic blank image and silence for rejection tests, never facial/voice identity data."""
from pathlib import Path
import wave
import cv2
import numpy as np

output=Path(__file__).resolve().parents[1]/'TestResults'
output.mkdir(exist_ok=True)
encoded=cv2.imencode('.png',np.zeros((320,320,3),dtype=np.uint8))[1]
(output/'blank.png').write_bytes(encoded.tobytes())
with wave.open(str(output/'silence.wav'),'wb') as audio:
    audio.setnchannels(1)
    audio.setsampwidth(2)
    audio.setframerate(16000)
    audio.writeframes(bytes(16000*2))
print('Generated blank image and 1-second silence; synthetic negative-test inputs only.')
