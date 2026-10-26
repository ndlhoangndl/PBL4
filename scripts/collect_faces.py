"""Launch dataset capture with the existing inference venv (OpenCV + Tk)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai" / "training"))
from pbl4_training.collect_faces import main

if __name__ == "__main__":
    main()
