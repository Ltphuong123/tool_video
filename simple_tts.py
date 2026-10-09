"""Run the separate, minimal TTS application from this checkout."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from apps.simple_tts import main
    main()
