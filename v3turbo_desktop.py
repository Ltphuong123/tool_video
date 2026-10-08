"""Run the desktop tool directly from a source checkout."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from apps.v3turbo_desktop import main
    main()
