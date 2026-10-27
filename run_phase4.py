"""One-command stage-four event-time replay experiment."""
from pathlib import Path
import runpy

root = Path(__file__).resolve().parent
runpy.run_path(str(root / "04_streaming_incremental/scripts/phase4_replay.py"), run_name="__main__")
