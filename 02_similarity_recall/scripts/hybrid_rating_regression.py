"""Run explicit-rating similarity regression on the supplied hybrid data."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "06_integration_ablation" / "scripts" / "hybrid_similarity_graph.py"

if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")
