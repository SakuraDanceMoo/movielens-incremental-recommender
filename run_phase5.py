"""Run the stage-five graph recommendation experiment."""
from pathlib import Path
import runpy

root = Path(__file__).resolve().parent
runpy.run_path(str(root / "05_graph_recommendation/scripts/graph_experiment.py"), run_name="__main__")
