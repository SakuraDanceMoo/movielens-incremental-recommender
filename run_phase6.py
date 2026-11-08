"""Run the stage-six integrated rank-fusion and ablation experiment."""
from pathlib import Path
import runpy

root = Path(__file__).resolve().parent
runpy.run_path(str(root / "06_integration_ablation/scripts/integrated_experiment.py"), run_name="__main__")
