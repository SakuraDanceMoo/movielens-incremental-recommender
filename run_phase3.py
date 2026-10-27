"""Run the stage-three Spark recommendation experiment."""
from pathlib import Path
import os
import runpy

root = Path(__file__).resolve().parent
shared_env = root.parent / ".venv"
bundled_jdk = shared_env / "jdk11"
if not os.environ.get("JAVA_HOME") and (bundled_jdk / "bin/java.exe").is_file():
    os.environ["JAVA_HOME"] = str(bundled_jdk)
    os.environ["PATH"] = str(bundled_jdk / "bin") + os.pathsep + os.environ.get("PATH", "")

script = root / "03_distributed_recommendation/scripts/phase3_experiment.py"
runpy.run_path(str(script), run_name="__main__")
