"""One-command stage-two reproduction on the frozen stage-one artifacts."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "02_similarity_recall"


def main():
    (STAGE / "logs").mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-X", "utf8", str(STAGE / "scripts" / "phase2_experiment.py"), "--sweep"]
    with (STAGE / "logs" / "phase2_run.log").open("w", encoding="utf-8") as handle:
        handle.write("Command: python -X utf8 02_similarity_recall/scripts/phase2_experiment.py --sweep\n")
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", cwd=ROOT,
                              env={**os.environ, "PYTHONUTF8": "1"}) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            if process.wait() != 0:
                raise SystemExit("Stage two failed; inspect 02_similarity_recall/logs/phase2_run.log")
    print("Stage two local validation passed. LSH acceptance remains a measured negative result; teacher-script alignment remains pending.")


if __name__ == "__main__":
    main()
