"""One-command stage-one reproduction. Commands and real terminal output are retained."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "01_data_baseline"


def main():
    (STAGE / "logs").mkdir(parents=True, exist_ok=True)
    tasks = [("download.py", [], "download.log"), ("test_protocol.py", [], "protocol_tests.log"),
             ("prepare.py", [], "prepare.log"), ("clean_metadata.py", [], "metadata_cleanup.log"), ("evaluate.py", [], "baseline.log"),
             ("evaluate.py", ["--output-prefix", "repeat"], "repeat.log"), ("verify.py", [], "verify.log")]
    for script, args, log in tasks:
        command = [sys.executable, "-X", "utf8", str(STAGE / "scripts" / script), *args]
        with (STAGE / "logs" / log).open("w", encoding="utf-8") as f:
            f.write("Command: python -X utf8 01_data_baseline/scripts/" + script + " " + " ".join(args) + "\n")
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"}) as p:
                for line in p.stdout:
                    print(line, end="", flush=True); f.write(line); f.flush()
                if p.wait() != 0:
                    raise SystemExit(f"Failed {script}; inspect 01_data_baseline/logs/{log}")
    print("Stage one local validation passed. Teacher-script alignment remains pending.")


if __name__ == "__main__":
    main()
