"""Write UTF-8 dependency records without shell/code-page conversion."""
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
environment = {**os.environ, "PYTHONUTF8": "1"}
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], env=environment).decode("utf-8")
(root / "requirements/windows-training.lock.txt").write_text(freeze, "utf-8")
constraints = [line for line in freeze.splitlines() if "==" in line
               and not line.startswith(("#", "torch==", "fusionvc=="))]
(root / "requirements/windows-constraints.txt").write_text("\n".join(constraints) + "\n", "utf-8")

