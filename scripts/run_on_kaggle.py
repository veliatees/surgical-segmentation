"""Run the study on Kaggle with one command: 01 (data check) -> 02 (train) -> 03 (validation) -> 04 (test).

One-time setup on your computer (the same as for the nucleus study):
  1. pip install kaggle
  2. kaggle.com/settings/api -> "Create Legacy API Key", move kaggle.json to ~/.kaggle/ and run
     chmod 600 ~/.kaggle/kaggle.json
  3. Open https://www.kaggle.com/datasets/newslab/cholecseg8k once in the browser while logged in.

Then, from the repository folder:
  python scripts/run_on_kaggle.py 01 02 03     # data check, training, validation
  python scripts/run_on_kaggle.py 04           # the test, once, after looking at 03

Every notebook runs as a private Kaggle notebook. 01 runs on CPU, the others on a GPU with internet on.
Each one gets the dataset and the output of the notebook before it as input.
The outputs land in runs/<date-time>/<notebook>/ and the key result files are printed at the end.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = "newslab/cholecseg8k"
STEPS = {  # name: (notebook, kaggle slug, step whose output it needs, GPU)
    "01": ("notebooks/01_data_check.ipynb", "surgseg-01-data-check", None, False),
    "02": ("notebooks/02_train.ipynb", "surgseg-02-train", "01", True),
    "03": ("notebooks/03_validation.ipynb", "surgseg-03-validation", "02", True),
    "04": ("notebooks/04_test_set.ipynb", "surgseg-04-test-set", "02", True),
}
SHOW = ["frames_per_video.csv", "mapping_check.csv", "class_pixels.csv", "classes.json",
        "train_log_A.csv", "train_log_B.csv", "val_per_class.csv", "val_summary.json",
        "test_per_class.csv", "test_summary.json"]
POLL_SECONDS = 60
MAX_HOURS = 6


def kaggle(*args):
    exe = shutil.which("kaggle")
    cmd = [exe, *args] if exe else [sys.executable, "-m", "kaggle", *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{res.stdout}\n{res.stderr}")
    return res.stdout


def username():
    # the Kaggle CLI logs in by itself; this script only needs your user name to build the notebook address
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text())["username"]
    name = input("Kaggle user name (as in kaggle.com/<name>): ").strip()
    if not name:
        sys.exit("no user name given")
    return name


def push(user, step):
    notebook, slug, needs, gpu = STEPS[step]
    sources = [f"{user}/{STEPS[needs][1]}"] if needs else []
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(REPO / notebook, Path(tmp) / Path(notebook).name)
        meta = {"id": f"{user}/{slug}", "title": slug, "code_file": Path(notebook).name, "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
                "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": sources}
        (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        print(f"[{step}] {kaggle('kernels', 'push', '-p', tmp).strip()}")
    return f"{user}/{slug}"


def wait(step, kernel_id):
    start = time.time()
    while True:
        out = kaggle("kernels", "status", kernel_id)
        m = re.search(r'status\s+"?(?:KernelWorkerStatus\.)?([A-Za-z_]+)', out)
        status = (m.group(1) if m else out.strip()).lower()
        print(f"[{step}] {datetime.now():%H:%M:%S}  {status}")
        if "complete" in status:
            return True
        if "error" in status or "cancel" in status:
            print(f"[{step}] stopped with status {status}; log: https://www.kaggle.com/code/{kernel_id}")
            return False
        if time.time() - start > MAX_HOURS * 3600:
            sys.exit(f"[{step}] still running after {MAX_HOURS} h; check https://www.kaggle.com/code/{kernel_id}")
        time.sleep(POLL_SECONDS)


def show_log(target, n=40):
    """Print the end of the Kaggle log, without the debugger noise, so the error is visible here."""
    noise = ("frozen_modules", "frozen modules", "Debugger warning", "PYDEVD", "Debugging will proceed",
             "NbConvertApp", "SyntaxWarning", "re.sub(")
    for log in sorted(target.glob("*.log")):
        raw = log.read_text()
        try:
            text = "".join(e.get("data", "") for e in json.loads(raw))
        except ValueError:
            text = raw
        lines = [l for l in text.splitlines() if l.strip() and not any(x in l for x in noise)]
        print(f"===== end of {log.name}")
        print("\n".join(lines[-n:]))


def main():
    steps = sys.argv[1:] or ["01", "02", "03"]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        sys.exit(f"unknown step(s) {unknown}; choose from {list(STEPS)}")
    user = username()
    run_dir = REPO / "runs" / f"{datetime.now():%Y-%m-%d_%H-%M}"
    for step in steps:
        kernel_id = push(user, step)
        time.sleep(10)
        ok = wait(step, kernel_id)
        target = run_dir / step
        target.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", kernel_id, "-p", str(target))
        if not ok:
            show_log(target)
            sys.exit(f"[{step}] failed; output and log are in {target}")
        print(f"[{step}] output in {target}")
    print()
    for name in SHOW:
        for f in sorted(run_dir.rglob(name)):
            if "out" not in f.parts:
                continue
            print(f"===== {f.relative_to(run_dir)}")
            print(f.read_text()[:3000])


if __name__ == "__main__":
    main()
