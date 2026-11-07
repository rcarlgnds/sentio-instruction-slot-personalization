import os
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[3])

AI = ROOT / "system"
NOV1 = Path(__file__).resolve().parents[1]
RAW = NOV1 / "results" / "raw"


def run_module(mod_args: list[str], timeout=7200) -> None:
    py = AI / ".venv/bin/python"
    if not py.exists():
        py = Path(sys.executable)
    r = subprocess.run([str(py), "-m", *mod_args],
                       cwd=AI, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise SystemExit(f"{mod_args[0]} failed:\n{r.stderr[-2500:]}")
    print(r.stdout[-600:])


def package(src: Path, name: str, notes: str) -> None:
    dst = RAW / src.name
    dst.write_bytes(src.read_bytes())
    (RAW / f"{name}.manifest.json").write_text(json.dumps({
        "run_id": f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "script": "study/retrieval/scripts/paired_hardset.py",
        "sumber": str(src),
        "label_source": "deterministic",
        "output": {"path": f"results/raw/{src.name}",
                   "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
                   "tingkat_reproduksi": "1_bit_identical"},
        "catatan": notes,
    }, indent=2, ensure_ascii=False))
    print(f"packaged: {dst.name}")


def main():
    required_files = [
        AI / "data/paired_set/paired_queries.jsonl",
        AI / "data/rads_sample/supplementary_hard_set.jsonl",
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise SystemExit(
            "ABORT: this script needs local evaluation data that is not in this repository: " + ", ".join(missing))

    print("=== PAIRED SET (Track 2) ===")
    run_module(["scripts.eval_paired"])
    package(AI / "data/paired_set/eval_paired.json", "paired_set",
          "52 pairs, arms A S1 S2 B C D, reported apart from the main benchmark.")

    print("=== HARD SET (selection-bias control) ===")
    out = AI / "data/eval_hardset_t2"
    run_module(["scripts.pilot_ablation", "--queries",
           "data/rads_sample/supplementary_hard_set.jsonl", "--all",
           "--out", str(out)])
    for f in sorted(out.glob("*.json")):
        package(f, f"hardset_{f.stem}",
              "603 queries rejected by the QC gate of Config B. If the relative order of the arms "
              "differs from the main benchmark, the QC gate affects the results.")

    print("done.")


if __name__ == "__main__":
    main()
