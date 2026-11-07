import os
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
RAW = ROOT / "study/channel/results/raw"

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
try:
    import score_architectures
    ARCH_CELLS = score_architectures.CELLS
except Exception as e:
    ARCH_CELLS = {}
    print(f"note: score_architectures did not import ({type(e).__name__}), so the cells are read from their "
          "file names")

NO_PROFILE = ("_default", "_text", "_none", "_bare", "_notprofile")


def pairs(p: Path):
    rows = {}
    for ln in p.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[r["query_id"]] = r
    group = collections.defaultdict(list)
    for q in rows:
        group[q.rsplit("-", 1)[0]].append(q)
    identical = total = 0
    for _, qs in group.items():
        if len(qs) != 2:
            continue
        total += 1
        identical += rows[qs[0]]["ranking"] == rows[qs[1]]["ranking"]
    return identical, total


def wins(p: Path):
    n = 0
    present = set()
    for ln in p.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        gold = set(r.get("gold") or [])
        cp = r.get("counterpart")
        if cp is None:
            continue
        cp = set(cp) if isinstance(cp, list) else {cp}
        rank = r["ranking"]
        pg = next((i for i, c in enumerate(rank) if c in gold), None)
        pc = next((i for i, c in enumerate(rank) if c in cp), None)
        if pg is None or pc is None:
            continue
        present.add(r["query_id"].rsplit("-", 1)[0])
        if pg < pc:
            n += 1
    return n, len(present)


def command(p: Path) -> str:
    for cand in (p.parent / (p.stem + ".manifest.json"),):
        if cand.exists():
            d = json.loads(cand.read_text())
            return str(d.get("perintah_persis") or d.get("perintah") or "")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    failed, checked, n_skipped = [], 0, 0
    for p in sorted(RAW.glob("scores_*.jsonl")):
        if "paired" not in p.name and "tangga" not in p.name:
            continue
        if "smoke" in p.name:
            n_skipped += 1
            continue
        try:
            identical, total = pairs(p)
        except (json.JSONDecodeError, KeyError):
            n_skipped += 1
            continue
        if total == 0:
            n_skipped += 1
            continue
        checked += 1
        cmd = command(p)
        has_profile = None
        if "--instruction" in cmd:
            inst = cmd.split("--instruction", 1)[1].split()[0]
            has_profile = inst.startswith("profile") or inst == "kondisi"
        elif "--cell" in cmd:
            cell = cmd.split("--cell", 1)[1].split()[0]
            block = ARCH_CELLS.get(cell)
            if block is not None:
                has_profile = block["inst"] not in ("none", "generic", "bare")
        if has_profile is None:
            has_profile = not any(k in p.name for k in NO_PROFILE)

        if has_profile and identical == total:
            failed.append((p.name, identical, total,
                          "carries a profile but every pair is identical, so the profile never reaches the model"))
        elif not has_profile and identical != total:
            w, n_pairs = wins(p)
            if w != n_pairs:
                failed.append((p.name, identical, total,
                              f"carries no profile, {total - identical} pairs "
                              f"differ, AND its verdict {w} is off the lock {n_pairs}"))
            elif args.verbose:
                print(f"  ok  {identical:>3}/{total:<3} {p.name}  "
                      f"({total - identical} pairs shifted below the gold, "
                      f"verdict still at the lock)")
        elif args.verbose:
            print(f"  ok  {identical:>3}/{total:<3} {p.name}")

    print(f"\nchecked {checked} paired-track cells, skipped {n_skipped}")
    if failed:
        print(f"\nFAILED on {len(failed)} cells:")
        for name, i, t, cause in failed:
            print(f"  {i}/{t}  {name}\n        {cause}")
        sys.exit(2)
    print("PASS: every cell agrees with its own command")


if __name__ == "__main__":
    main()
