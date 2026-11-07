import os
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
RAW = ROOT / "study/channel/results/raw"
TRACK = {"paired": 104, "xpair": 1621, "ctrl": 116}
POOL = {
    "paired": ROOT / "study/retrieval/results/raw/pools_paired.jsonl",
    "xpair": RAW / "pools_paired_expanded.jsonl",
    "ctrl": RAW / "pools_control_textual.jsonl",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def flag(command: str, name: str):
    m = re.search(rf"{re.escape(name)}\s+(\S+)", command or "")
    return m.group(1) if m else None


def read_cell(p: Path) -> dict:
    man = p.with_name(p.stem + ".manifest.json")
    cell = {"berkas": p.name, "manifest": man.exists(), "smoke": "smoke" in p.name.lower()}
    ids = []
    for l in p.open():
        if l.strip():
            try:
                ids.append(json.loads(l)["query_id"])
            except Exception:
                ids.append("?")
    cell["n"] = len(ids)
    pre = collections.Counter(i.split("-")[0] for i in ids)
    cell["prefiks"] = dict(pre)
    cell["track_isi"] = pre.most_common(1)[0][0] if pre else None
    cell["duplikat"] = sum(v - 1 for v in collections.Counter(ids).values() if v > 1)
    if man.exists():
        d = json.loads(man.read_text())
        cmd = d.get("perintah_persis") or ""
        cell["model"] = d.get("model")
        cell["dtype"] = flag(cmd, "--dtype") or d.get("dtype")
        cell["max_len"] = flag(cmd, "--max-len") or d.get("max_len")
        cell["batch"] = flag(cmd, "--batch") or d.get("batch_dipakai")
        cell["pool_sha"] = (d.get("sumber") or {}).get("pools")
        cell["config"] = d.get("config_version")
        cell["cell"] = d.get("sel") or flag(cmd, "--doc-label-gaya") or flag(cmd, "--tag")
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    pool_sha = {t: sha(p) for t, p in POOL.items() if p.exists()}
    cell = [read_cell(p) for p in sorted(RAW.glob("scores_*.jsonl"))]
    actual = [s for s in cell if not s["smoke"] and "_envb1_" not in s["berkas"]]
    running = {s["berkas"] for s in actual
                    if not s["manifest"] and s["n"] and s["n"] < TRACK.get(s["track_isi"], 0)}

    failed = collections.defaultdict(list)
    for s in actual:
        n = s["berkas"]
        if not s["manifest"]:
            if n in running:
                failed["0 still being written, not a defect"].append(f"{n}: {s['n']} rows")
            else:
                failed["9 manifest missing, provenance in its table"].append(n)
            continue
        claim = ("xpair" if "xpair" in n else "ctrl" if "control" in n or "_ctrl" in n
                 else "paired" if "_paired" in n or "tangga" in n else None)
        if claim and s["track_isi"] != claim:
            failed["2 content track does not match the name"].append(f"{n}: name {claim}, content {s['track_isi']}")
        t = s["track_isi"]
        if t in TRACK and s["n"] != TRACK[t]:
            failed["3 row count does not match the track"].append(f"{n}: {s['n']} instead of {TRACK[t]}")
        if s["duplikat"]:
            failed["4 duplicate query_id"].append(f"{n}: {s['duplikat']}")
        if s["pool_sha"] and t in pool_sha and s["pool_sha"] != pool_sha[t]:
            failed["5 pool sha of another track"].append(n)

    group = collections.defaultdict(list)
    for s in actual:
        if s.get("model") and s["track_isi"]:
            group[(s["model"], s["track_isi"])].append(s)
    for (model, t), members in sorted(group.items()):
        for field, num in (("dtype", "6"), ("max_len", "7")):
            value = collections.Counter(str(a.get(field)) for a in members)
            if len(value) > 1:
                mode = value.most_common(1)[0][0]
                for a in members:
                    if str(a.get(field)) != mode:
                        failed[f"{num} {field} differs from the majority of its group"].append(
                            f"{a['berkas']}: {field} {a.get(field)} against the majority {mode} ({model.split('/')[-1]}"
                            f", {t})")
        if len(members) == 1:
            failed["8 only cell of its group, nothing to compare with"].append(
                f"{members[0]['berkas']} ({model.split('/')[-1]}, {t})")

    if args.json:
        print(json.dumps({"gagal": {k: v for k, v in failed.items()},
                          "n_sel": len(actual)}, indent=1, ensure_ascii=False))
    else:
        print(f"  {len(cell)} score files, {len(actual)} of them not smoke tests\n")
        print("  groups (model, track) and their settings:")
        for (model, t), members in sorted(group.items()):
            dt = sorted({str(a.get("dtype")) for a in members})
            ml = sorted({str(a.get("max_len")) for a in members})
            print(f"    {model.split('/')[-1]:44s} {t:7s} {len(members):>3} cells "
                  f"dtype {','.join(dt):12s} max_len {','.join(ml)}")
        print()
        if not [k for k in failed if k[0] in "1234567"]:
            print("  PASS: the six blocking checks (2 to 7) hold, and every cell is comparable")
            if len(pool_sha) < len(POOL):
                print(f"  check 5 (pool sha) covered {len(pool_sha)} of {len(POOL)} tracks, because the "
                      "other text pools are not here")
        else:
            for k in sorted(failed):
                print(f"  FAILED {k}: {len(failed[k])}")
                for v in failed[k][:12]:
                    print(f"      {v}")
    blockers = [k for k in failed if k[0] in "1234567"]
    raise SystemExit(2 if blockers else 0)


if __name__ == "__main__":
    main()
