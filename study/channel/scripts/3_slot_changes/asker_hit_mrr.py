import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import channel_cells as K  # noqa: E402

OUT = K.TABLE / "asker_hit_mrr.json"
MANIFEST = K.RAW / "asker_hit_mrr.manifest.json"
CELLS = {"paired": ["none", "slot", "text", "both"], "xpair": ["none", "slot", "both"]}


def measure(p: Path, comparable: set) -> dict:
    hit = rival = 0
    rr = 0.0
    n = 0
    for q, r in K.read(p).items():
        if q not in comparable:
            continue
        n += 1
        gold = set(r.get("gold") or [])
        counterpart = r.get("counterpart") or []
        counterpart = set(counterpart) if isinstance(counterpart, list) else {counterpart}
        rank = r["ranking"]
        top = rank[0] if rank else None
        hit += top in gold
        rival += top in counterpart
        pg = next((i for i, c in enumerate(rank) if c in gold), None)
        if pg is not None and pg < 10:
            rr += 1.0 / (pg + 1)
    return {"n": n, "hit1": round(hit / n, 4) if n else None,
            "rival1": round(rival / n, 4) if n else None,
            "mrr10": round(rr / n, 4) if n else None, "hit1_baris": hit, "rival1_baris": rival}


def flip_rate(p: Path) -> dict:
    group = defaultdict(list)
    for q, r in K.read(p).items():
        group[q.rsplit("-", 1)[0]].append(r["ranking"][0] if r["ranking"] else None)
    n = sum(1 for v in group.values() if len(v) >= 2)
    n_differ = sum(1 for v in group.values() if len(v) >= 2 and len(set(v)) > 1)
    return {"grup": n, "grup_top1_berbeda": n_differ, "laju": round(n_differ / n, 4) if n else None}


def main():
    flip = {t: {m: flip_rate(K.cell_path(m, t, "slot")) for m in K.MODEL} for t in CELLS}
    flip_none = {t: {m: flip_rate(K.cell_path(m, t, "none")) for m in K.MODEL} for t in CELLS}
    ref = json.loads((K.TABLE / "flip_rate.json").read_text())["per_arm"]
    for m, key in (("Qwen3-Reranker-0.6B", "C3_0.6b_profil"), ("Qwen3-Reranker-4B", "C4_4b_profil")):
        if abs(flip["paired"][m]["laju"] - ref[key]["flip_top1"]) > 1e-4:
            raise SystemExit(f"GATE STOP: flip rate {m} {flip['paired'][m]['laju']} is not "
                             f"{ref[key]['flip_top1']} as in flip_rate.json")

    result = {}
    for t, cells in CELLS.items():
        result[t] = {}
        for s in cells:
            wins = K.manuscript_gate(t, s)
            result[t][s] = {m: measure(K.cell_path(m, t, s), set(wins[m])) for m in K.MODEL}

    delta = {}
    for t in CELLS:
        delta[t] = {}
        for m in K.MODEL:
            a, b = result[t]["none"][m], result[t]["slot"][m]
            delta[t][m] = {"hit1_slot_dikurangi_none": round(b["hit1"] - a["hit1"], 4),
                             "rival1_slot_dikurangi_none": round(b["rival1"] - a["rival1"], 4),
                             "mrr10_slot_dikurangi_none": round(b["mrr10"] - a["mrr10"], 4)}

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": "0a7b2e4de856",
        "_apa_ini": ("Hit@1, Rival@1, and MRR@10 of the asker's gold for each channel cell, six models, two "
                     "tracks."),
        "definisi": {
            "hit1": "share of rows whose top rank is the asker's gold",
            "rival1": "share of rows whose top rank is a gold of another profile in the group",
            "mrr10": "mean of 1/rank of the asker's gold when it is in the top ten, zero below",
            "populasi": "comparable rows, 96 paired and 1,621 broad, the same as the channel table",
        },
        "gerbang": "every cell reproduces the win counts printed in the paper before it is measured",
        "zerank2": "official vendor path",
        "hasil": result,
        "selisih_slot_lawan_none": delta,
        "laju_flip_top1_sel_slot": flip,
        "laju_flip_top1_sel_none": flip_none,
        "catatan_flip": ("flip_rate.json gives this rate for the two Qwen3 models only (0.1538 and 0.2115); here "
                         "it covers all six, and the gate checks that those two values reproduce. On the paired "
                         "track it is computed over 52 pairs, as in flip_rate.json."),
        "label_source": "deterministic (gold matching on frozen scores, no rater)",
    }
    OUT.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    MANIFEST.write_text(json.dumps({
        "run_id": "asker_hit_mrr_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/3_slot_changes/asker_hit_mrr.py",
        "output": {"path": "results/tables/asker_hit_mrr.json", "sha256": K.sha(OUT)},
        "label_source": "deterministic",
    }, indent=1))

    for t, cells in CELLS.items():
        print(f"\n{t}  (Hit@1 / Rival@1 / MRR@10)")
        print("  " + f"{'model':<22}" + "".join(f"{s:>22}" for s in cells))
        for m in K.MODEL:
            print("  " + f"{m:<22}" + "".join(
                f"{result[t][s][m]['hit1']:>8.3f}{result[t][s][m]['rival1']:>7.3f}{result[t][s][m]['mrr10']:>7.3f}"
                for s in cells))
    print("\ntop-1 flip rate in the slot cell (none cell in brackets)")
    for t in CELLS:
        print(f"  {t:<7}" + "".join(f"{m.split('-')[-1][:6]:>9}={flip[t][m]['laju']:.3f}({flip_none[t][m]['laju']:.2f})"
                                    for m in K.MODEL))
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
