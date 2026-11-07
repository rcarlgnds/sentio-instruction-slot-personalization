import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import channel_cells as K  # noqa: E402

OUT = K.TABLE / "gpu_cost.json"
MANIFEST = K.RAW / "gpu_cost.manifest.json"
CELLS = {"paired": ["none", "slot", "text", "both", "wrong", "sem", "prog"],
       "xpair": ["none", "slot", "both", "wrong"]}
CORE = {"paired": ["none", "slot", "text", "both"], "xpair": ["none", "slot", "both"]}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    board, source = {}, {}
    for track, cells in CELLS.items():
        board[track] = {}
        for m in K.MODEL:
            rows = {}
            for s in cells:
                p = K.cell_path(m, track, s)
                man = p.with_name(p.stem + ".manifest.json")
                if not man.exists():
                    rows[s] = {"durasi_menit": None, "vram_puncak_mib": None, "catatan": "no manifest"}
                    continue
                d = json.loads(man.read_text())
                dur = d.get("durasi_s")
                rows[s] = {"durasi_menit": round(dur / 60, 1) if dur is not None else None,
                            "vram_puncak_mib": d.get("vram_peak_mib"),
                            "presisi": d.get("dtype")}
                source[man.name] = sha(man)
            measured = [v["durasi_menit"] for v in rows.values() if v["durasi_menit"] is not None]
            core = [rows[s]["durasi_menit"] for s in CORE[track] if rows[s]["durasi_menit"] is not None]
            board[track][m] = {
                "sel": rows,
                "median_menit_per_sel": round(statistics.median(measured), 1) if measured else None,
                "vram_puncak_maks_mib": max((v["vram_puncak_mib"] or 0) for v in rows.values()) or None,
                "total_jam_sel_inti_terukur": round(sum(core) / 60, 2),
                "sel_inti_terukur": f"{len(core)} of {len(CORE[track])}",
            }

    total = {}
    for track in CELLS:
        hours = sum(board[track][m]["total_jam_sel_inti_terukur"] for m in K.MODEL)
        total[track] = {"jam_gpu_sel_inti_enam_model": round(hours, 2),
                        "sel_inti": CORE[track]}

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": "0a7b2e4de856",
        "_apa_ini": ("Run time and peak GPU memory of every channel cell, read from the score manifests."),
        "mesin": "one NVIDIA RTX 5080 16 GB that also drives the desktop; details in machine_environment.json",
        "papan": board,
        "total": total,
        "batas": ("the four Qwen3 paired cells without a label (none and slot on 0.6B and 4B) have no "
                  "manifest and so no duration; the total sums only the measured cells, and the number of "
                  "measured cells is recorded in every row"),
        "zerank2": "official vendor path",
        "sumber_manifest_sha256": source,
        "label_source": "deterministic (the durasi_s and vram_peak_mib fields of the score manifests)",
    }
    OUT.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    MANIFEST.write_text(json.dumps({
        "run_id": "gpu_cost_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/8_audits/gpu_cost.py",
        "output": {"path": "results/tables/gpu_cost.json", "sha256": sha(OUT)},
        "label_source": "deterministic",
    }, indent=1))

    for track in CELLS:
        print(f"\n{track}")
        for m in K.MODEL:
            r = board[track][m]
            print(f"  {m:<21} median {r['median_menit_per_sel']:>6} min/cell  peak {r['vram_puncak_maks_mib']:>8} MiB  "
                  f"core cells {r['total_jam_sel_inti_terukur']:>5} h ({r['sel_inti_terukur']})")
        print(f"  total core cells, six models: {total[track]['jam_gpu_sel_inti_enam_model']} h")


if __name__ == "__main__":
    main()
