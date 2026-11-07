import json
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import revision_analysis as A  # noqa: E402
from revision_statistics import SEED, contrast  # noqa: E402

PREREG = A.ROOT / "result/paper_q2/PRAREGISTRASI_R1_IJIES.md"
FAMILIES = {"slot_vs_none": ("none", "slot"), "both_vs_slot": ("slot", "both"),
            "wrong_vs_both": ("both", "wrong")}
VENDOR = {"bge-gemma": {"slot": "slot_vendor", "both": "both_vendor", "wrong": "wrong_vendor"}}


class TablePart:
    def __init__(self):
        self.skipped = {}

    def __call__(self, name: str, fn, *arg):
        try:
            return fn(*arg)
        except A.MissingInput as e:
            self.skipped[name] = str(e)
            return {"not_computed": str(e)}


def subset_family(cell_source, load, pairs, subset):
    if not subset:
        return {"n_groups": 0}
    res = A.contrast_family_table(load, cell_source, pairs, subset=subset, replacement=VENDOR)
    return {"n_groups": len(subset), "models": res, "summary": A.summarize(res)}


def main_family(cell_source, load, pairs, book):
    res = A.contrast_family_table(load, cell_source, pairs, clusters=book, replacement=VENDOR)
    return {"models": res, "summary": A.summarize(res)}


def gemma_appended(cell_source, load, sa, sb):
    return contrast(load(cell_source("bge-gemma", sa)), load(cell_source("bge-gemma", sb)))


def blind_count(cell_source, load, m, n_groups):
    return {"resolved": sum(load(cell_source(m, "none")).values()), "n_groups": n_groups}


def bound_summary(name: str):
    p = A.TAB / name
    if not p.exists():
        return None
    s = json.loads(p.read_text())["contrasts"]["slot_vs_none"]["summary"]
    return {"table": name, **{k: s[k] for k in ("largest_gain_pp_max_over_models",
                                                "family_wise_largest_gain_pp_max_over_models",
                                                "significant_after_holm_mcnemar")}}


def dits_rows(dits: Path, manifest: Path, track: str, load, id_pool: set) -> dict:
    if not manifest.exists():
        raise A.MissingInput(f"{manifest} missing: DITS on the {track} pool did not finish")
    if not dits.exists():
        raise A.MissingInput(str(dits))
    ids = {json.loads(b)["query_id"] for b in dits.open(encoding="utf-8") if b.strip()}
    if ids != id_pool:
        raise A.MissingInput(f"{dits} covers {len(ids & id_pool)} of the pool's {len(id_pool)} rows "
                             f"and {len(ids - id_pool)} rows outside it")
    wins = load(dits)
    return {"DITS_resolved": sum(wins.values()), "DITS_rows": len(wins)}


def write_table(*, out: Path, raw: Path, name: str, script: str, section_name: str, status: str,
                label_source: str, inputs: dict, content: dict, section: TablePart, result: dict,
                n_groups: int, n_rows: int):
    output = {"created": datetime.now(timezone.utc).isoformat(),
              "script": script, "section": section_name,
              "preregistration": {"path": str(PREREG.relative_to(A.ROOT))},
              "analysis_status": status, "seed": SEED,
              "inputs": inputs,
              "label_source": label_source, **content}
    A.TAB.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    (raw / f"{name}.manifest.json").write_text(json.dumps({
        "run_id": f"{name}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": script, "command": Path(script).name,
        "output": {"path": f"results/tables/{out.name}", "sha256": A.sha(out)},
        "inputs": inputs, "not_computed": sorted(section.skipped), "seed": SEED}, indent=1))
    summary = {"n_groups": n_groups, "n_rows": n_rows}
    for fam, key in (("slot_vs_none", "slot"), ("both_vs_slot", "label")):
        s = result[fam].get("summary")
        if s:
            summary[f"{key}_significant_after_holm"] = s["significant_after_holm_mcnemar"]
            summary[f"{key}_bound_pp"] = s["largest_gain_pp_max_over_models"]
            summary[f"{key}_bound_family_wise_pp"] = s["family_wise_largest_gain_pp_max_over_models"]
        else:
            summary[f"{key}_family"] = "not computed"
    print(json.dumps(summary, indent=1))
    print(f"written: {out}")
    if section.skipped:
        print("parts not computed:\n  " + "\n  ".join(f"{k}: {v}" for k, v in
                                                      sorted(section.skipped.items())))
        raise SystemExit(1)
