"""Scores one cell for FollowIR-7B, ctxl-6B, bge-gemma, or zerank-2. Needs a GPU.

--cell sets the input: the profile in the model's own slot (slot), a source label in the passage (text),
or both. The rows have the same format as score_arms.py, which also holds the shared helpers.

    python score_architectures.py --family followir --model jhu-clsp/FollowIR-7B --cell slot
"""

import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
from score_arms import (ROOT, Q2, SHARED, TASK_DEFAULT, Zerank2Scorer,  # noqa: E402
                          seal_gate, profile_instruction, profile_instruction_id,
                          manifest_output_path,
                          label_program_style, label_program_en,
                          label_semester_style, profile_instruction_name_en,
                          source_label,
                          wrong_source_label, load_metadata_map,
                          load_global_profiles, sha)
from architecture_scorers import (MAX_BATCH, NEW_FAMILY, wait_for_free_gpu,      # noqa: E402
                             vram_free_mib)

from label_map import candidate_meta, load_id_label_map, source_summary  # noqa: E402
from model_pins import revision_for  # noqa: E402
from revision_labels import (revision_label_rules, program_only_instruction,  # noqa: E402
                         semester_only_instruction, random_label, uniform_label,
                         wrong_both_source_label, wrong_semester_source_label,
                         fixed_profile, observed_semesters)

RAW = Q2 / "raw"
POOL_DEFAULT = ROOT / "study/retrieval/results/raw/pools_paired.jsonl"
STATUS = RAW / "arsitektur_status_sel.json"
PREREG = "PRAREGISTRASI_LINTAS_ARSITEKTUR.md"

CELLS = {
    "none": {"inst": "none", "docmeta": False, "salah": False},
    "none_generic": {"inst": "generic", "docmeta": False, "salah": False},
    "bare": {"inst": "bare", "docmeta": False, "salah": False},
    "slot": {"inst": "profile", "docmeta": False, "salah": False},
    "text": {"inst": "none", "docmeta": True, "salah": False},
    "both": {"inst": "profile", "docmeta": True, "salah": False},
    "wrong": {"inst": "profile", "docmeta": True, "salah": True},
    "both_program": {"inst": "profile", "docmeta": True, "salah": False, "field": "program"},
    "both_semester": {"inst": "profile", "docmeta": True, "salah": False, "field": "semester"},
    "slot_vendor": {"inst": "vendor", "docmeta": False, "salah": False},
    "slot_id": {"inst": "profile_id", "docmeta": False, "salah": False},
    "label_nama": {"inst": "profile", "docmeta": True, "salah": False, "gaya": "nama"},
    "label_alami": {"inst": "profile", "docmeta": True, "salah": False, "gaya": "alami"},
    "label_opak": {"inst": "profile", "docmeta": True, "salah": False, "gaya": "opak"},
    "label_en": {"inst": "profile_en", "docmeta": True, "salah": False, "gaya_en": "en"},
    "label_id_en": {"inst": "profile_en", "docmeta": True, "salah": False, "gaya_en": "id"},
    "label_kode_en": {"inst": "profile_en", "docmeta": True, "salah": False, "gaya_en": "kode"},
    "sem_kata": {"inst": "profile", "docmeta": True, "salah": False, "gaya_sem": "kata"},
    "sem_opak": {"inst": "profile", "docmeta": True, "salah": False, "gaya_sem": "opak"},
    "slot_program": {"inst": "profile_program", "docmeta": False, "salah": False},
    "slot_semester": {"inst": "profile_semester", "docmeta": False, "salah": False},
    "slot_semtetap1": {"inst": "profile", "docmeta": False, "salah": False, "tetap": {"semester": 1}},
    "slot_semtetap2": {"inst": "profile", "docmeta": False, "salah": False, "tetap": {"semester": 2}},
    "slot_prodtetap_csci": {"inst": "profile", "docmeta": False, "salah": False, "tetap": {"prodi": "CSCI"}},
    "slot_prodtetap_stat": {"inst": "profile", "docmeta": False, "salah": False, "tetap": {"prodi": "STAT"}},
    "both_semsalah": {"inst": "profile", "docmeta": True, "salah": False, "salah_sem": True},
    "both_duasalah": {"inst": "profile", "docmeta": True, "salah": True, "salah_sem": True},
    "label_acak": {"inst": "profile", "docmeta": True, "salah": False, "acak": True},
    "label_seragam": {"inst": "profile", "docmeta": True, "salah": False, "seragam": True},
    "both_vendor": {"inst": "vendor", "docmeta": True, "salah": False},
    "wrong_vendor": {"inst": "vendor", "docmeta": True, "salah": True},
}


def vendor_instruction(family: str, p: dict) -> str:
    dept = p.get("department") or p.get("department_name") or "unknown"
    sem = p.get("current_semester") or p.get("semester") or "?"
    if family == "followir":
        return (f"A relevant document would come from a textbook assigned to the {dept} "
                f"degree program and would suit a student in semester {sem}, rather than "
                f"from a textbook of another degree program.")
    if family == "ctxl":
        return (f"Prioritize passages from the {dept} degree program's own textbooks at a "
                f"level suitable for a semester {sem} student")
    if family == "gemma":
        return (f"Given a query A from a {dept} student in semester {sem} and a passage B, "
                f"determine whether the passage comes from that student's own degree program "
                f"and contains an answer to the query at a level appropriate for semester {sem} "
                f"by providing a prediction of either 'Yes' or 'No'.")
    raise SystemExit(f"slot_vendor is defined only for followir, ctxl, gemma, not {family}")


def write_status(**k):
    k["diperbarui"] = datetime.now(timezone.utc).isoformat()
    STATUS.write_text(json.dumps(k, indent=1))


def bf16_weights_mib(name: str):
    try:
        from huggingface_hub import snapshot_download
        d = Path(snapshot_download(name, revision=revision_for(name), allow_patterns=["config.json", "*.index.json"]))
        idx = json.loads((d / "model.safetensors.index.json").read_text())
        cfg = json.loads((d / "config.json").read_text())
        per = {"float32": 4, "bfloat16": 2, "float16": 2}.get(cfg.get("torch_dtype"), 2)
        return idx["metadata"]["total_size"] / per * 2 / 2**20
    except Exception:
        return None


def decide_cap(family: str, model: str, gap: int, mode: str):
    if mode == "off":
        return None, "off"
    if mode not in ("auto",):
        return int(mode), "manual"
    free = vram_free_mib()
    weight = bf16_weights_mib(model)
    spare = 400
    if weight is None:
        return None, "auto: weight size unreadable, loaded in full"
    if free - spare - weight >= gap:
        return None, (f"auto: free {free} - reserve {spare} - weights {weight:.0f} >= gap {gap}"
                      f", loaded fully on the GPU")
    cap = int(free - gap - 700)
    return cap, (f"auto: weights {weight:.0f} MiB do not fit with a gap of {gap} (free {free}"
                 f"); GPU cap {cap} MiB, the rest of the weights in RAM")


def find_batch(scorer, query, doc, max_len, gap, limit):
    import torch
    longest_doc = sorted(doc, key=len, reverse=True)
    for b in [x for x in (8, 4, 2, 1) if x <= limit]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            scorer.score(query, longest_doc[:max(1, min(len(doc), b * 2))], b, max_len)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        free = vram_free_mib()
        torch.cuda.empty_cache()
        if free >= gap:
            return b, free
    return 1, vram_free_mib()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(NEW_FAMILY) + ["zerank2"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", choices=list(CELLS), required=True)
    ap.add_argument("--pools", default=str(POOL_DEFAULT))
    ap.add_argument("--tag", default="paired")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--batch", default="auto")
    ap.add_argument("--gap-mib", type=int, default=500,
                    help="minimum free VRAM kept during the run (owner's setting for this run)")
    ap.add_argument("--gpu-cap", default="auto", help="auto | off | <MiB>")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: the output name gets the suffix _smoke, and no table reads it")
    ap.add_argument("--peta-label", choices=["id", "teks"], default="id",
                    help="source of the label metadata: the candidate's own record (default) or a match on the "
                         "first 180 characters of its text")
    ap.add_argument("--out-dir", default=None,
                    help="output folder for the jsonl and manifest; default raw/")
    args = ap.parse_args()

    cell = CELLS[args.cell]
    if args.cell == "none_generic" and args.family != "ctxl":
        raise SystemExit("ABORT: none_generic is pre-registered for ctxl only.")
    if args.cell == "bare" and args.family != "zerank2":
        raise SystemExit("ABORT: bare is pre-registered for zerank2 only.")
    if args.family == "zerank2" and args.cell != "bare":
        raise SystemExit("ABORT: every other zerank2 cell is frozen; this script "
                         "does not re-score frozen cells.")

    pools = [json.loads(l) for l in Path(args.pools).open() if l.strip()]
    seal_gate([r["query_id"] for r in pools])
    if args.limit:
        pools = pools[:args.limit]
    global_profile = load_global_profiles() if cell["inst"] in (
        "profile", "vendor", "profile_id", "profile_en",
        "profile_program", "profile_semester") else {}
    meta_map = ((load_id_label_map() if args.peta_label == "id"
                  else load_metadata_map()) if cell["docmeta"] else {})

    def find_meta(c):
        return (candidate_meta(meta_map, c) if args.peta_label == "id"
                else meta_map.get((c["text"] or "")[:180].strip()))
    all_depts = sorted({(m or {}).get("department") for m in meta_map.values()
                         if (m or {}).get("department")})
    if cell["salah"] and len(all_depts) < 2:
        raise SystemExit("ABORT: fewer than two programs in the metadata map.")
    all_sems = observed_semesters(meta_map) if cell["docmeta"] else []
    if (cell.get("salah_sem") or cell.get("acak")) and len(all_sems) < 2:
        raise SystemExit("ABORT: fewer than two semester values in the metadata map.")
    if cell.get("tetap"):
        program_pool = {(r.get("profile") or {}).get("department_name")
                      or (r.get("profile") or {}).get("department") for r in pools}
        if not program_pool <= {"CSCI", "STAT"}:
            raise SystemExit("ABORT: fixed-field cells are defined for the paired track "
                             f"only; this pool holds programs {sorted(map(str, program_pool))}.")
    n_r1 = {"semsalah": 0, "duasalah": 0, "acak": 0, "seragam": 0}

    lock = json.loads((SHARED / "config_lock.json").read_text())
    out_name = (f"scores_{args.family}_{Path(args.model).name}_{args.tag}_{args.cell}"
                f"{'_smoke' if args.limit else ''}")
    raw_out = Path(args.out_dir) if args.out_dir else RAW
    raw_out.mkdir(parents=True, exist_ok=True)
    global STATUS
    STATUS = raw_out / "arsitektur_status_sel.json"
    out_path = raw_out / f"{out_name}.jsonl"
    man_path = raw_out / f"{out_name}.manifest.json"
    if man_path.exists() and out_path.exists():
        n_present = sum(1 for _ in out_path.open())
        if n_present >= len(pools):
            print(f"SKIP (idempotent): {out_path.name} already complete, {n_present} rows")
            return
    mode = "w"
    if out_path.exists() and out_path.stat().st_size > 0:
        existing_rows = out_path.read_text().splitlines()
        whole = []
        for b in existing_rows:
            try:
                json.loads(b)
                whole.append(b)
            except json.JSONDecodeError:
                break
        if len(whole) != len(existing_rows):
            print(f"resume: dropping {len(existing_rows) - len(whole)} broken trailing line(s)")
            out_path.write_text("".join(b + "\n" for b in whole))
        n_present = len(whole)
        print(f"resuming after {n_present} existing rows")
        pools = pools[n_present:]
        mode = "a"

    key = open(RAW / ".gpu.lock", "w")
    print("waiting for the GPU lock ...", flush=True)
    fcntl.flock(key, fcntl.LOCK_EX)
    key.write(f"{os.getpid()} {out_name}\n")
    key.flush()
    print("GPU lock acquired", flush=True)
    pending = wait_for_free_gpu()
    if pending:
        print(f"waited {pending}s for the previous process to release the GPU", flush=True)

    free_before = vram_free_mib()
    if free_before < args.gap_mib + 1000:
        raise SystemExit(f"ABORT VRAM: only {free_before} MiB free before loading.")
    cap, cap_reason = decide_cap(args.family, args.model, args.gap_mib, args.gpu_cap)
    print(f"free before load {free_before} MiB; cap decision: {cap_reason}", flush=True)

    import torch
    import transformers
    write_status(nama=out_name, keadaan="memuat", model=args.model, sel=args.cell,
                 n_selesai=0, n_total=len(pools))
    t_load = time.perf_counter()
    if args.family == "zerank2":
        scorer = Zerank2Scorer(args.model, args.dtype)
        scorer.offload = None
        scorer.n_dokumen_terpotong = None
    else:
        scorer = NEW_FAMILY[args.family](args.model, args.dtype, cap)
    load_time = time.perf_counter() - t_load
    print(f"loaded in {load_time:.0f}s; process VRAM "
          f"{torch.cuda.memory_allocated() / 2**20:.0f} MiB; free {vram_free_mib()} MiB",
          flush=True)

    def document_for(row):
        if not cell["docmeta"]:
            return [c["text"][:4000] for c in row["cands"]], 0, 0
        doc, unmapped, flipped = [], 0, 0
        p_rows = row.get("profile") or global_profile.get(row["query_id"], {})
        if cell.get("seragam") and not (p_rows.get("department") or p_rows.get("department_name")):
            raise SystemExit(f"ABORT: row {row['query_id']} has no profile, the uniform "
                             "label cannot be built.")
        for c in row["cands"]:
            m = find_meta(c)
            if m is None:
                unmapped += 1
                pre = ""
            elif cell.get("acak"):
                pre = random_label(c["text"] or "", all_depts, all_sems) + "\n\n"
                n_r1["acak"] += 1
            elif cell.get("seragam"):
                pre = uniform_label(p_rows) + "\n\n"
                n_r1["seragam"] += 1
            elif cell["salah"] and cell.get("salah_sem"):
                pre = wrong_both_source_label(m, c["text"] or "", all_depts, all_sems) + "\n\n"
                n_r1["duasalah"] += 1
                flipped += 1
            elif cell.get("salah_sem"):
                pre = wrong_semester_source_label(m, c["text"] or "", all_sems) + "\n\n"
                n_r1["semsalah"] += 1
            elif cell["salah"]:
                pre = wrong_source_label(m, c["text"] or "", all_depts) + "\n\n"
                flipped += 1
            elif cell.get("gaya"):
                pre = label_program_style(m, cell["gaya"]) + "\n\n"
            elif cell.get("gaya_en") == "en":
                pre = label_program_en(m) + "\n\n"
            elif cell.get("gaya_en") == "id":
                pre = label_program_style(m, "nama") + "\n\n"
            elif cell.get("gaya_en") == "kode":
                pre = source_label(m, "program") + "\n\n"
            elif cell.get("gaya_sem"):
                pre = label_semester_style(m, cell["gaya_sem"]) + "\n\n"
            else:
                pre = source_label(m, cell.get("field", "both")) + "\n\n"
            doc.append(pre + c["text"][:4000])
        return doc, unmapped, flipped

    def instruction_for(row):
        if cell["inst"] == "profile":
            p = row.get("profile") or global_profile.get(row["query_id"], {})
            if cell.get("tetap"):
                p = fixed_profile(p, program=cell["tetap"].get("prodi"),
                                 semester=cell["tetap"].get("semester"))
            return profile_instruction(p)
        if cell["inst"] == "profile_program":
            p = row.get("profile") or global_profile.get(row["query_id"], {})
            return program_only_instruction(p)
        if cell["inst"] == "profile_semester":
            p = row.get("profile") or global_profile.get(row["query_id"], {})
            return semester_only_instruction(p)
        if cell["inst"] == "vendor":
            p = row.get("profile") or global_profile.get(row["query_id"], {})
            return vendor_instruction(args.family, p)
        if cell["inst"] == "profile_en":
            p = row.get("profile") or global_profile.get(row["query_id"], {})
            return profile_instruction_name_en(p)
        if cell["inst"] == "profile_id":
            p = row.get("profile") or global_profile.get(row["query_id"], {})
            return profile_instruction_id(p)
        if cell["inst"] == "generic":
            return TASK_DEFAULT
        return None

    def query_for(row, inst):
        if args.family == "zerank2":
            return row["query"]
        if cell["inst"] == "vendor" and args.family == "gemma":
            scorer.set_full_prompt(inst)
        else:
            scorer.set_instruction(inst)
        return row["query"]

    if args.batch == "auto":
        r0 = pools[0]
        d0, _, _ = document_for(r0)
        batch, free_after_probe = find_batch(scorer, query_for(r0, instruction_for(r0)),
                                        d0, args.max_len, args.gap_mib,
                                        MAX_BATCH.get(args.family, 2))
        print(f"batch probe: batch {batch}, free after probe {free_after_probe} MiB", flush=True)
    else:
        batch = int(args.batch)
    if getattr(scorer, "n_dokumen_terpotong", None) is not None:
        scorer.n_dokumen_terpotong = 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    n = unmapped_total = total_flipped = 0
    lat = []
    with out_path.open(mode) as f:
        for row in pools:
            free_now = vram_free_mib()
            if free_now < args.gap_mib:
                write_status(nama=out_name, keadaan="dibatalkan_vram", n_selesai=n,
                             n_total=len(pools), vram_free_mib=free_now)
                raise SystemExit(f"ABORT VRAM DURING RUN: {free_now} MiB free < "
                                 f"{args.gap_mib} at row {n + 1}. The output file is "
                                 f"resumable.")
            doc, tp, db = document_for(row)
            unmapped_total += tp
            total_flipped += db
            query = query_for(row, instruction_for(row))
            t1 = time.perf_counter()
            score = scorer.score(query, doc, batch, args.max_len)
            dt = time.perf_counter() - t1
            lat.append(dt)
            order = sorted(zip((c["id"] for c in row["cands"]), score),
                          key=lambda x: (-x[1], x[0]))
            f.write(json.dumps({
                "query_id": row["query_id"], "gold": row["gold"],
                "counterpart": row.get("counterpart"),
                "ranking": [i for i, _ in order],
                "scores": [round(s, 6) for _, s in order],
                "latency_s": round(dt, 4),
            }) + "\n")
            f.flush()
            n += 1
            torch.cuda.empty_cache()
            write_status(nama=out_name, keadaan="berjalan", model=args.model, sel=args.cell,
                         n_selesai=n, n_total=len(pools),
                         elapsed_s=round(time.perf_counter() - t0, 1), batch=batch,
                         vram_peak_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                         vram_free_mib=free_now,
                         n_dokumen_terpotong=getattr(scorer, "n_dokumen_terpotong", None))
            if n % 10 == 0:
                print(f"  {n}/{len(pools)} {time.perf_counter() - t0:.0f}s "
                      f"peak={torch.cuda.max_memory_allocated() / 2**20:.0f}MiB "
                      f"free={free_now}MiB", flush=True)

    duration = time.perf_counter() - t0
    all_lat = [json.loads(l)["latency_s"] for l in out_path.open()]
    lat_s = sorted(all_lat)
    meta = {
        "run_id": out_name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/2_scoring/score_architectures.py",
        "praregistrasi": PREREG,
        "perintah_persis": " ".join(["python score_architectures.py"] + sys.argv[1:]),
        "config_version": lock["config_version"],
        "status_arm": "post hoc cross-architecture test, outside the pre-registered family of eight",
        "model": args.model, "keluarga": args.family, "sel": args.cell,
        "definisi_sel": cell, "dtype": args.dtype, "max_len": args.max_len,
        "pemotongan": "documents are cut to 4,000 characters, then the document tokens are cut so the template "
                      "frame is never cut",
        "n_dokumen_terpotong": getattr(scorer, "n_dokumen_terpotong", None),
        "batch_dipakai": batch,
        "batas_sisa_vram_mib": args.gap_mib,
        "cap_gpu": {"mib": cap, "keputusan": cap_reason, "offload": getattr(scorer, "offload", None)},
        "instruksi_default_verbatim": TASK_DEFAULT,
        "instruksi_profil_kerangka": profile_instruction({"department": "<DEPT>", "current_semester": "<N>"}),
        "instruksi_profil_id_kerangka": (profile_instruction_id({"department": "<DEPT>", "current_semester": "<N>"})
                                         if cell["inst"] == "profile_id" else None),
        "instruksi_vendor_kerangka": (vendor_instruction(args.family, {"department": "<DEPT>", "current_semester": "<N>"})
                                      if cell["inst"] == "vendor" else None),
        "instruksi_vendor_penempatan": ({"followir": "narrative appended to the query text, the README convention",
                                         "ctxl": "a short custom instruction after the query, the README convention",
                                         "gemma": "replaces the default prompt entirely, in the README prompt form"}[args.family]
                                        if cell["inst"] == "vendor" else None),
        "format_label_program_gaya": (label_program_style({"department": "STAT"}, cell["gaya"])
                                      if cell.get("gaya") else None),
        "format_label_sumber": (source_label({"department": "<DEPT>", "semester": "<N>"},
                                             cell.get("field", "both"))
                                if cell["docmeta"] else None),
        "kontrol_label_salah": ({"manipulasi": ("department and semester both replaced"
                                                if cell.get("salah_sem") else
                                                "only the department field replaced; the semester stays right"),
                                 "pemilihan": "deterministic: sha256('dept_salah_v1|' + text[:180])",
                                 "kolam_prodi": all_depts, "n_label_dibalik": total_flipped}
                                if cell["salah"] else None),
        "r1_ijies": {
            "profil_tetap": cell.get("tetap"),
            "instruksi_kerangka": (
                program_only_instruction({"department": "<DEPT>"})
                if cell["inst"] == "profile_program" else
                semester_only_instruction({"current_semester": "<N>"})
                if cell["inst"] == "profile_semester" else None),
            "aturan_label": (revision_label_rules()
                             if any(cell.get(k) for k in ("salah_sem", "acak", "seragam")) else None),
            "kolam_semester": all_sems if cell["docmeta"] else None,
            "kolam_prodi": all_depts if cell["docmeta"] else None,
            "n_label": n_r1,
        },
        "n_kandidat_tak_terpetakan": unmapped_total,
        "peta_label_sumber": (source_summary() if args.peta_label == "id"
                              else {"peta_label": "teks"}),
        "kontrak_inferensi": scorer.template_info,
        "versi_pustaka": {"transformers": transformers.__version__, "torch": torch.__version__},
        "n": n, "durasi_s": round(duration, 1), "durasi_muat_s": round(load_time, 1),
        "latensi_mean_s": round(sum(all_lat) / len(all_lat), 4) if all_lat else None,
        "latensi_p95_s": round(lat_s[int(0.95 * len(lat_s)) - 1], 4) if lat_s else None,
        "vram_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "vram_bebas_sebelum_muat_mib": free_before,
        "smoke": bool(args.limit),
        "sumber": {"pools": sha(Path(args.pools))},
        "output": {"path": manifest_output_path(out_path), "sha256": sha(out_path)},
        "label_source": "deterministic (model scores; temperature does not apply)",
    }
    man_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    write_status(nama=out_name, keadaan="selesai", n_selesai=n, n_total=n,
                 elapsed_s=round(duration, 1), batch=batch, vram_peak_mib=meta["vram_peak_mib"],
                 n_dokumen_terpotong=meta["n_dokumen_terpotong"])
    print(json.dumps({k: meta[k] for k in ("n", "durasi_s", "batch_dipakai",
                                           "vram_peak_mib", "n_dokumen_terpotong")}))
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
