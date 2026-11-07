import argparse
import fcntl
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
from followir import DATASET, calculate_pmrr, ndcg_at_k, sha  # noqa: E402
from score_arms import Q2, SHARED  # noqa: E402
from architecture_scorers import (MAX_BATCH, NEW_FAMILY, wait_for_free_gpu,      # noqa: E402
                             vram_free_mib)
from score_architectures import PREREG, find_batch, decide_cap, write_status  # noqa: E402

PROMPT_VENDOR_FOLLOWIR = ("Relevance requirement: {instruction}\nGiven a query A and a passage "
                          "B, determine whether the passage contains an answer to the query "
                          "and meets the relevance requirement by providing a prediction of "
                          "either 'Yes' or 'No'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET), default="Core17")
    ap.add_argument("--family", choices=list(NEW_FAMILY), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", default="auto")
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--kandidat", type=int, default=1000)
    ap.add_argument("--gap-mib", type=int, default=500)
    ap.add_argument("--gpu-cap", default="auto")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: the output name gets the suffix _smoke, and it is not reported")
    ap.add_argument("--penempatan-vendor", action="store_true",
                    help="bge-gemma only: write the dataset instruction into a replacement "
                         "prompt that ends with the README's Yes/No request (the documented "
                         "placement), instead of appending it after the default prompt")
    args = ap.parse_args()

    if args.penempatan_vendor and args.family != "gemma":
        raise SystemExit("ABORT: --penempatan-vendor is defined for bge-gemma only.")
    tag = f"{args.dataset}_{args.family}_{Path(args.model).name}"
    if args.penempatan_vendor:
        tag += "_vendor"
    if args.limit:
        tag += "_smoke"
    out_run = Q2 / f"raw/followir_run_{tag}.jsonl"
    out_tab = Q2 / f"tables/followir_{tag}.json"
    if out_tab.exists():
        print(f"SKIP (idempotent): {out_tab.name} exists")
        return
    n_present = 0
    if out_run.exists() and out_run.stat().st_size:
        existing_rows = out_run.read_text().splitlines()
        whole = []
        for b in existing_rows:
            try:
                json.loads(b)
                whole.append(b)
            except json.JSONDecodeError:
                break
        if len(whole) != len(existing_rows):
            print(f"resume: dropping {len(existing_rows) - len(whole)} broken trailing line(s)")
            out_run.write_text("".join(b + "\n" for b in whole))
        n_present = len(whole)

    from datasets import load_dataset
    nm = DATASET[args.dataset]
    print(f"loading {nm} ...", flush=True)
    queries = list(load_dataset(nm, "queries", split="test"))
    if args.limit:
        queries = queries[:args.limit]
    corpus = {r["id"]: ((r.get("title") or "") + "\n" + (r.get("text") or "")).strip()
              for r in load_dataset(nm, "corpus", split="test")}
    top = {r["query-id"]: r["corpus-ids"] for r in load_dataset(nm, "top_ranked", split="test")}
    qrels = {}
    for r in load_dataset(nm, "qrels", split="test"):
        qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
    diff = {r["query-id"]: r["corpus-ids"] for r in load_dataset(nm, "qrel_diff", split="qrel_diff")}
    print(f"  queries {len(queries)}  corpus {len(corpus)}  changed topics {len(diff)}", flush=True)

    key = open(Q2 / "raw/.gpu.lock", "w")
    print("waiting for the GPU lock ...", flush=True)
    fcntl.flock(key, fcntl.LOCK_EX)
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
    status_name = f"followir_{tag}"
    write_status(nama=status_name, keadaan="memuat", model=args.model, sel="core17",
                 n_selesai=n_present, n_total=len(queries))
    t_load = time.perf_counter()
    scorer = NEW_FAMILY[args.family](args.model, args.dtype, cap)
    load_time = time.perf_counter() - t_load
    print(f"loaded in {load_time:.0f}s; free {vram_free_mib()} MiB", flush=True)

    def prepare_population(q):
        candidates = top.get(q["id"], [])[:args.kandidat]
        doc = [corpus.get(c, "")[:4000] for c in candidates]
        inst = q.get("instruction") or None
        if args.penempatan_vendor and inst:
            scorer.set_full_prompt(PROMPT_VENDOR_FOLLOWIR.format(instruction=inst.strip()))
        else:
            scorer.set_instruction(inst)
        return candidates, doc, q["text"]

    if args.batch == "auto":
        k0, d0, q0 = prepare_population(queries[0])
        batch, free_after_probe = find_batch(scorer, q0, d0, args.max_len, args.gap_mib,
                                        MAX_BATCH.get(args.family, 2))
        print(f"batch probe: batch {batch}, free after probe {free_after_probe} MiB", flush=True)
    else:
        batch = int(args.batch)
    scorer.n_dokumen_terpotong = 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    mode = "a" if n_present else "w"
    with out_run.open(mode) as f:
        for i, q in enumerate(queries):
            if i < n_present:
                continue
            free_now = vram_free_mib()
            if free_now < args.gap_mib:
                write_status(nama=status_name, keadaan="dibatalkan_vram", n_selesai=i,
                             n_total=len(queries), vram_free_mib=free_now)
                raise SystemExit(f"ABORT VRAM DURING RUN: {free_now} MiB at query {i + 1}")
            candidates, doc, query = prepare_population(q)
            t1 = time.perf_counter()
            score = scorer.score(query, doc, batch, args.max_len)
            f.write(json.dumps({"query_id": q["id"],
                                "scores": {c: round(s, 6) for c, s in zip(candidates, score)},
                                "latency_s": round(time.perf_counter() - t1, 3)}) + "\n")
            f.flush()
            torch.cuda.empty_cache()
            write_status(nama=status_name, keadaan="berjalan", model=args.model, sel="core17",
                         n_selesai=i + 1, n_total=len(queries),
                         elapsed_s=round(time.perf_counter() - t0, 1), batch=batch,
                         vram_peak_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                         vram_free_mib=free_now,
                         n_dokumen_terpotong=scorer.n_dokumen_terpotong)
            print(f"  {i + 1}/{len(queries)} {time.perf_counter() - t0:.0f}s "
                  f"peak={torch.cuda.max_memory_allocated() / 2**20:.0f}MiB", flush=True)

    runs = {}
    for l in out_run.open():
        r = json.loads(l)
        runs[r["query_id"]] = r["scores"]
    og_run = {k: v for k, v in runs.items() if k.endswith("-og")}
    ch_run = {k: v for k, v in runs.items() if k.endswith("-changed")}
    pmrr, per_qid = calculate_pmrr(og_run, ch_run, diff)
    nd = {"og": [], "changed": []}
    for qid, sk in runs.items():
        rank = [c for c, _ in sorted(sk.items(), key=lambda x: (x[1], x[0]), reverse=True)]
        nd["og" if qid.endswith("-og") else "changed"].append(ndcg_at_k(rank, qrels.get(qid, {})))
    lock = json.loads((SHARED / "config_lock.json").read_text())
    result = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "praregistrasi": PREREG,
        "_apa_ini": ("External positive control for bge-gemma, ctxl-6B, and FollowIR-7B, with the same "
                     "protocol as followir.py (FollowIR, NAACL 2025)."),
        "dataset": nm, "model": args.model, "keluarga": args.family,
        "dtype": args.dtype, "max_len": args.max_len, "kandidat_per_kueri": args.kandidat,
        "batch_dipakai": batch, "n_dokumen_terpotong": scorer.n_dokumen_terpotong,
        "cap_gpu": {"mib": cap, "keputusan": cap_reason, "offload": scorer.offload},
        "n_kueri": len(queries), "n_topik_berubah": len(diff),
        "p_MRR": round(pmrr, 6) if pmrr is not None else None,
        "p_MRR_per_topik": {k: round(v, 6) for k, v in per_qid.items()},
        "ndcg5_og": round(sum(nd["og"]) / len(nd["og"]), 6) if nd["og"] else None,
        "ndcg5_changed": round(sum(nd["changed"]) / len(nd["changed"]), 6) if nd["changed"] else None,
        "implementasi_metrik": "imported unchanged from followir.py (adapted from MTEB, see NOTICE.md)",
        "kontrak_inferensi": scorer.template_info,
        "penempatan_instruksi": (
            {"mode": "vendor", "template": PROMPT_VENDOR_FOLLOWIR,
             "catatan": "the dataset instruction is written into a replacement prompt that ends with the Yes/No "
                         "sentence of the README; queries without an instruction use the default prompt"}
            if args.penempatan_vendor else {"mode": "family slot (appended after the default prompt for gemma)"}),
        "versi_pustaka": {"transformers": transformers.__version__, "torch": torch.__version__},
        "durasi_s": round(time.perf_counter() - t0, 1), "durasi_muat_s": round(load_time, 1),
        "vram_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "smoke": bool(args.limit),
        "label_source": "deterministic (model scores)",
    }
    out_tab.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (Q2 / f"raw/followir_{tag}.manifest.json").write_text(json.dumps({
        "run_id": f"followir_{tag}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/2_scoring/followir_architectures.py",
        "praregistrasi": PREREG,
        "perintah_persis": " ".join(sys.argv),
        "config_version": lock["config_version"],
        "status_arm": "post hoc, external positive control, cross-architecture test",
        "durasi_s": result["durasi_s"], "vram_peak_mib": result["vram_peak_mib"],
        "output": {"path": f"results/tables/followir_{tag}.json", "sha256": sha(out_tab),
                   "run": f"results/raw/followir_run_{tag}.jsonl", "run_sha256": sha(out_run)},
    }, indent=2, ensure_ascii=False))
    write_status(nama=status_name, keadaan="selesai", n_selesai=len(queries),
                 n_total=len(queries), elapsed_s=result["durasi_s"], batch=batch,
                 vram_peak_mib=result["vram_peak_mib"])
    print(f"\np-MRR {result['p_MRR']}  nDCG@5 og {result['ndcg5_og']} changed {result['ndcg5_changed']}")
    print(f"written: {out_tab}")


if __name__ == "__main__":
    main()
