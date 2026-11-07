import os
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])

Q2 = ROOT / "study/channel"
RAW = Q2 / "results/raw"
TAB = Q2 / "results/tables"
PP = ROOT / "study/retrieval/results/raw/pools_paired.jsonl"
N_PAIRS = 10

sys.path.insert(0, str(Q2 / "scripts"))

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
from label_map import candidate_meta, load_id_label_map  # noqa: E402


def profile_sentence(p):
    return (f"a student of the {p.get('department_name')} degree program, "
            f"semester {p.get('current_semester')}")


def combinations(family):
    from score_arms import TASK_DEFAULT

    def inst_profile(p):
        return ("Given a student profile and a query, retrieve the passage "
                "most relevant FOR THIS STUDENT. The student is "
                + profile_sentence(p) + ". Prefer material from this student's "
                "own degree program and semester.")

    def inst_imperative(p):
        return ("You MUST rank passages for " + profile_sentence(p) + ". RULE: "
                "when two passages are similarly relevant, ALWAYS rank the one "
                "from the " + str(p.get("department_name")) + " program's own "
                "textbook higher. This rule overrides general relevance.")

    def inst_json(p):
        return ("Rank for this user. User profile JSON: "
                + json.dumps({"degree_program": p.get("department_name"),
                              "semester": p.get("current_semester")})
                + ". Prefer passages matching the profile.")

    def plain_q(q, p):
        return q

    def q_profile(q, p):
        return ("[Asking student: " + profile_sentence(p) + "]\n" + q)

    K = [
        ("K1_default",            None,            plain_q,   False),
        ("K2_slot_instruksi",     inst_profile,    plain_q,   False),
        ("K3_profil_di_kueri",    None,            q_profile, False),
        ("K4_label_di_dokumen",   None,            plain_q,   True),
        ("K5_slot_plus_dokumen",  inst_profile,    plain_q,   True),
        ("K6_frasa_imperatif",    inst_imperative, plain_q,   False),
    ]
    if family == "qwen3rr":
        K.append(("K7_profil_json", inst_json, plain_q, False))
    else:
        def vmem(q, p):
            return ("<user_memories>\nThe user is " + profile_sentence(p)
                    + ".\nThe user prefers material from their own degree "
                    "program's textbook.\n</user_memories>\n" + q)
        K.append(("K7_memori_vendor", None, vmem, False))
    return K, TASK_DEFAULT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--peta-label", choices=["id", "teks"], default="id",
                    help="source of the label metadata: the candidate's own record (default) or a match on the "
                         "first 180 characters of its text")
    args = ap.parse_args()

    from score_arms import FAMILY, load_metadata_map, source_label

    import fcntl
    key = open(RAW / ".gpu.lock", "w")
    print("waiting for the GPU lock ...", flush=True)
    fcntl.flock(key, fcntl.LOCK_EX)
    print("lock acquired", flush=True)

    rows = [json.loads(l) for l in PP.open() if l.strip()]
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    pairs = sorted(by_pair)[:N_PAIRS]
    sample = [r for pid in pairs for r in by_pair[pid]]
    print(f"{len(pairs)} pairs = {len(sample)} rows")

    mapping = (load_id_label_map() if args.peta_label == "id"
            else load_metadata_map())
    K, TASK_DEFAULT = combinations(args.family)
    scorer = FAMILY[args.family](args.model, args.dtype)

    row_results = []
    t0 = time.perf_counter()
    for r in sample:
        gold = set(r["gold"] or [])
        other = set(r["counterpart"] or [])
        prof = r["profile"]
        record = {"query_id": r["query_id"], "pair_id": r["pair_id"],
                 "dept": prof.get("department_name"), "kombinasi": {}}
        for name, f_inst, f_q, use_label in K:
            doc, ids = [], []
            for c in r["cands"]:
                ids.append(c["id"])
                if use_label:
                    m = (candidate_meta(mapping, c) if args.peta_label == "id"
                         else mapping.get((c["text"] or "")[:180].strip()))
                    pre = (source_label(m) + "\n\n") if m else ""
                    doc.append(pre + c["text"][:4000])
                else:
                    doc.append(c["text"][:4000])
            inst = f_inst(prof) if f_inst else TASK_DEFAULT
            query = f_q(r["query"], prof)
            if args.family == "qwen3rr":
                scorer.instruction = inst
            else:
                if f_inst:
                    query = inst + "\n\nQuery: " + query
            score = scorer.score(query, doc, args.batch, args.max_len)
            order = [i for _, i in sorted(zip(score, ids),
                                         key=lambda t: (-t[0], t[1]))]
            pos = {c: i for i, c in enumerate(order)}
            g = min((pos[c] for c in gold if c in pos), default=None)
            o = min((pos[c] for c in other if c in pos), default=None)
            record["kombinasi"][name] = {
                "rank_gold": g, "rank_lawan": o,
                "menang": (g is not None and o is not None and g < o)}
        row_results.append(record)
        print(f"  {r['query_id']} done ({time.perf_counter()-t0:.0f}s)",
              flush=True)

    agg = {}
    for name, *_ in K:
        m = [b["kombinasi"][name]["menang"] for b in row_results
             if b["kombinasi"][name]["rank_gold"] is not None
             and b["kombinasi"][name]["rank_lawan"] is not None]
        rg = [b["kombinasi"][name]["rank_gold"] for b in row_results
              if b["kombinasi"][name]["rank_gold"] is not None]
        agg[name] = {"menang": sum(m), "n_komparabel": len(m),
                     "mean_rank_gold": round(sum(rg) / len(rg), 2)}

    model_name = Path(args.model).name
    out = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "_apa_ini": f"Format sweep: 7 input formats, each on the same {len(pairs)}"
                    f" pairs of the original track. It is a first screening run. The paper reports one "
                    f"result from it: every format that leaves the passage unchanged stays within two rows of "
                    f"the baseline.",
        "model": args.model, "family": args.family, "dtype": args.dtype,
        "dasar_dokumen": {
            "qwen3rr": "official README: Instruction Aware Yes, format <Instruct>/<Query>/<Document>, English "
                       "instructions recommended per scenario",
            "zerank2": "README + chat_template.jinja: plain (query, document), query->system, document->user, "
                       "logit Yes; no separate instruction parameter"},
        "agregat_per_kombinasi": agg,
        "per_sampel": row_results,
    }
    TAB.mkdir(parents=True, exist_ok=True)
    suffix = "_petaid" if args.peta_label == "id" else ""
    p = TAB / f"injection_probe_{args.family}_{model_name}{suffix}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n{'combination':22s} {'wins':>7s} {'mean_rank_gold':>15s}")
    for name, v in agg.items():
        print(f"{name:22s} {v['menang']:>3d}/{v['n_komparabel']:<3d} "
              f"{v['mean_rank_gold']:>15.2f}")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
