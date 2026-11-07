import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import extended_paired_common as U  # noqa: E402
import revision_broad_sample as T1  # noqa: E402


def load_order() -> list:
    if not (U.PAIRS.exists() and U.PAIRS_MANIFEST.exists()):
        raise SystemExit("ABORT: the topic-pair order is missing.")
    m = json.loads(U.PAIRS_MANIFEST.read_text())
    if U.sha(U.PAIRS) != m["output"]["sha256"]:
        raise SystemExit("ABORT: pasangan_topik_p2.jsonl does not match its manifest.")
    return U.read_jsonl(U.PAIRS)


def stored_rounds() -> list:
    return sorted(int(p.name.split("_")[1]) for p in U.BATCH.glob("ronde_*") if p.is_dir())


def next_round(ordering: list, sent: set, group: int, round_no: int) -> dict:
    if group >= U.TARGET_GROUPS:
        return {"stop": f"target reached: {group} groups"}
    for entries in ("fokus", "acct"):
        rest = [r for r in ordering if r["kelompok"] == entries and r["tp_id"] not in sent]
        if rest:
            return {"ronde": round_no + 1, "list": entries, "tp_from": rest[0]["urut"],
                    "tp_to": rest[min(U.N_ROUNDS, len(rest)) - 1]["urut"]}
    return {"stop": f"topic pairs exhausted: {group} groups"}


def count_round(n: int, folder: Path, write: list, screen: list) -> dict:
    m = json.loads((folder / "manifest_ronde.json").read_text())
    if not (folder / "simpan_ronde.json").exists():
        raise SystemExit(f"ABORT: round {n} is not stored (extended_paired_store.py).")
    if not (folder / "juri_ronde.json").exists():
        raise SystemExit(f"ABORT: round {n} is not judged (extended_paired_judge.py).")
    judge = json.loads((folder / "juri_ronde.json").read_text())
    if judge.get("rule") != U.JUDGE_RULE:
        raise SystemExit(f"ABORT: round {n} was judged under rule {judge.get('rule')}, not "
                         f"{U.JUDGE_RULE}; judge it again under {U.JUDGE_RULE}.")
    t = [r for r in write if r["ronde"] == n]
    s = [r for r in screen if r["ronde"] == n]
    passed = {r["tp_id"] for r in s if r["lolos"]}
    rated = {h["tp_id"] for h in judge["results"]}
    if rated != passed:
        raise SystemExit(f"ABORT: round {n}: {len(passed)} questions passed the screen but "
                         f"{len(rated)} were judged.")
    if judge["unreadable_replies"] > 0.01 * judge["verdicts_used"]:
        raise SystemExit(f"ABORT: round {n} has more than 1% unreadable verdicts; it is refused.")
    return {"ronde": n, "list": m["list"], "tp_from": m["tp_from"], "tp_to": m["tp_to"],
            "sent": m["topic_pairs"], "written": sum(r["usable"] for r in t),
            "declined": sum(not r["usable"] for r in t), "passed_screen": len(passed),
            "judged": len(rated), "clear": sum(h["jelas"] == "ya" for h in judge["results"]),
            "groups": sum(h["group"] for h in judge["results"]),
            "strict_groups": sum(h["strict"] for h in judge["results"]),
            "unreadable_replies": judge["unreadable_replies"]}


def stage_status() -> dict:
    ordering = load_order()
    write, screen = U.read_jsonl(U.WRITE_FILE), U.read_jsonl(U.SCREEN_FILE)
    rounds, sent, group = [], set(), 0
    for n in stored_rounds():
        permit = next_round(ordering, sent, group, n - 1)
        h = count_round(n, U.BATCH / f"ronde_{n:02d}", write, screen)
        if permit.get("ronde") != n or (permit["list"], permit["tp_from"], permit["tp_to"]) != \
                (h["list"], h["tp_from"], h["tp_to"]):
            raise SystemExit(f"ABORT: round {n} ({h['list']} {h['tp_from']}-{h['tp_to']}) is not "
                             f"the round the stopping rule allowed ({permit}).")
        sent |= {r["tp_id"] for r in ordering if h["tp_from"] <= r["urut"] <= h["tp_to"]
                     and r["kelompok"] == h["list"]}
        group += h["groups"]
        rounds.append(h)
    content = {"updated": U.iso(), "target_groups": U.TARGET_GROUPS, "rounds": rounds,
           "sent_total": len(sent), "groups_total": group,
           "next": next_round(ordering, sent, group, len(rounds))}
    U.write_json(U.ROUND_FILE, content)
    print(json.dumps({k: content[k] for k in ("sent_total", "groups_total", "next")}, indent=1))
    return content


def judge_results() -> dict:
    out = {}
    for n in stored_rounds():
        for h in json.loads((U.BATCH / f"ronde_{n:02d}" / "juri_ronde.json").read_text())["results"]:
            out[h["tp_id"]] = h
    return out


def exclusion_reason(t, s, j) -> str:
    if t is None:
        return "not_written"
    if not t["usable"]:
        return "declined_by_writer"
    if s is None or not s["lolos"]:
        return "failed_screen"
    if j is None:
        return "not_judged"
    return j["exclusion"]


def build_pool(d, ca: str, cb: str) -> list:
    sim = np.maximum(d.X @ d.X[d.pos[ca]], d.X @ d.X[d.pos[cb]])
    order = sorted(range(len(d.ids)), key=lambda j: (-float(sim[j]), d.ids[j]))
    pool = [ca, cb]
    for j in order:
        if len(pool) >= U.POOL_SIZE:
            break
        if d.ids[j] not in (ca, cb):
            pool.append(d.ids[j])
    return pool


def check_rows(rows: list):
    problems, per_group = [], defaultdict(list)
    for r in rows:
        per_group[r["pair_id"]].append(r)
        cand = {c["id"] for c in r["cands"]}
        if not set(r["gold"]) <= cand or not set(r["counterpart"]) <= cand:
            problems.append(f"{r['query_id']}: a passage is missing from its pool")
        if len(r["cands"]) != U.POOL_SIZE or len(cand) != U.POOL_SIZE:
            problems.append(f"{r['query_id']}: pool of {len(r['cands'])} candidates")
    for g, rs in per_group.items():
        prog = [r["profile"]["department_name"] for r in rs]
        if len(rs) != 2 or len(set(prog)) != 2 or rs[0]["cands"] != rs[1]["cands"]:
            problems.append(f"{g}: {len(rs)} rows, programs {prog}")
    if len({r["query_id"] for r in rows}) != len(rows):
        problems.append("duplicate row ids")
    if problems:
        raise SystemExit("ABORT: row check failed, nothing written:\n  " + "\n  ".join(problems[:20]))


def stage_assemble():
    st = U.require_preregistration()
    if U.POOL.exists() or U.POOL_MANIFEST.exists() or U.POPULATION.exists():
        raise SystemExit("ABORT: the p2 pool exists. It is assembled once.")
    status = stage_status()
    if "stop" not in status["next"]:
        raise SystemExit(f"ABORT: the stopping rule allows another round ({status['next']}).")
    ordering = load_order()
    sent = {r["tp_id"] for h in status["rounds"] for r in ordering
                if h["tp_from"] <= r["urut"] <= h["tp_to"] and r["kelompok"] == h["list"]}
    write = {r["tp_id"]: r for r in U.read_jsonl(U.WRITE_FILE)}
    screen = {r["tp_id"]: r for r in U.read_jsonl(U.SCREEN_FILE)}
    judge = judge_results()
    ids, text, meta = T1.load_corpus()
    d = type("D", (), {})()
    d.ids, d.X = ids, T1.load_embeddings(ids)
    d.pos = {c: i for i, c in enumerate(ids)}
    round_tp = {r["tp_id"]: h["ronde"] for h in status["rounds"] for r in ordering
                if h["tp_from"] <= r["urut"] <= h["tp_to"] and r["kelompok"] == h["list"]}
    population, rows, n_groups = [], [], 0
    for tp in (r for r in ordering if r["tp_id"] in sent):
        t, s, j = write.get(tp["tp_id"]), screen.get(tp["tp_id"]), judge.get(tp["tp_id"])
        reason = exclusion_reason(t, s, j)
        record = {"tp_id": tp["tp_id"], "order": tp["urut"], "list": tp["kelompok"],
                 "round": round_tp[tp["tp_id"]], "program_pair": tp["program_pair"],
                 "cosine": tp["cosine"], "jaccard": tp["jaccard"],
                 "passages": [{k: p[k] for k in ("program", "book", "semester", "chunk")}
                              for p in tp["paragraf"]],
                 "written": bool(t and t["usable"]), "question": t["question"] if t else None,
                 "writer_reason": t["reason"] if t else None,
                 "passed_screen": bool(s and s["lolos"]), "screen_reason": s["alasan"] if s else None,
                 "judge": j, "exclusion": reason, "group": reason is None, "group_id": None,
                 "class": j["class"] if j else None, "strict": bool(j and j["strict"])}
        if reason is None:
            n_groups += 1
            gid = f"p2-{n_groups}"
            record["group_id"] = gid
            psg = sorted(tp["paragraf"], key=lambda p: p["program"])
            pool = build_pool(d, psg[0]["chunk"], psg[1]["chunk"])
            cands = [{"id": c, "text": text[d.pos[c]][:U.MAX_TEXT]} for c in pool]
            judged = {p["chunk"]: p for p in j["paragraf"]}
            for p in psg:
                other = next(x for x in psg if x is not p)
                rows.append({
                    "query_id": f"{gid}-{p['program'].lower()}", "pair_id": gid,
                    "query": t["question"], "gold": [p["chunk"]], "counterpart": [other["chunk"]],
                    "profile": {"department_name": p["program"], "department": p["program"],
                                "current_semester": p["semester"]},
                    "cands": cands,
                    "provenance": {"kueri_asal": tp["tp_id"], "class_label": "p2_constructed",
                                   "buku": p["book"], "sim_padanan": [tp["cosine"]],
                                   "program_pair": tp["program_pair"], "jaccard": tp["jaccard"],
                                   "round": round_tp[tp["tp_id"]], "writer_agent": t["agent_id"],
                                   "screen_agent": s["agent_id"],
                                   "judge": {"rule": U.JUDGE_RULE, "clear": j["jelas"],
                                             "class": j["class"], "strict": j["strict"],
                                             "own": judged[p["chunk"]],
                                             "other": judged[other["chunk"]]}}})
        population.append(record)
    if not rows:
        raise SystemExit("ABORT: no group; nothing to assemble.")
    check_rows(rows)
    tmp = U.POOL.with_name(U.POOL.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for b in rows:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    def per_program(key):
        out = defaultdict(Counter)
        for r in population:
            out[r["program_pair"]][key(r)] += 1
        return {k: dict(v) for k, v in sorted(out.items())}
    U.write_json(U.POPULATION, {"created": U.iso(),
                              "what": "every topic pair sent to a writer, "
                                      "with the writer output, the screen verdict, the judge verdicts "
                                      "and the reason for exclusion", "records": population})
    manifest = {
        "run_id": "pools_p2_" + U.iso().replace(":", "").replace("-", ""),
        "script": "study/channel/scripts/10_rebuilt_and_extended/extended_paired_assemble.py", "command": "extended_paired_assemble.py --tahap rakit",
        "rules": {"group": "written, passed the screen, judged clear, no 'tidak' or unreadable verdict for either "
                           "passage under either wording, both passages of one class in wording A",
                  "strict_subset": "both passages 'penuh' under both wordings",
                  "rows_per_group": 2, "pool_size": U.POOL_SIZE,
                  "pool": "both passages and the 69 chunks with the highest cosine to either "
                          "passage, ties by chunk id", "text_cut": U.MAX_TEXT},
        "stopping": status,
        "attrition_by_program_pair": per_program(lambda r: r["exclusion"] or "group"),
        "groups_by_program_pair": dict(sorted(Counter(r["program_pair"] for r in population
                                                      if r["group"]).items())),
        "groups_by_class": dict(Counter(r["class"] for r in population if r["group"])),
        "strict_subset": sum(r["strict"] for r in population),
        "sample": {"n_groups": n_groups, "n_rows": len(rows)},
        "inputs": {"order": U.sha(U.PAIRS), "writer": U.sha(U.WRITE_FILE), "screen": U.sha(U.SCREEN_FILE),
                   "judge": U.sha(U.JUDGE), "corpus": U.sha(T1.CORPUS), "embedding_cache": U.sha(T1.CACHE)},
        "output": {"path": "results/raw_p2/pools_p2.jsonl", "sha256": U.sha(tmp),
                   "population_table": {"path": "results/tables/p2_populasi.json",
                                        "sha256": U.sha(U.POPULATION)}},
        "label_source": "LLM writer and screen agents (Claude), judge Qwen3-14B-AWQ (two wordings); "
                        "the raters' ratings of the p2 kit, not the agents or the judge, are the "
                        "reference for validity",
    }
    U.write_json(U.POOL_MANIFEST, manifest)
    tmp.replace(U.POOL)
    print(json.dumps({"n_groups": n_groups, "n_rows": len(rows),
                      "groups_by_program_pair": manifest["groups_by_program_pair"]}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tahap", choices=["status", "rakit"], required=True)
    args = ap.parse_args()
    if args.tahap == "status":
        U.require_preregistration()
        stage_status()
    else:
        stage_assemble()


if __name__ == "__main__":
    main()
