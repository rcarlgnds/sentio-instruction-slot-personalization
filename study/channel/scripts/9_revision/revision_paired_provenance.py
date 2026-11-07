import os
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
EVIDENCE = Path("<home>/bukti_asal_paired_29jul2026")
JOURNAL = EVIDENCE / "wf_d095d1b4-baf/journal.jsonl"
TOPICS = ROOT / "system/data/paired_set/topic_pairs.jsonl"
PAIRED = ROOT / "system/data/paired_set/paired_queries.jsonl"
TABLE = ROOT / "study/channel/results/tables/r1_paired_provenance.json"
MANIFEST = ROOT / "study/channel/results/raw_r1/r1_paired_provenance.manifest.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    journal = [json.loads(b) for b in JOURNAL.open() if b.strip()]
    topic = [json.loads(b) for b in TOPICS.open() if b.strip()]
    paired = [json.loads(b) for b in PAIRED.open() if b.strip()]
    item, verdict = {}, {}
    dup_items, dup_verdicts = [], []
    for r in journal:
        res = r.get("result")
        if not isinstance(res, dict):
            continue
        for x in res.get("items") or []:
            k = x["pair_index"]
            if k in item:
                dup_items.append(k)
            item[k] = {**x, "writer_agent": r["agentId"]}
        for x in res.get("verdicts") or []:
            k = x["pair_index"]
            if k in verdict:
                dup_verdicts.append(k)
            verdict[k] = {**x, "judge_agent": r["agentId"]}
    paired_text = {}
    gold_paired = {}
    for r in paired:
        paired_text.setdefault(r["pair_id"], r["query_text"])
        program = r["student_profile"].get("department_name") or r["student_profile"].get("department")
        gold_paired.setdefault(r["pair_id"], {})[program] = (r["gold_relevant_chunks"],
                                                          r["counterpart_chunks"])
    rows, wrong_index, text_mismatch = [], [], []
    for k in range(150):
        it = item.get(k, {})
        written = bool((it.get("question") or "").strip())
        v = verdict.get(k)
        passed = bool(v and v.get("lolos"))
        pid = f"paired-{k}"
        rec = {"pair_index": k, "topic_pair_similarity": float(topic[k]["sim"]),
               "csci_chunk": topic[k]["csci"]["id"], "stat_chunk": topic[k]["stat"]["id"],
               "writer_agent": it.get("writer_agent"), "question_written": written,
               "question": it.get("question") if written else None,
               "writer_reason": it.get("reason"), "writer_usable_flag": it.get("usable"),
               "judged": v is not None, "passed": passed if v else None,
               "judge_reason": v.get("alasan") if v else None,
               "judge_suggestion": v.get("saran_perbaikan") if v else None,
               "judge_agent": v.get("judge_agent") if v else None,
               "released_pair_id": pid if pid in paired_text else None}
        if passed:
            if paired_text.get(pid) != it.get("question"):
                text_mismatch.append(k)
            g = gold_paired.get(pid, {})
            expected = {"CSCI": ([topic[k]["csci"]["id"]], [topic[k]["stat"]["id"]]),
                     "STAT": ([topic[k]["stat"]["id"]], [topic[k]["csci"]["id"]])}
            if g != expected:
                wrong_index.append(k)
        rows.append(rec)
    written = [r for r in rows if r["question_written"]]
    rated = [r for r in rows if r["judged"]]
    passed = [r for r in rows if r["passed"]]
    test_mentions = Counter(int(m) for r in rated if r["passed"] is False
                        for m in set(re.findall(r"UJI\s*([1-4])", r["judge_reason"] or "")))
    checks = {
        "topic_pairs_sorted_by_similarity": all(float(topic[i]["sim"]) >= float(topic[i + 1]["sim"])
                                                for i in range(len(topic) - 1)),
        "one_writer_item_per_pair_index": sorted(item) == list(range(150)) and not dup_items,
        "one_verdict_per_written_question": sorted(verdict) == sorted(r["pair_index"] for r in written)
                                            and not dup_verdicts,
        "no_verdict_for_refused_pairs": all(not r["judged"] for r in rows if not r["question_written"]),
        "passing_questions_equal_release_text": not text_mismatch,
        "passing_golds_equal_topic_pair_chunks": not wrong_index,
        "released_pairs_equal_passing_pairs": sorted(r["released_pair_id"] for r in passed)
                                              == sorted(paired_text),
    }
    output = {
        "created": datetime.now(timezone.utc).isoformat(),
        "script": "study/channel/scripts/9_revision/revision_paired_provenance.py",
        "what": "provenance of the paired-track questions: 150 most similar cross-book topic pairs "
                "-> writer agents -> judge agents -> 52 released pairs",
        "flow": {"topic_pairs_total": len(topic), "pairs_sent_to_writers": 150,
                 "questions_written": len(written), "refused_by_writers": 150 - len(written),
                 "judged": len(rated), "passed": len(passed), "failed": len(rated) - len(passed),
                 "released_pairs": len(paired_text)},
        "failed_reasons_mention_test": {f"UJI {k}": v for k, v in sorted(test_mentions.items())},
        "failed_reasons_note": "counts of failure reasons that mention each judge test; one reason "
                               "can mention several tests, and a mention is not an adjudicated "
                               "category",
        "checks": checks, "all_checks_passed": all(checks.values()),
        "mismatches": {"text": text_mismatch, "gold_chunks": wrong_index},
        "inputs": {"journal": {"path": str(JOURNAL), "sha256": sha(JOURNAL),
                               "note": "Claude Code workflow journal kept outside the repository"},
                   "topic_pairs": {"path": str(TOPICS.relative_to(ROOT)), "sha256": sha(TOPICS)},
                   "paired_queries": {"path": str(PAIRED.relative_to(ROOT)), "sha256": sha(PAIRED)}},
        "label_source": "agent outputs as logged; no new judgement",
        "pairs": rows,
    }
    TABLE.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "run_id": "r1_paired_provenance_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/9_revision/revision_paired_provenance.py",
        "output": {"path": "results/tables/r1_paired_provenance.json", "sha256": sha(TABLE)},
        "inputs": output["inputs"]}, indent=1))
    print(json.dumps({"flow": output["flow"], "checks": checks,
                      "failed_reasons_mention_test": output["failed_reasons_mention_test"]}, indent=1))
    print(f"written: {TABLE}")
    if not output["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
