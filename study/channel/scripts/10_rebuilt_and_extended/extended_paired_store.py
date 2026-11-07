import argparse
import json
import re
import shutil
import sys
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import extended_paired_common as U  # noqa: E402

WORKFLOW_JS = U.SCRIPT / "q2_p2_alur_tulis.js"
WRITE_TASK_MARK = "TUGAS: tulis pertanyaan untuk setiap pasangan di berkas berikut."
SCREEN_TASK_MARK = "TUGAS VALIDASI ADVERSARIAL."
FILE_PATTERN = re.compile(r"File pasangan: (\S+\.json)")
CANDIDATES_START = "Kandidat yang harus kamu periksa:\n"
CANDIDATES_END = "\n\nUntuk TIAP kandidat"
FRAMES = "[Workflow harness — computed task]"
TASK_TEXT_MARK = "The computed task text follows:\n"


def fixed_prompt_blocks() -> dict:
    js = WORKFLOW_JS.read_text(encoding="utf-8")
    context = re.search(r"const KONTEKS = `(.*?)`", js, re.S).group(1)
    test_match = re.search(r"(UJI 1 KEBOCORAN LEKSIKAL.*?saran_perbaikan\.)", js, re.S).group(1)
    return {"konteks": context, "uji": test_match}


def message_text(rec: dict) -> str:
    content = (rec.get("message") or {}).get("content")
    if isinstance(content, list):
        return "\n".join(x.get("text", "") for x in content if isinstance(x, dict))
    return content or ""


def agent_prompt(rec: list) -> str:
    text = [message_text(r) for r in rec if r.get("type") == "user"]
    for x in text:
        if x.startswith(FRAMES):
            i = x.find(TASK_TEXT_MARK)
            if i < 0:
                raise SystemExit("ABORT: a computed-task turn without its text.")
            return "\n".join(b[2:] if b.startswith("  ") else ("" if not b.strip() else b)
                             for b in x[i + len(TASK_TEXT_MARK):].split("\n"))
    if not text:
        raise SystemExit("ABORT: an agent record holds no prompt.")
    return text[0]


def read_agent(folder: Path, agent_id: str) -> dict:
    p = folder / f"agent-{agent_id}.jsonl"
    if not p.exists():
        raise SystemExit(f"ABORT: {p} is missing; the agent record cannot be checked.")
    rec = U.read_jsonl(p)
    model = sorted({(r.get("message") or {}).get("model") for r in rec
                    if r.get("type") == "assistant"} - {None})
    attempt = sorted({str(r.get("effort")) for r in rec if r.get("type") == "assistant"} - {"None"})
    return {"prompt": agent_prompt(rec), "model": model, "effort": attempt,
            "transcript_sha256": U.sha(p)}


def archive_transcripts(folder: Path, run: str) -> dict:
    goal = U.WORKFLOW_ARCHIVE / run
    if goal.exists():
        raise SystemExit(f"ABORT: {goal} exists; a run is archived once.")
    shutil.copytree(folder, goal)
    entries = {str(p.relative_to(goal)): U.sha(p) for p in sorted(goal.rglob("*")) if p.is_file()}
    (goal / "MANIFEST.sha256").write_text(
        "".join(f"{h}  {n}\n" for n, h in entries.items()), encoding="utf-8")
    return {"archive": str(goal), "files": len(entries),
            "journal_sha256": entries.get("journal.jsonl")}


def candidates_from_prompt(prompt: str) -> list:
    i, j = prompt.find(CANDIDATES_START), prompt.find(CANDIDATES_END)
    if i < 0 or j < 0:
        raise SystemExit("ABORT: a screen prompt lacks its candidate list.")
    return json.loads(prompt[i + len(CANDIDATES_START):j])


def collect_results(folders: list, runs: list, round_batch: set, fixed: dict, save_archive=True) -> tuple:
    write, screen, archive = [], [], []
    for folder, run in zip(folders, runs):
        journal = Path(folder) / "journal.jsonl"
        if not journal.exists():
            if save_archive:
                raise SystemExit(f"ABORT: {journal} is missing.")
            continue
        if save_archive:
            archive.append({"run": run, **archive_transcripts(Path(folder), run)})
        for order, r in enumerate(U.read_jsonl(journal)):
            if r.get("type") != "result" or not isinstance(r.get("result"), dict):
                continue
            ag = read_agent(Path(folder), r["agentId"])
            prompt = ag["prompt"]
            if fixed["konteks"] not in prompt:
                raise SystemExit(f"ABORT: agent {r['agentId']} was not given the committed context text.")
            m = FILE_PATTERN.search(prompt)
            if not m or m.group(1) not in round_batch:
                raise SystemExit(f"ABORT: agent {r['agentId']} names no batch file of this round.")
            base = {"run": run, "journal_line": order, "agent_id": r["agentId"],
                     "model": ag["model"], "effort": ag["effort"],
                     "transcript_sha256": ag["transcript_sha256"], "batch_file": m.group(1),
                     "result": r["result"]}
            if WRITE_TASK_MARK in prompt:
                write.append(base)
            elif SCREEN_TASK_MARK in prompt:
                if fixed["uji"] not in prompt:
                    raise SystemExit(f"ABORT: screen {r['agentId']} was not given the four tests word for word.")
                screen.append({**base, "shown": candidates_from_prompt(prompt)})
            else:
                raise SystemExit(f"ABORT: agent {r['agentId']} is neither a writer nor a screen.")
    return write, screen, archive


def is_complete(expected_ids: list, items: list) -> bool:
    ids = [x.get("tp_id") for x in items]
    return sorted(ids) == sorted(expected_ids) and len(set(ids)) == len(ids)


def choose_outputs(round_no: int, manifest: dict, write: list, screen: list) -> tuple:
    write_record, filter_record, unused, problems = [], [], [], []
    for b in manifest["batches"]:
        f = str(U.ROOT / b["file"])
        name = Path(b["file"]).stem
        cands = [t for t in write if t["batch_file"] == f]
        used = next((t for t in cands if is_complete(b["tp_ids"], t["result"]["items"])), None)
        unused += [{"batch": name, "kind": "writer", "agent_id": t["agent_id"], "run": t["run"]}
                          for t in cands if t is not used]
        if used is None:
            problems.append(f"{name}: no complete writer output ({len(cands)} outputs)")
            continue
        usable = [x for x in used["result"]["items"] if x["usable"]]
        for x in used["result"]["items"]:
            write_record.append({"tp_id": x["tp_id"], "ronde": round_no, "batch": name,
                                "usable": x["usable"], "question": x["question"],
                                "topic": x["topic"], "reason": x["reason"],
                                **{k: used[k] for k in ("agent_id", "model", "effort", "run",
                                                           "transcript_sha256")}})
        if not usable:
            continue
        cand_s = [s for s in screen if s["batch_file"] == f and s["shown"] == usable]
        other = [s for s in screen if s["batch_file"] == f and s["shown"] != usable]
        s_used = next((s for s in cand_s if is_complete([x["tp_id"] for x in usable],
                                                      s["result"]["items"])), None)
        unused += [{"batch": name, "kind": "screen", "agent_id": s["agent_id"], "run": s["run"]}
                          for s in cand_s + other if s is not s_used]
        if s_used is None:
            problems.append(f"{name}: no complete screen of the writer output used "
                           f"({len(cand_s)} screens of it)")
            continue
        if s_used["agent_id"] == used["agent_id"]:
            problems.append(f"{name}: the screen agent is the writer agent")
            continue
        questions = {x["tp_id"]: x["question"] for x in usable}
        for x in s_used["result"]["items"]:
            filter_record.append({"tp_id": x["tp_id"], "ronde": round_no, "batch": name,
                                 "lolos": x["lolos"], "alasan": x["alasan"],
                                 "saran_perbaikan": x["saran_perbaikan"],
                                 "question_screened": questions[x["tp_id"]],
                                 **{k: s_used[k] for k in ("agent_id", "model", "effort", "run",
                                                            "transcript_sha256")}})
    return write_record, filter_record, unused, problems


def run_record_file(n: int) -> Path:
    return U.BATCH / f"ronde_{n:02d}" / "alur_runs.json"


def recorded_runs(n: int) -> list:
    p = run_record_file(n)
    return json.loads(p.read_text()) if p.exists() else []


def record_run(n: int, flow: Path, run: str):
    if not (U.BATCH / f"ronde_{n:02d}" / "manifest_ronde.json").exists():
        raise SystemExit(f"ABORT: round {n} has no batches.")
    existing = recorded_runs(n)
    if any(r["run"] == run for r in existing):
        raise SystemExit(f"ABORT: run {run} is already recorded for round {n}.")
    existing.append({"run": run, "alur": str(flow), "recorded": U.iso()})
    U.write_json(run_record_file(n), existing)
    print(json.dumps({"ronde": n, "runs": len(existing)}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ronde", type=int, required=True)
    ap.add_argument("--catat", action="store_true", help="record one workflow run of the round")
    ap.add_argument("--alur", action="append", type=Path, default=None,
                    help="workflow transcript folder; repeat for a batch sent again")
    ap.add_argument("--run", action="append", default=None,
                    help="workflow run id, one per --alur, in the same order")
    args = ap.parse_args()
    if args.catat:
        if not args.alur or not args.run or len(args.alur) != 1 or len(args.run) != 1:
            ap.error("--catat needs one --alur and one --run")
        record_run(args.ronde, args.alur[0], args.run[0])
        return
    if args.alur is None:
        recorded = recorded_runs(args.ronde)
        args.alur = [Path(r["alur"]) for r in recorded]
        args.run = [r["run"] for r in recorded]
    if not args.alur or len(args.alur) != len(args.run or []):
        ap.error("no recorded run for this round, or not one --run per --alur")
    U.require_preregistration()
    round_folder = U.BATCH / f"ronde_{args.ronde:02d}"
    manifest = json.loads((round_folder / "manifest_ronde.json").read_text())
    for b in manifest["batches"]:
        if U.sha(U.ROOT / b["file"]) != b["sha256"]:
            raise SystemExit(f"ABORT: {b['file']} changed after it was written.")
    done = {r["tp_id"] for r in U.read_jsonl(U.WRITE_FILE)}
    if done & {t for b in manifest["batches"] for t in b["tp_ids"]}:
        raise SystemExit(f"ABORT: round {args.ronde} is already stored.")
    round_batch = {str(U.ROOT / b["file"]) for b in manifest["batches"]}
    write, screen, archive = collect_results(args.alur, args.run, round_batch, fixed_prompt_blocks())
    write_record, filter_record, unused, problems = choose_outputs(args.ronde, manifest, write, screen)
    if problems:
        raise SystemExit("ABORT: nothing stored; send these batches again in a new run and pass "
                         "every run:\n  " + "\n  ".join(problems))
    with U.WRITE_FILE.open("a", encoding="utf-8") as f:
        for r in write_record:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with U.SCREEN_FILE.open("a", encoding="utf-8") as f:
        for r in filter_record:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    notes = {"ronde": args.ronde, "stored": U.iso(), "runs": archive,
               "writer_records": len(write_record), "screen_records": len(filter_record),
               "outputs_not_used": unused,
               "models": sorted({m for r in write_record + filter_record for m in r["model"]})}
    U.write_json(round_folder / "simpan_ronde.json", notes)
    print(json.dumps({k: notes[k] for k in ("ronde", "writer_records", "screen_records", "models")}
                     | {"usable": sum(r["usable"] for r in write_record),
                        "passed_screen": sum(r["lolos"] for r in filter_record),
                        "outputs_not_used": len(unused)}, indent=1))


if __name__ == "__main__":
    main()
