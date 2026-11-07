import os
import hashlib
import json
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
RAW = ROOT / "study/channel/results/raw"
RAW1 = ROOT / "study/retrieval/results/raw"
TABLE = ROOT / "study/channel/results/tables"

MODEL = ["Qwen3-Reranker-0.6B", "Qwen3-Reranker-4B", "zerank-2",
         "FollowIR-7B", "ctxl-6B", "gemma"]

_ARCHITECTURE = {
    "FollowIR-7B": "scores_followir_FollowIR-7B",
    "ctxl-6B": "scores_ctxl_ctxl-rerank-v2-instruct-multilingual-6b",
    "gemma": "scores_gemma_bge-reranker-v2-gemma",
}
_Z = "scores_zerank2official_zerank-2"


def _qwen(m):
    return f"scores_qwen3rr_{m}"


def cell_path(model: str, track: str, cell: str) -> Path:
    if track == "paired":
        if model in ("Qwen3-Reranker-0.6B", "Qwen3-Reranker-4B"):
            small = "0.6b" if "0.6B" in model else "4b"
            name = {
                "none": RAW1 / f"qwen_scores_{small}_paired_default.jsonl",
                "slot": RAW1 / f"qwen_scores_{small}_paired_profile.jsonl",
                "text": RAW / f"{_qwen(model)}_paired_default_docmeta.jsonl",
                "both": RAW / f"{_qwen(model)}_paired_profile_docmeta.jsonl",
                "wrong": RAW / f"{_qwen(model)}_paired_profile_docmetasalah.jsonl",
                "prog": RAW / f"{_qwen(model)}_paired_profile_docmeta_fieldprogram.jsonl",
                "sem": RAW / f"{_qwen(model)}_paired_profile_docmeta_fieldsemester.jsonl",
            }
            return name[cell]
        if model == "zerank-2":
            end = {"none": "paired_default", "slot": "paired_profile",
                     "text": "paired_default_docmeta", "both": "paired_profile_docmeta",
                     "wrong": "paired_profile_docmetasalah",
                     "prog": "paired_profile_docmeta_fieldprogram",
                     "sem": "paired_profile_docmeta_fieldsemester"}[cell]
            return RAW / f"{_Z}_{end}.jsonl"
        end = {"none": "none", "slot": "slot", "text": "text", "both": "both",
                 "wrong": "wrong", "prog": "both_program", "sem": "both_semester"}[cell]
        return RAW / f"{_ARCHITECTURE[model]}_paired_{end}.jsonl"

    if model in ("Qwen3-Reranker-0.6B", "Qwen3-Reranker-4B"):
        end = {"none": "xpair_default", "slot": "xpair_profile",
                 "both": "xpair_profile_docmeta", "wrong": "xpair_profile_docmetasalah"}[cell]
        return RAW / f"{_qwen(model)}_{end}.jsonl"
    if model == "zerank-2":
        end = {"none": "xpair_default", "slot": "xpair_profile",
                 "both": "xpair_profile_docmeta", "wrong": "xpair_profile_docmetasalah"}[cell]
        return RAW / f"{_Z}_{end}.jsonl"
    return RAW / f"{_ARCHITECTURE[model]}_xpair_{cell}.jsonl"


PAPER = {
    "paired": {
        "none": [48, 48, 48, 48, 48, 48],
        "slot": [49, 50, 48, 47, 52, 43],
        "text": [48, 48, 48, 48, 48, 48],
        "both": [88, 83, 96, 91, 96, 49],
        "wrong": [76, 72, 91, 86, 94, 49],
        "sem": [78, 84, 93, 91, 96, 51],
        "prog": [71, 73, 85, 51, 89, 49],
    },
    "xpair": {
        "none": [600, 600, 600, 600, 600, 600],
        "slot": [597, 598, 597, 593, 603, 602],
        "both": [697, 641, 819, 841, 921, 603],
        "wrong": [645, 653, 705, 812, 705, 599],
    },
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _old_sha() -> dict:
    board = json.loads((TABLE / "channel_board_paired.json").read_text())["papan"]
    out = {}
    for cell in board.values():
        for content in cell.values():
            if isinstance(content, dict):
                integ = content.get("integritas") or {}
                if not integ.get("manifest") and integ.get("sha256"):
                    out[Path(content["berkas"]).name] = integ["sha256"]
    return out


_OLD = None


def read(p: Path) -> dict:
    global _OLD
    if not p.exists():
        raise SystemExit(f"STOP: score file missing: {p}")
    manifest = p.with_name(p.stem + ".manifest.json")
    if not manifest.exists():
        if _OLD is None:
            _OLD = _old_sha()
        ref = _OLD.get(p.name)
        if ref is None:
            raise SystemExit(f"STOP: {p.name} has no manifest and no sha256 on the frozen board")
        if sha(p) != ref:
            raise SystemExit(f"STOP: {p.name} no longer matches the sha256 on the frozen board")
    rows = {}
    for ln in p.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[r["query_id"]] = r
    return rows


def positions(r: dict):
    gold = set(r.get("gold") or [])
    counterpart = r.get("counterpart") or []
    counterpart = set(counterpart) if isinstance(counterpart, list) else {counterpart}
    pg = next((i for i, c in enumerate(r["ranking"]) if c in gold), None)
    pc = next((i for i, c in enumerate(r["ranking"]) if c in counterpart), None)
    return pg, pc


def wins_per_row(p: Path) -> dict:
    out = {}
    for q, r in read(p).items():
        pg, pc = positions(r)
        if pg is None or pc is None:
            continue
        out[q] = 1 if pg < pc else 0
    return out


def manuscript_gate(track: str, cell: str) -> dict:
    result = {}
    for m, expected in zip(MODEL, PAPER[track][cell]):
        w = wins_per_row(cell_path(m, track, cell))
        result[m] = w
        if sum(w.values()) != expected:
            raise SystemExit(f"GATE STOP: {m} {track}/{cell} wins {sum(w.values())}, "
                             f"the manuscript prints {expected}")
    return result
