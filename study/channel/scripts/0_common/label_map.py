import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
CORPUS = ROOT / "system/data/benchmark_release/corpus/corpus.jsonl"
ALIAS = ROOT / "system/data/benchmark_release/uuid_to_canonical.json"
VERSION = "id-v1"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_id_label_map(corpus: Path = CORPUS, alias: Path = ALIAS) -> dict:
    if not corpus.exists():
        raise SystemExit(f"ABORT: corpus not found at {corpus}")
    mapping = {}
    for rows in corpus.open(encoding="utf-8"):
        r = json.loads(rows)
        mapping[r["_id"]] = r["metadata"]
    if alias.exists():
        for uuid, canonical in json.loads(alias.read_text()).items():
            m = mapping.get(canonical)
            if m is not None:
                mapping[uuid] = m
    return mapping


def candidate_meta(mapping: dict, candidates) -> dict | None:
    key_id = candidates["id"] if isinstance(candidates, dict) else candidates
    return mapping.get(key_id)


def load_text_label_map(corpus: Path = CORPUS) -> dict:
    mapping = {}
    for rows in corpus.open(encoding="utf-8"):
        r = json.loads(rows)
        k = (r.get("text") or "")[:180].strip()
        if k:
            mapping.setdefault(k, r["metadata"])
    return mapping


def source_summary() -> dict:
    return {
        "peta_label": "id",
        "versi": VERSION,
        "sumber": {
            "korpus": str(CORPUS.relative_to(ROOT)),
            "korpus_sha256": _sha(CORPUS),
            "alias_uuid": str(ALIAS.relative_to(ROOT)) if ALIAS.exists() else None,
            "alias_sha256": _sha(ALIAS) if ALIAS.exists() else None,
        },
        "aturan": ("program and semester come from the candidate's identity; text matching is not used"),
    }
