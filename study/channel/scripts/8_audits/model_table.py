import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
RAW = ROOT / "study/channel/results/raw"
OUT = ROOT / "study/channel/results/tables/model_table.json"
MANIFEST = RAW / "model_table.manifest.json"
HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"

MODEL = [
    ("Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-0.6B",
     "scores_qwen3rr_Qwen3-Reranker-0.6B_paired_profile_docmeta"),
    ("Qwen3-Reranker-4B", "Qwen/Qwen3-Reranker-4B",
     "scores_qwen3rr_Qwen3-Reranker-4B_paired_profile_docmeta"),
    ("zerank-2", "zeroentropy/zerank-2",
     "scores_zerank2official_zerank-2_paired_profile_docmeta"),
    ("FollowIR-7B", "jhu-clsp/FollowIR-7B", "scores_followir_FollowIR-7B_paired_both"),
    ("ctxl-6B", "ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b",
     "scores_ctxl_ctxl-rerank-v2-instruct-multilingual-6b_paired_both"),
    ("gemma", "BAAI/bge-reranker-v2-gemma", "scores_gemma_bge-reranker-v2-gemma_paired_both"),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def model_card(repo: str) -> dict:
    d = HUB / ("models--" + repo.replace("/", "--"))
    snaps = sorted((d / "snapshots").iterdir()) if (d / "snapshots").exists() else []
    ref = (d / "refs/main").read_text().strip() if (d / "refs/main").exists() else None
    if len(snaps) != 1:
        raise SystemExit(f"STOP: {repo} has {len(snaps)} snapshots, so the revision is ambiguous")
    s = snaps[0]
    if ref and ref != s.name:
        raise SystemExit(f"STOP: {repo} refs/main {ref} does not point at snapshot {s.name}")
    readme = (s / "README.md").read_text() if (s / "README.md").exists() else ""
    front = readme.split("---")[1] if readme.startswith("---") else ""
    lic = re.search(r"^license:\s*(\S+)", front, re.M)
    base = re.search(r"^base_model:\s*\n?\s*-?\s*([\w./-]+)", front, re.M)
    cfg = json.loads((s / "config.json").read_text())
    return {
        "revisi": s.name,
        "diunduh": datetime.fromtimestamp(s.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d"),
        "lisensi_kartu_model": lic.group(1) if lic else None,
        "base_model_kartu_model": base.group(1) if base else None,
        "arsitektur": (cfg.get("architectures") or [None])[0],
        "model_type": cfg.get("model_type"),
        "lapisan": cfg.get("num_hidden_layers"),
        "hidden": cfg.get("hidden_size"),
    }


def read_run(filename: str) -> dict:
    m = json.loads((RAW / (filename + ".manifest.json")).read_text())
    cmd = str(m.get("perintah_persis") or "")
    budget = m.get("max_len")
    budget_source = "manifest"
    if budget is None:
        g = re.search(r"--max-len\s+(\d+)", cmd)
        if g:
            budget, budget_source = int(g.group(1)), "command line"
        elif "--family " in cmd:
            budget, budget_source = 2048, ("default --max-len of the arm scoring script, never changed in any command")
    k = m.get("kontrak_inferensi") or {}
    return {
        "presisi": m.get("dtype"),
        "anggaran_token": budget,
        "sumber_anggaran_token": budget_source,
        "sumber_template": k.get("sumber"),
        "format_template": k.get("format"),
        "aturan_skor": k.get("skor") or k.get("jalur"),
        "jalur": (f"vendor library sentence-transformers {k.get('versi_sentence_transformers')}"
                  f", run locally"
                  if "CrossEncoder" in str(k.get("jalur")) else
                  "own harness, vendor template copied verbatim, run locally"),
        "skrip": m.get("script"),
        "berkas_acuan": filename + ".jsonl",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timpa", action="store_true",
                    help="replace the existing table with a reading of this machine's model cache")
    if OUT.exists() and not ap.parse_args().timpa:
        print("STOP: model_table.json exists and its revisions are read from this machine's model cache.")
        print("Pass --timpa only if you mean to replace the record.")
        raise SystemExit(2)
    rows = []
    for name, repo, fpath in MODEL:
        rows.append({"model": name, "repo": repo, **model_card(repo), **read_run(fpath)})

    emb_dir = HUB / "models--BAAI--bge-m3"
    emb = {
        "repo": "BAAI/bge-m3",
        "snapshot_di_cache": sorted(p.name for p in (emb_dir / "snapshots").iterdir()),
        "refs_main": (emb_dir / "refs/main").read_text().strip(),
        "catatan": ("the two snapshots in the cache and the frozen record (env_versions.json) name only "
                    "'BAAI/bge-m3 fp16' without a revision, so which snapshot built the index cannot be "
                    "settled from the files"),
    }

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": "0a7b2e4de856",
        "_apa_ini": ("The six rerankers as they ran: weight revision, license, architecture, precision, token "
                     "budget, template, and scoring path."),
        "sumber_setiap_kolom": {
            "revisi": "local Hugging Face cache, one snapshot per model, refs/main points to it",
            "lisensi_dan_base": "front matter of the model card in the same snapshot",
            "arsitektur": "config.json in the same snapshot",
            "presisi_anggaran_template_skor": "closing manifest of the paired both cell of each model",
        },
        "offline": "every run set HF_HUB_OFFLINE=1, so the weights loaded are that snapshot",
        "model": rows,
        "embedder": emb,
        "label_source": "deterministic (read from the model cache and the run manifests)",
    }
    OUT.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    MANIFEST.write_text(json.dumps({
        "run_id": "model_table_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/8_audits/model_table.py",
        "output": {"path": "results/tables/model_table.json", "sha256": sha(OUT)},
        "label_source": "deterministic",
    }, indent=1))

    for b in rows:
        print(f"{b['model']:<21}{b['revisi'][:12]}  {b['lisensi_kartu_model']:<16}{b['arsitektur']:<20}"
              f"{b['presisi']:<9}{b['anggaran_token']:>6}  {b['jalur'][:44]}")
    print(f"embedder: {emb['snapshot_di_cache']} refs/main {emb['refs_main'][:12]}")


if __name__ == "__main__":
    main()
