"""Builds the rebuilt broad track, where a judge model checks that each partner gold answers the question.

It keeps the population, pool rule, and stratified draw of the second broad sample. A passage from
another program joins a group only when the judge (Qwen3-14B-AWQ on a local server) rates it a full
answer. The stages run in order with --tahap and write the judge verdicts, the pool, and
results/tables/r2_population_broad.json.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import requests

try:
    import fcntl
except ImportError:
    fcntl = None

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import revision_broad_sample as T1  # noqa: E402

ROOT = T1.ROOT
AI = ROOT / "system"
RAW_R2 = ROOT / "study/channel/results/raw_r2"
POOL_R2 = RAW_R2 / "pools_xpair_r2.jsonl"
MANIFEST = RAW_R2 / "pools_xpair_r2.manifest.json"
JUDGE = RAW_R2 / "juri_r2.jsonl"
SERVER = RAW_R2 / "juri_r2_server.jsonl"
EMB_Q = RAW_R2 / "emb_kueri_r2.npy"
EMB_Q_ID = RAW_R2 / "emb_kueri_r2_ids.json"
POPULATION = T1.TABLE / "r2_population_broad.json"
PREREG = ROOT / "result/paper_q2/PRAREGISTRASI_R1_IJIES.md"
OLD_KIT_KEY = ROOT / "study/channel/validasi_luas_r1/answer_key/kunci_kit_luas.json"
SCRIPT = Path(__file__).resolve()

SEED = 20261016
SEED_STABILITY = 20261019
N_GROUPS = 600
N_STABILITY = 300
K_PER_DIRECTION = 5
MAX_MATCHES = 2
JUDGE_MODEL = "Qwen/Qwen3-14B-AWQ"
JUDGE_VERSION = "juri_r2_v1"
PARALLEL = 16
MAX_UNREADABLE = 0.01
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "repetition_penalty": 1.0, "max_tokens": 8}
R2_SCRIPT = ("rebuilt_broad_sample.py", "q2_r2_kit.py", "q2_r2_rencana.py", "rebuilt_broad_analysis.py",
            "q2_r2_validasi.py", "q2_r2_pengawas.py", "q2_r2_uji.py", "q2_r2_orkestra.sh")
R1_MODULE = ("revision_broad_sample.py", "q2_r1_rencana.py", "q2_r1_runner.py", "revision_broad_kit.py",
            "q2_r1_kit_paired.py", "q2_r1_petunjuk_pdf.py", "revision_analysis.py", "revision_statistics.py",
            "q2_r1_validasi.py", "q2_r1_pengawas.py", "inference_ladder.py", "metrics.py",
            "mined_pairs.py", "score_arms.py", "score_architectures.py")

CLEAR_OPTIONS = ("  ya        pertanyaan jelas dan wajar diajukan seorang mahasiswa\n"
              "  tidak     pertanyaan ambigu, janggal, atau tidak bisa dipahami")
ANSWER_OPTIONS = ("  penuh     paragraf menjawab pertanyaan itu secara lengkap\n"
              "  sebagian  paragraf menjawab sebagian pertanyaan atau hanya sebagian aspeknya\n"
              "  tidak     paragraf tidak menjawab pertanyaan itu")
PROMPT = {
    "jelas": ("Berikut pertanyaan seorang mahasiswa untuk asisten tanya jawab mata kuliah. "
              "Pertanyaan itu dibaca sendirian, tanpa gambar, tabel, soal, atau percakapan "
              "sebelumnya.\n\nPertanyaan:\n{pertanyaan}\n\nPilih satu:\n" + CLEAR_OPTIONS +
              "\n\nJawab dengan satu kata saja: ya atau tidak."),
    "jawab": ("Berikut pertanyaan seorang mahasiswa (bahasa Indonesia) dan satu paragraf dari "
              "buku teks (bahasa Inggris). Nilai berdasarkan teks yang tertulis saja.\n\n"
              "Pertanyaan:\n{pertanyaan}\n\nParagraf:\n{paragraf}\n\nPilih satu:\n" + ANSWER_OPTIONS +
              "\n\nJawab dengan satu kata saja: penuh, sebagian, atau tidak."),
    "jawab_b": ("Baca paragraf buku teks (bahasa Inggris) di bawah ini, lalu tentukan apakah "
                "paragraf itu menjawab pertanyaan mahasiswa (bahasa Indonesia) yang menyertainya. "
                "Gunakan hanya isi teks yang tertulis.\n\nParagraf:\n{paragraf}\n\n"
                "Pertanyaan mahasiswa:\n{pertanyaan}\n\nPilihan:\n" + ANSWER_OPTIONS +
                "\n\nTulis satu kata saja: penuh, sebagian, atau tidak."),
}
CHOICES = {"jelas": ("ya", "tidak"), "jawab": ("penuh", "sebagian", "tidak"),
           "jawab_b": ("penuh", "sebagian", "tidak")}


def sha(p: Path) -> str:
    return T1.sha(Path(p))


def iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def strip_think(reply: str) -> str:
    return re.sub(r"<think>.*?</think>", " ", reply or "", flags=re.S).strip()


def read_label(reply: str, choice) -> str:
    text = strip_think(reply).lower()
    m = re.match(r"[^a-z]*([a-z]+)", text)
    words = m.group(1) if m else ""
    return words if words in choice else "tak_terbaca"


def verdict_key(task: str, prompt: str) -> str:
    return hashlib.sha256(f"{JUDGE_VERSION}|{JUDGE_MODEL}|{task}|{prompt}".encode("utf-8")).hexdigest()


def kappa(a: list, b: list) -> float:
    class_label = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in class_label)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


def load_verdict_cache() -> dict:
    cache = {}
    if not JUDGE.exists():
        return cache
    rows = JUDGE.read_text(encoding="utf-8").split("\n")
    for i, b in enumerate(rows):
        if not b.strip():
            continue
        try:
            r = json.loads(b)
        except json.JSONDecodeError:
            if i == len(rows) - 1:
                print("note: the last line of juri_r2.jsonl is incomplete (interrupted write) "
                      "and is ignored; that verdict is requested again", flush=True)
                continue
            raise SystemExit(f"ABORT: line {i + 1} of {JUDGE} is not valid JSON.")
        cache[r["kunci"]] = r
    return cache


def trim_partial_tail() -> int:
    if not JUDGE.exists():
        return 0
    data = JUDGE.read_bytes()
    if not data or data.endswith(b"\n"):
        return 0
    truncate = data.rfind(b"\n") + 1
    with JUDGE.open("r+b") as f:
        f.truncate(truncate)
    return len(data) - truncate


class Judge:
    def __init__(self, server=None):
        self.cache = load_verdict_cache()
        self.server = server
        self.fpath = None
        self.write_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.local = threading.local()
        self.new_verdicts = Counter()
        self.used = set()

    def rate(self, task: str, query_id: str, chunk_id, **content) -> str:
        prompt = PROMPT[task].format(**content)
        k = verdict_key(task, prompt)
        rec = self.cache.get(k)
        if rec is None:
            if self.server is None:
                raise KeyError(f"no stored verdict for {task} {query_id} {chunk_id}; run "
                               "--tahap juri (it asks only for missing verdicts). If juri already "
                               "finished cleanly, the candidate order differs between the stages "
                               "(numpy or BLAS changed?) and must be explained before any draw")
            rec = self._request(task, prompt, query_id, chunk_id, k)
        self.used.add(k)
        return rec["label"]

    def full_answer(self, query_id: str, chunk_id: str, question: str, text: str) -> tuple:
        a = self.rate("jawab", query_id, chunk_id, pertanyaan=question, paragraf=text)
        b = self.rate("jawab_b", query_id, chunk_id, pertanyaan=question, paragraf=text) \
            if a == "penuh" else None
        return a == "penuh" and b == "penuh", a, b

    def _session(self):
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def _request(self, task, prompt, query_id, chunk_id, k):
        if self.stop_event.is_set():
            raise RuntimeError("judge stopped after an earlier error")
        url, headers = self.server
        payload = {"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}],
                   "stream": False, "n": 1, **SAMPLING,
                   "chat_template_kwargs": {"enable_thinking": False}}
        failed = n_empty = 0
        while True:
            try:
                r = self._session().post(f"{url}/v1/chat/completions", json=payload, headers=headers,
                                      timeout=180)
                r.raise_for_status()
                choice = r.json()["choices"][0]
                reply = choice["message"].get("content") or ""
            except (requests.RequestException, KeyError, IndexError, ValueError) as e:
                failed += 1
                if failed == 4 or self.stop_event.is_set():
                    self.stop_event.set()
                    raise RuntimeError(f"vLLM request failed {failed} times: {type(e).__name__}: {e}") from e
                time.sleep(10 * failed)
                continue
            if strip_think(reply):
                break
            n_empty += 1
            if n_empty == 4:
                break
            time.sleep(10 * n_empty)
        empty = not strip_think(reply)
        rec = {"kunci": k, "versi": JUDGE_VERSION, "model": JUDGE_MODEL, "tugas": task,
               "query_id": query_id, "chunk_id": chunk_id,
               "label": read_label(reply, CHOICES[task]), "balasan": reply, "kosong": empty,
               "empty_replies_before": n_empty if not empty else n_empty - 1,
               "finish_reason": choice.get("finish_reason"), "waktu": iso()}
        if empty:
            rec["reasoning_content"] = str(choice["message"].get("reasoning_content") or "")[:200]
        with self.write_lock:
            if k not in self.cache:
                self.fpath.write(json.dumps(rec) + "\n")
                self.fpath.flush()
                self.cache[k] = rec
                self.new_verdicts[task] += 1
        return self.cache[k]


def server_address():
    sys.path.insert(0, str(AI))
    os.chdir(AI)
    from app.config import settings
    url = settings.vllm_url.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.vllm_api_key}"} if settings.vllm_api_key else {}
    return url, headers


def server_info(url: str, headers: dict) -> dict:
    r = requests.get(f"{url}/v1/models", headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    try:
        version = requests.get(f"{url}/version", timeout=10).json().get("version")
    except (requests.RequestException, ValueError):
        version = None
    model = []
    for m in data:
        snap = re.search(r"snapshots/([0-9a-f]{40})", m.get("root") or "")
        model.append({"id": m.get("id"), "root": m.get("root"), "max_model_len": m.get("max_model_len"),
                      "revision": snap.group(1) if snap else None})
    return {"served_models": [m["id"] for m in model], "models": model, "vllm_version": version}


def git(*arg) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *arg], cwd=ROOT, capture_output=True, text=True)


def git_status(p: Path) -> dict:
    try:
        rel = str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return {"tracked": False, "clean": False, "sha256": None, "commit": None,
                "note": f"{p} is outside {ROOT}"}
    recorded = p.exists() and git("ls-files", "--error-unmatch", rel).returncode == 0
    clean = recorded and git("diff", "--quiet", "HEAD", "--", rel).returncode == 0
    return {"tracked": recorded, "clean": clean, "sha256": sha(p) if p.exists() else None,
            "commit": git("log", "-1", "--format=%H", "--", rel).stdout.strip() or None}


def preregistration_status() -> dict:
    text = PREREG.read_text(encoding="utf-8") if PREREG.exists() else ""
    flat = " ".join(text.split())
    required = [f"seed {SEED}", f"seed {SEED_STABILITY}", JUDGE_VERSION, JUDGE_MODEL]
    prompt_lines = [b for tmpl in PROMPT.values() for b in tmpl.splitlines() if b.strip()]
    st = {"missing_text": [s for s in required if s not in flat]
                          + [b for b in dict.fromkeys(prompt_lines) if b not in text],
          "sha256": sha(PREREG) if PREREG.exists() else None,
          "preregistration": git_status(PREREG),
          "scripts": {n: git_status(SCRIPT.parent / n) for n in R2_SCRIPT},
          "r1_modules": {n: git_status(SCRIPT.parent / n) for n in R1_MODULE
                         if (SCRIPT.parent / n).exists()}}
    st["script"] = st["scripts"][SCRIPT.name]
    st["ready"] = (not st["missing_text"] and st["preregistration"]["clean"]
                   and all(v["clean"] for v in st["scripts"].values())
                   and all(v["clean"] for v in st["r1_modules"].values() if v["tracked"]))
    return st


def require_preregistration() -> dict:
    st = preregistration_status()
    if not st["ready"]:
        raise SystemExit("ABORT: the r2 rules must be in the pre-registration, and the pre-registration and "
                         "every r2 script must be committed without later edits (tracked R1 modules unchanged), "
                         "before any judge call or draw:\n"
                         + json.dumps(st, indent=1, ensure_ascii=False))
    return st


def prepare_population():
    split, ids, text, meta, X, pos, program_idx, candidates = T1.prepare_population()
    eligible = [k for k in candidates if len(k["q"].get("gold_anchor") or []) == 1]
    multi_gold = [k for k in candidates if len(k["q"].get("gold_anchor") or []) != 1]
    return SimpleNamespace(split=split, ids=ids, text=text, meta=meta, X=X, pos=pos,
                           program_idx=program_idx, kandidat=candidates, eligible=eligible,
                           multi_gold=multi_gold)


def norm_deviation(d) -> float:
    return float(np.abs(np.linalg.norm(d.X, axis=1) - 1).max())


def require_unit_norm(d):
    if norm_deviation(d) > 1e-3:
        raise SystemExit(f"ABORT: corpus embeddings are not unit length (largest deviation {norm_deviation(d):.4g}"
                         f"); the cosine limits of the r2 rule assume they are.")


def load_question_embeddings(d) -> dict:
    qids = [k["q"]["query_id"] for k in d.eligible]
    if not (EMB_Q.exists() and EMB_Q_ID.exists()) or json.loads(EMB_Q_ID.read_text()) != qids:
        raise SystemExit("ABORT: question embeddings are missing or do not match the eligible "
                         "queries; run --tahap embed first.")
    Q = np.load(EMB_Q)
    return {q: Q[i] for i, q in enumerate(qids)}


def partner_candidates(cid: str, dept: str, qvec, d) -> list:
    sim_gold = d.X @ d.X[d.pos[cid]]
    query_sim = d.X @ qvec
    gold_words = set(d.text[d.pos[cid]].lower().split())
    cands = []
    for prog in sorted(p for p in d.program_idx if p is not None and p != dept):
        idxs = d.program_idx[prog]
        sq, sg = query_sim[idxs], sim_gold[idxs]
        pick = (set(np.argsort(-sq, kind="stable")[:K_PER_DIRECTION].tolist())
                 | set(np.argsort(-sg, kind="stable")[:K_PER_DIRECTION].tolist()))
        for j in sorted(pick):
            c = d.ids[int(idxs[j])]
            words = set(d.text[d.pos[c]].lower().split())
            jaccard = len(gold_words & words) / max(len(gold_words | words), 1)
            cands.append({"program": prog, "chunk": c, "sim_kueri": float(sq[j]),
                          "sim_gold": float(sg[j]), "jaccard": round(jaccard, 4),
                          "duplikat": float(sg[j]) > T1.UPPER_THRESHOLD or jaccard > T1.JACCARD_MAX})
    cands.sort(key=lambda c: (-c["sim_kueri"], c["chunk"]))
    return cands


def near_duplicate(d, a: str, b: str) -> tuple:
    cos = float(d.X[d.pos[a]] @ d.X[d.pos[b]])
    ka, kb = (set(d.text[d.pos[x]].lower().split()) for x in (a, b))
    jac = len(ka & kb) / max(len(ka | kb), 1)
    return cos, jac, cos > T1.UPPER_THRESHOLD or jac > T1.JACCARD_MAX


def empty_verdicts() -> dict:
    return {"jelas": None, "gold_a": None, "gold_b": None, "gold_penuh": False, "n_kandidat": 0,
            "n_duplikat": 0, "dinilai": [], "kembar": [], "padanan": []}


def judge_query(k: dict, d, qvec, judge: Judge) -> dict:
    qid, question = k["q"]["query_id"], k["q"]["query_text"]
    h = empty_verdicts()
    h["jelas"] = judge.rate("jelas", qid, None, pertanyaan=question)
    if h["jelas"] != "ya":
        return h
    h["gold_penuh"], h["gold_a"], h["gold_b"] = judge.full_answer(qid, k["cid"], question,
                                                           d.text[d.pos[k["cid"]]][:4000])
    if not h["gold_penuh"]:
        return h
    cands = partner_candidates(k["cid"], k["dept"], qvec, d)
    h["n_kandidat"] = len(cands)
    h["n_duplikat"] = sum(c["duplikat"] for c in cands)
    picked = {}
    for c in cands:
        if c["duplikat"] or c["program"] in picked:
            continue
        dupes = None
        for p in picked.values():
            cos, jac, same = near_duplicate(d, c["chunk"], p["chunk"])
            if same:
                dupes = {"chunk": c["chunk"], "program": c["program"], "of": p["chunk"],
                          "cosine": round(cos, 4), "jaccard": round(jac, 4)}
                break
        if dupes:
            h["kembar"].append(dupes)
            continue
        ok, a, b = judge.full_answer(qid, c["chunk"], question, d.text[d.pos[c["chunk"]]][:4000])
        h["dinilai"].append({"chunk": c["chunk"], "a": a, "b": b})
        if ok:
            picked[c["program"]] = c
            if len(picked) == MAX_MATCHES:
                break
    h["padanan"] = list(picked.values())
    return h


def exclusion_reason(k: dict, h) -> str:
    if len(k["q"].get("gold_anchor") or []) != 1:
        return "more_than_one_gold"
    if h["jelas"] != "ya":
        return "question_not_clear"
    if not h["gold_penuh"]:
        return "own_gold_not_full"
    if not h["padanan"]:
        return "no_full_partner"
    return None


def wording_a_pairs(k: dict, h) -> list:
    if h["gold_a"] is None:
        return []
    return [(k["q"]["query_id"], k["cid"])] + [(k["q"]["query_id"], x["chunk"]) for x in h["dinilai"]]


def stability_sample(items: list) -> list:
    pairs = sorted(set(items))
    rng = np.random.default_rng(SEED_STABILITY)
    idx = rng.choice(len(pairs), size=min(N_STABILITY, len(pairs)), replace=False)
    return [pairs[int(i)] for i in sorted(idx)]


def first_kit_items() -> list:
    if not OLD_KIT_KEY.exists():
        raise SystemExit(f"ABORT: {OLD_KIT_KEY} not found. The kit key of the first broad sample is needed "
                         "to measure the judge against the raters.")
    pool = {r["query_id"]: r for r in map(json.loads, filter(str.strip, T1.OLD_POOL.open(encoding="utf-8")))}
    out = []
    for k in json.loads(OLD_KIT_KEY.read_text(encoding="utf-8")):
        r = pool[k["query_id"]]
        out.append({"item_id": k["item_id"], "query_id": k["query_id"], "chunk": k["gold_chunk"],
                    "role": k["role"], "pertanyaan": r["query"],
                    "paragraf": next(c["text"] for c in r["cands"] if c["id"] == k["gold_chunk"])})
    return out


def judge_first_kit(judge: Judge, item: list) -> list:
    out = []
    for it in item:
        a = judge.rate("jawab", it["query_id"], it["chunk"], pertanyaan=it["pertanyaan"],
                       paragraf=it["paragraf"])
        b = judge.rate("jawab_b", it["query_id"], it["chunk"], pertanyaan=it["pertanyaan"],
                       paragraf=it["paragraf"])
        j = judge.rate("jelas", it["query_id"], None, pertanyaan=it["pertanyaan"])
        out.append({**{k: it[k] for k in ("item_id", "query_id", "chunk", "role")},
                    "a": a, "b": b, "jelas": j, "penuh": a == b == "penuh"})
    return out


def two_wordings(judge: Judge, d, questions: dict, qid: str, cid: str) -> tuple:
    text = d.text[d.pos[cid]][:4000]
    return (judge.rate("jawab", qid, cid, pertanyaan=questions[qid], paragraf=text),
            judge.rate("jawab_b", qid, cid, pertanyaan=questions[qid], paragraf=text))


def full_or_other(label: list) -> list:
    return ["penuh" if x == "penuh" else "lain" for x in label]


def prompt_stability(judge: Judge, sample: list, d) -> dict:
    questions = {k["q"]["query_id"]: k["q"]["query_text"] for k in d.eligible}
    pairs = [two_wordings(judge, d, questions, qid, cid) for qid, cid in sample]
    a, b = ([x for x, _ in pairs], [y for _, y in pairs])
    return {"n": len(sample), "seed": SEED_STABILITY,
            "agreement_three_classes": round(sum(x == y for x, y in zip(a, b)) / max(len(a), 1), 4),
            "kappa_three_classes": kappa(a, b) if a else None,
            "agreement_full_vs_not": round(sum(x == y for x, y in zip(full_or_other(a), full_or_other(b))) / max(len(a), 1), 4),
            "kappa_full_vs_not": kappa(full_or_other(a), full_or_other(b)) if a else None,
            "table": {f"{x}|{y}": n for (x, y), n in sorted(Counter(zip(a, b)).items())}}


def stage_check(d):
    st = preregistration_status()
    out = {"train_dev_queries": sum(1 for s in d.split.values() if s in ("train", "dev")),
           "eligible_r1_rule": len(d.kandidat),
           "excluded_more_than_one_gold": len(d.multi_gold),
           "excluded_more_than_one_gold_by_class": dict(Counter(
               k["q"].get("class_label") for k in d.multi_gold)),
           "eligible_single_gold": len(d.eligible),
           "by_gate": dict(Counter((k["q"].get("qc") or {}).get("protocol") for k in d.eligible)),
           "by_program": dict(sorted(Counter(k["dept"] for k in d.eligible).items())),
           "gold_relevant_chunks_disagree_with_gold_anchor": sum(
               1 for k in d.eligible if len(k["q"].get("gold_relevant_chunks") or []) != 1),
           "corpus_embedding_length_max_deviation_from_1": round(norm_deviation(d), 6),
           "preregistration": st}
    try:
        Q = load_question_embeddings(d)
        n_cand = [sum(1 for c in partner_candidates(k["cid"], k["dept"], Q[k["q"]["query_id"]], d)
                       if not c["duplikat"]) for k in d.eligible]
        n_kit = len(json.loads(OLD_KIT_KEY.read_text())) if OLD_KIT_KEY.exists() else 0
        out["judge_calls_upper_bound"] = (3 * len(d.eligible) + 2 * sum(n_cand) + 3 * n_kit + N_STABILITY)
        out["partner_candidates_per_query"] = {"mean": round(float(np.mean(n_cand)), 1),
                                               "max": int(max(n_cand))}
    except SystemExit as e:
        out["judge_calls_upper_bound"] = str(e)
    try:
        from revision_broad_kit import PETUNJUK
        out["prompt_options_match_rater_instructions"] = all(
            b in PETUNJUK.splitlines() for b in (CLEAR_OPTIONS + "\n" + ANSWER_OPTIONS).splitlines())
    except ImportError as e:
        out["prompt_options_match_rater_instructions"] = f"not checked: {e}"
    out["first_kit_key_found"] = OLD_KIT_KEY.exists()
    try:
        url, headers = server_address()
        out["server"] = {"url": url, "key_set": bool(headers), **server_info(url, headers)}
        out["server"]["judge_model_served"] = JUDGE_MODEL in out["server"]["served_models"]
    except Exception as e:
        out["server"] = f"not reachable: {type(e).__name__}"
    print(json.dumps(out, indent=1, ensure_ascii=False))


def stage_embed(d):
    qids = [k["q"]["query_id"] for k in d.eligible]
    if EMB_Q.exists() and EMB_Q_ID.exists() and json.loads(EMB_Q_ID.read_text()) == qids:
        print(f"question embeddings already present for {len(qids)} queries")
        return
    os.environ["EMBEDDING_DEVICE"] = "cpu"
    sys.path.insert(0, str(AI))
    os.chdir(AI)
    from app.ingestion.embedder import HybridEmbedder
    emb = HybridEmbedder()
    emb.load()
    text = [k["q"]["query_text"] for k in d.eligible]
    vecs = []
    for i in range(0, len(text), 64):
        vecs.extend(emb.embed_text_dense(text[i:i + 64]))
    Q = np.asarray(vecs, dtype=np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    if Q.shape != (len(qids), d.X.shape[1]):
        raise SystemExit(f"ABORT: question embeddings have shape {Q.shape}, expected "
                         f"({len(qids)}, {d.X.shape[1]}).")
    RAW_R2.mkdir(parents=True, exist_ok=True)
    np.save(EMB_Q, Q)
    EMB_Q_ID.write_text(json.dumps(qids))
    print(f"question embeddings written: {Q.shape[0]} x {Q.shape[1]}")


def summarize(result: dict) -> dict:
    hs = list(result.values())
    return {"queries": len(hs),
            "question_clear": dict(Counter(h["jelas"] for h in hs)),
            "own_gold_wording_a": dict(Counter(h["gold_a"] for h in hs if h["jelas"] == "ya")),
            "own_gold_full_both_wordings": sum(h["gold_penuh"] for h in hs),
            "partners_found": dict(Counter(len(h["padanan"]) for h in hs if h["gold_penuh"])),
            "partner_candidates_judged": sum(len(h["dinilai"]) for h in hs),
            "candidates_skipped_as_near_duplicate_of_a_partner": sum(len(h["kembar"]) for h in hs)}


def unreadable_verdicts(judge: Judge) -> tuple:
    rec = [judge.cache[k] for k in judge.used]
    return (sum(1 for r in rec if r["label"] == "tak_terbaca"),
            sum(1 for r in rec if r.get("kosong")))


def stage_judge(d, parallel: int, bound):
    st = require_preregistration()
    require_unit_norm(d)
    Q = load_question_embeddings(d)
    url, headers = server_address()
    info = server_info(url, headers)
    if JUDGE_MODEL not in info["served_models"]:
        raise SystemExit(f"ABORT: the server serves {info['served_models']}, not {JUDGE_MODEL}.")
    RAW_R2.mkdir(parents=True, exist_ok=True)
    target = d.eligible if bound is None else d.eligible[:bound]
    result, t0 = {}, time.time()
    with JUDGE.open("a", encoding="utf-8") as fpath:
        if fcntl is not None:
            try:
                fcntl.flock(fpath, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(f"ABORT: another judge process holds {JUDGE}; run one at a time.")
        with SERVER.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": iso(), "url": url, **info, "sampling": SAMPLING,
                                "parallel_requests": parallel, "limit": bound,
                                "scripts": {n: {"sha256": v["sha256"], "commit": v["commit"]}
                                            for n, v in st["scripts"].items()}}) + "\n")
        discarded = trim_partial_tail()
        if discarded:
            print(f"note: removed a partial last line ({discarded} bytes) left by an interrupted "
                  "write", flush=True)
        judge = Judge(server=(url, headers))
        judge.fpath = fpath
        print(f"judging {len(target)} queries with {parallel} parallel requests; "
              f"{len(judge.cache)} verdicts already stored", flush=True)
        ex = ThreadPoolExecutor(max_workers=parallel)
        try:
            task = {ex.submit(judge_query, k, d, Q[k["q"]["query_id"]], judge): k["q"]["query_id"]
                     for k in target}
            for i, f in enumerate(as_completed(task), start=1):
                result[task[f]] = f.result()
                if i % 25 == 0 or i == len(task):
                    old = time.time() - t0
                    n_new = sum(judge.new_verdicts.values())
                    print(f"{iso()} {i}/{len(task)} queries, {n_new} new verdicts "
                          f"({n_new / max(old, 1e-9):.1f}/s), about "
                          f"{old / i * (len(task) - i) / 60:.0f} min left", flush=True)
            if bound is None:
                print("judging the items of the first broad kit and the prompt-stability sample", flush=True)
                item = first_kit_items()
                list(ex.map(lambda it: judge_first_kit(judge, [it]), item))
                items = [p for k in target for p in wording_a_pairs(k, result[k["q"]["query_id"]])]
                questions = {k["q"]["query_id"]: k["q"]["query_text"] for k in target}
                list(ex.map(lambda qc: two_wordings(judge, d, questions, *qc), stability_sample(items)))
        except BaseException:
            judge.stop_event.set()
            ex.shutdown(wait=True, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
    unreadable, empty = unreadable_verdicts(judge)
    repeated = sum(1 for k in judge.used if judge.cache[k].get("empty_replies_before"))
    print(json.dumps({**summarize(result), "unreadable_replies": unreadable, "of_which_empty": empty,
                      "verdicts_after_an_empty_reply_was_asked_again": repeated,
                      "verdicts_used": len(judge.used), "new_verdicts": dict(judge.new_verdicts)}, indent=1))
    if bound is not None and unreadable:
        raise SystemExit(f"STOP for review: the smoke run has {unreadable} unreadable replies among {len(judge.used)}"
                         f" verdicts, {empty}"
                         f" of them empty (see 'balasan' in juri_r2.jsonl). Under the r2 rule "
                         f"the full run goes ahead only when none is empty and they are at most "
                         f"1%; empty replies point to a server setting: set juri_r2.jsonl aside "
                         f"under another name, fix the setting and repeat the smoke run.")
    if unreadable > MAX_UNREADABLE * len(judge.used):
        raise SystemExit(f"STOP: {unreadable} of {len(judge.used)} replies are unreadable ({empty} empty), "
                         f"above the pre-registered {MAX_UNREADABLE:.0%}; no draw is made from this "
                         "run. Report the cause.")
    if unreadable:
        print(f"note: {unreadable} replies did not start with an allowed word; they count as negative "
              "verdicts and are reported.", flush=True)


def stage_draw(d):
    st = require_preregistration()
    require_unit_norm(d)
    if POOL_R2.exists():
        raise SystemExit(f"ABORT: {POOL_R2}"
                         f" exists. The sample is drawn only once. To draw it again, delete the "
                         f"file and record why.")
    Q = load_question_embeddings(d)
    judge = Judge()
    population = []
    for k in d.kandidat:
        if len(k["q"].get("gold_anchor") or []) == 1:
            try:
                h = judge_query(k, d, Q[k["q"]["query_id"]], judge)
            except KeyError as e:
                raise SystemExit(f"ABORT: {e}")
        else:
            h = empty_verdicts()
        population.append({**k, "h": h, "alasan": exclusion_reason(k, h),
                         "gate": str((k["q"].get("qc") or {}).get("protocol"))})
    try:
        old_kit = judge_first_kit(judge, first_kit_items())
        stable = prompt_stability(judge, stability_sample(
            [p for x in population for p in wording_a_pairs(x, x["h"])]), d)
    except KeyError as e:
        raise SystemExit(f"ABORT: {e}. The two measurements of tahap juri are incomplete.")
    unreadable, empty = unreadable_verdicts(judge)
    if unreadable > MAX_UNREADABLE * len(judge.used):
        raise SystemExit(f"ABORT: {unreadable} of {len(judge.used)} verdicts are unreadable ({empty} "
                         f"empty), above the pre-registered {MAX_UNREADABLE:.0%}; no draw is made.")
    usable = [p for p in population if p["alasan"] is None]
    strata = defaultdict(list)
    for p in usable:
        strata[(p["gate"], p["dept"])].append(p)
    sizes = {k: len(v) for k, v in strata.items()}
    census = len(usable) <= N_GROUPS
    if census:
        alloc, selected = dict(sizes), list(usable)
    else:
        alloc = T1.proportional_allocation(sizes, N_GROUPS)
        rng = np.random.default_rng(SEED)
        selected = []
        for key_s in sorted(strata):
            members = strata[key_s]
            if alloc[key_s]:
                idx = rng.choice(len(members), size=alloc[key_s], replace=False)
                selected.extend(members[i] for i in sorted(int(x) for x in idx))
    selected.sort(key=lambda p: p["indeks_berkas"])
    if not selected:
        raise SystemExit("ABORT: no query has a clear question, a full own gold and a full "
                         "partner; nothing to draw.")
    if any(d.split.get(p["q"]["query_id"]) not in ("train", "dev") for p in selected):
        raise SystemExit("ABORT: a sampled query is outside train and dev.")

    rows = []
    for n, p in enumerate(selected, start=1):
        matches = p["h"]["padanan"]
        used = [(c["program"], c["chunk"], c["sim_gold"]) for c in matches]
        pool = T1.build_pool(p["cid"], used, d.X, d.pos, d.ids)
        for r in T1.group_rows(f"xr2-{n}", p, used, pool, d.text, d.pos, d.meta):
            r["provenance"]["sim_kueri_padanan"] = [round(c["sim_kueri"], 4) for c in matches]
            if len(matches) == 2:
                cos, jac, _ = near_duplicate(d, matches[0]["chunk"], matches[1]["chunk"])
                r["provenance"]["padanan_ke_padanan"] = {"cosine": round(cos, 4),
                                                         "jaccard": round(jac, 4)}
            r["provenance"]["aturan_padanan"] = JUDGE_VERSION
            rows.append(r)
    check_rows(rows, d)

    RAW_R2.mkdir(parents=True, exist_ok=True)
    tmp = POOL_R2.with_name(POOL_R2.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for b in rows:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    old_origin = pool_provenance(T1.OLD_POOL)
    r1_origin = pool_provenance(T1.NEW_POOL)
    selected_q = {p["q"]["query_id"] for p in selected}
    POPULATION.parent.mkdir(parents=True, exist_ok=True)
    POPULATION.write_text(json.dumps({
        "created": iso(),
        "what": "one record per training or development query whose gold lies in the asker's program "
                "(the population of the second broad sample), with the judge's verdicts, the partners "
                "found and the sampling outcome of the pre-registered r2 rule",
        "records": [{"query_id": p["q"]["query_id"], "file_index": p["indeks_berkas"],
                     "gate": p["gate"], "program": p["dept"],
                     "class_label": p["q"].get("class_label"),
                     "n_gold": len(p["q"].get("gold_anchor") or []),
                     "question_clear": p["h"]["jelas"], "own_gold_wording_a": p["h"]["gold_a"],
                     "own_gold_wording_b": p["h"]["gold_b"], "own_gold_full": p["h"]["gold_penuh"],
                     "partner_candidates": p["h"]["n_kandidat"],
                     "near_duplicates_skipped": p["h"]["n_duplikat"],
                     "near_duplicates_of_a_partner_skipped": len(p["h"]["kembar"]),
                     "skipped_as_near_duplicate_of_a_partner": p["h"]["kembar"],
                     "candidates_judged": len(p["h"]["dinilai"]),
                     "candidates_full_in_a_not_b": sum(1 for x in p["h"]["dinilai"]
                                                       if x["a"] == "penuh" and x["b"] != "penuh"),
                     "partners": [{"program": c["program"], "chunk": c["chunk"],
                                   "sim_question": round(c["sim_kueri"], 4),
                                   "sim_gold": round(c["sim_gold"], 4)} for c in p["h"]["padanan"]],
                     "exclusion": p["alasan"], "groupable": p["alasan"] is None,
                     "sampled_r2": p["q"]["query_id"] in selected_q,
                     "in_original_sample": p["q"]["query_id"] in old_origin,
                     "in_r1_sample": p["q"]["query_id"] in r1_origin} for p in population],
    }, indent=1, ensure_ascii=False))

    def compute(entries, key):
        return dict(sorted(Counter(key(p) for p in entries).items(), key=lambda kv: str(kv[0])))
    single = [p for p in population if p["alasan"] != "more_than_one_gold"]
    rated = [x for p in single for x in p["h"]["dinilai"]]
    manifest = {
        "run_id": "pools_xpair_r2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/10_rebuilt_and_extended/rebuilt_broad_sample.py",
        "command": "rebuilt_broad_sample.py --tahap undi",
        "config_version_lock": json.loads((T1.SHARED / "config_lock.json").read_text())["config_version"],
        "purpose": "broad-track groups whose partner golds answer the question",
        "rules": {"population": "train and dev queries with exactly one gold passage, in the "
                                "asker's program",
                  "question": "judge rates the question clear ('ya')",
                  "full_answer": "'penuh' under both answer wordings; wording B is asked only "
                                 "when wording A says 'penuh'",
                  "own_gold": "a full answer",
                  "candidates_per_program": f"top {K_PER_DIRECTION} by cosine with the question, "
                                            f"union top {K_PER_DIRECTION} by cosine with the gold",
                  "near_duplicate_of_gold": {"cosine_above": T1.UPPER_THRESHOLD,
                                             "word_jaccard_above": T1.JACCARD_MAX},
                  "partner": "first full answer in order of cosine with the question, one per "
                             "program",
                  "near_duplicate_of_partner": "a candidate above either limit with a partner "
                                               "already taken is skipped unjudged",
                  "max_partners": MAX_MATCHES, "pool_size": T1.POOL_SIZE,
                  "pool_centre": "gold chunk embedding", "corpus_min_chars": 200,
                  "judge_passage": "chunk text cut at 4,000 characters, as in the pool"},
        "judge": {"model": JUDGE_MODEL, "prompt_version": JUDGE_VERSION, "sampling": SAMPLING,
                  "thinking": False, "prompts": PROMPT,
                  "parse": "first word of the reply after any think block; a reply that does "
                           "not start with an allowed word counts as a negative verdict",
                  "server_runs": [json.loads(b) for b in SERVER.read_text().split("\n") if b.strip()]
                  if SERVER.exists() else [],
                  "verdicts_used": len(judge.used),
                  "unreadable_replies_used": unreadable, "empty_replies_used": empty,
                  "unreadable_limit": MAX_UNREADABLE,
                  "verdict_file": {"path": "results/raw_r2/juri_r2.jsonl", "sha256": sha(JUDGE),
                                   "verdicts_stored": len(judge.cache)}},
        "prompt_stability": stable,
        "first_kit_judged": {"what": "both wordings and the clarity prompt on every passage of the "
                                     "rating kit drawn from the first broad sample (55 groups); "
                                     "selects nothing",
                             "by_role": {r: {"wording_a": compute([x for x in old_kit if x["role"] == r],
                                                                 lambda x: x["a"]),
                                             "full_both_wordings": sum(x["penuh"] for x in old_kit
                                                                       if x["role"] == r),
                                             "items": sum(1 for x in old_kit if x["role"] == r)}
                                         for r in sorted({x["role"] for x in old_kit})}},
        "sampling": {"design": "stratified random sampling without replacement, or every "
                               "group when at most n_groups qualify",
                     "strata": "gate version x asker program",
                     "allocation": "proportional to groupable queries, largest remainder",
                     "n_groups": N_GROUPS, "census": census, "seed": SEED,
                     "rng": "numpy.random.default_rng (PCG64)", "numpy_version": np.__version__},
        "population": {"eligible_r1_rule": len(population),
                       "exclusions": compute(population, lambda p: p["alasan"] or "groupable"),
                       "more_than_one_gold_by_class": compute(
                           [p for p in population if p["alasan"] == "more_than_one_gold"],
                           lambda p: p["q"].get("class_label")),
                       "question_clear": compute(single, lambda p: p["h"]["jelas"]),
                       "own_gold_wording_a": compute([p for p in single if p["h"]["jelas"] == "ya"],
                                                    lambda p: p["h"]["gold_a"]),
                       "own_gold_full_in_a_not_b": sum(1 for p in single if p["h"]["gold_a"] == "penuh"
                                                       and not p["h"]["gold_penuh"]),
                       "partner_candidates_judged": len(rated),
                       "partner_candidates_wording_a": compute(rated, lambda x: x["a"]),
                       "partner_candidates_full_in_a_not_b": sum(1 for x in rated if x["a"] == "penuh"
                                                                 and x["b"] != "penuh"),
                       "candidates_skipped_as_near_duplicate_of_a_partner": sum(
                           len(p["h"]["kembar"]) for p in single),
                       "partners_found": compute([p for p in single if p["h"]["gold_penuh"]],
                                                lambda p: len(p["h"]["padanan"])),
                       "groupable_queries": len(usable),
                       "groupable_by_gate": compute(usable, lambda p: p["gate"]),
                       "groupable_by_stratum": {f"{g}|{dp}": v for (g, dp), v in sorted(sizes.items())},
                       "allocation_by_stratum": {f"{g}|{dp}": v for (g, dp), v in sorted(alloc.items())}},
        "sample": {"n_groups": len(selected), "n_rows": len(rows),
                   "pairs": sum(1 for p in selected if len(p["h"]["padanan"]) == 1),
                   "triples": sum(1 for p in selected if len(p["h"]["padanan"]) == 2),
                   "by_gate": compute(selected, lambda p: p["gate"]),
                   "by_program": compute(selected, lambda p: p["dept"]),
                   "by_class": compute(selected, lambda p: p["q"].get("class_label")),
                   "overlap_with_original_sample": len(selected_q & old_origin),
                   "overlap_with_r1_sample": len(selected_q & r1_origin),
                   "query_ids": [p["q"]["query_id"] for p in selected]},
        "preregistration": {"path": "result/paper_q2/PRAREGISTRASI_R1_IJIES.md"},
        "code": {"miner_sha256": sha(SCRIPT), "miner_commit": st["script"]["commit"],
                 "frozen_rules": {"path": "study/channel/scripts/9_revision/revision_broad_sample.py",
                                  "sha256": sha(Path(T1.__file__))}},
        "inputs": {"corpus": sha(T1.CORPUS), "queries": sha(T1.QUERIES),
                   "embedding_cache": sha(T1.CACHE), "question_embeddings": sha(EMB_Q),
                   "split_manifest": sha(T1.SHARED / "split_manifest.json"),
                   "first_kit_key": sha(OLD_KIT_KEY), "original_broad_pool": sha(T1.OLD_POOL)},
        "holdout_check": {"source_queries": len(selected_q), "outside_train_dev": 0},
        "output": {"path": "results/raw_r2/pools_xpair_r2.jsonl", "sha256": sha(tmp),
                   "population_table": {"path": "results/tables/r2_population_broad.json",
                                        "sha256": sha(POPULATION)}},
        "label_source": "Qwen3-14B-AWQ judge (question clarity and answerability, two wordings), "
                        "embedding similarity (candidates and pool), seeded draw; the raters' "
                        "ratings, not the judge, are the reference for validity",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    tmp.replace(POOL_R2)
    print(json.dumps({k: manifest[k] for k in ("population", "prompt_stability", "sample")},
                     indent=1, ensure_ascii=False)[:6000])
    print(f"written: {POOL_R2}")


def pool_provenance(p: Path) -> set:
    if not p.exists():
        return set()
    return {json.loads(b)["provenance"]["kueri_asal"] for b in p.open(encoding="utf-8") if b.strip()}


def check_rows(rows: list, d):
    problems = []
    per_group = defaultdict(list)
    for r in rows:
        per_group[r["pair_id"]].append(r)
        cand = {c["id"] for c in r["cands"]}
        if not set(r["gold"]) <= cand or not set(r["counterpart"]) <= cand:
            problems.append(f"{r['query_id']}: a gold is missing from its pool")
        if d.split.get(r["provenance"]["kueri_asal"]) not in ("train", "dev"):
            problems.append(f"{r['query_id']}: source query outside train and dev")
    for g, rs in per_group.items():
        prog = [r["profile"]["department"] for r in rs]
        if len(rs) not in (2, 3) or len(set(prog)) != len(prog):
            problems.append(f"{g}: {len(rs)} rows, programs {prog}")
    if len({r["query_id"] for r in rows}) != len(rows):
        problems.append("duplicate row ids")
    if problems:
        raise SystemExit("ABORT: pool check failed, nothing written:\n  " + "\n  ".join(problems[:20]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tahap", choices=["cek", "embed", "juri", "undi"], required=True)
    ap.add_argument("--paralel", type=int, default=PARALLEL,
                    help="parallel requests to the vLLM server (tahap juri)")
    ap.add_argument("--batas", type=int, default=None,
                    help="tahap juri: judge only the first N eligible queries (smoke run)")
    args = ap.parse_args()
    if args.batas is not None and args.batas < 1:
        ap.error("--batas must be at least 1")
    d = prepare_population()
    if args.tahap == "cek":
        stage_check(d)
    elif args.tahap == "embed":
        stage_embed(d)
    elif args.tahap == "juri":
        stage_judge(d, args.paralel, args.batas)
    else:
        stage_draw(d)


if __name__ == "__main__":
    main()
