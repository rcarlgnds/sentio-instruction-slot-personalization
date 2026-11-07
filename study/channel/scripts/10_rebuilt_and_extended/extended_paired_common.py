import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import sys
for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import revision_broad_sample as T1  # noqa: E402

ROOT = T1.ROOT
SCRIPT = Path(__file__).resolve().parent
Q2 = ROOT / "study/channel"
RAW_P2 = Q2 / "results/raw_p2"
TABLE = T1.TABLE
PREREG = ROOT / "result/paper_q2/PRAREGISTRASI_R1_IJIES.md"
OLD_PAIRED = ROOT / "study/channel/provenance/paired_queries.jsonl"

PAIRS = RAW_P2 / "pasangan_topik_p2.jsonl"
PAIRS_MANIFEST = RAW_P2 / "pasangan_topik_p2.manifest.json"
BATCH = RAW_P2 / "batch"
WRITE_FILE = RAW_P2 / "tulis_p2.jsonl"
SCREEN_FILE = RAW_P2 / "saring_p2.jsonl"
JUDGE = RAW_P2 / "juri_p2.jsonl"
JUDGE_SERVER = RAW_P2 / "juri_p2_server.jsonl"
ROUND_FILE = RAW_P2 / "ronde_p2.json"
POOL = RAW_P2 / "pools_p2.jsonl"
POOL_MANIFEST = RAW_P2 / "pools_p2.manifest.json"
POPULATION = TABLE / "p2_populasi.json"
WORKFLOW_ARCHIVE = Path("<home>/bukti_asal_p2")

FOCUS = ("CSCI", "ISYS", "MGMT")
SPARE = "ACCT"
FOCUS_ROTATION = ("ISYS-MGMT", "CSCI-MGMT", "CSCI-ISYS")
ACCT_ROTATION = ("ACCT-MGMT", "ACCT-ISYS", "ACCT-CSCI")
SAME_WORK_EDITIONS = (frozenset({"acct-accountingprinciplesvol2-hermanson-ccby3",
                               "acct-accounting-principles-globaltext-hermanson-ccby3"}),)
SIM_LOW, SIM_HIGH = 0.62, 0.93
MIN_SIM = 0.66
DUP_COS, DUP_JACCARD = T1.UPPER_THRESHOLD, T1.JACCARD_MAX
MIN_TOKEN = 80
MAX_TEXT = 4000
N_ROUNDS = 500
N_BATCH = 25
TARGET_GROUPS = 400
MAX_PAIRS = 6000
POOL_SIZE = T1.POOL_SIZE
SEED_DISPLAY = 20261022
SEED_KIT = 20261020
N_KIT_GROUPS = 60

P2_SCRIPT = ("extended_paired_common.py", "extended_paired_topic_pairs.py", "extended_paired_store.py", "extended_paired_judge.py",
            "extended_paired_assemble.py", "q2_p2_kit.py", "q2_p2_rencana.py", "extended_paired_analysis.py",
            "q2_p2_uji.py", "q2_p2_alur_tulis.js", "q2_p2_orkestra.sh", "q2_p2_langkah.py",
            "q2_p2_jaga.py", "q2_p2_pengawas.py")
R1_MODULE = ("revision_broad_sample.py", "q2_r1_rencana.py", "q2_r1_runner.py", "revision_broad_kit.py",
            "q2_r1_kit_paired.py", "q2_r1_petunjuk_pdf.py", "revision_analysis.py", "revision_statistics.py",
            "q2_r1_pengawas.py", "inference_ladder.py", "metrics.py", "mined_pairs.py",
            "score_arms.py", "score_architectures.py", "rebuilt_broad_sample.py")
JUDGE_RULE = "p2"
REQUIRED_TEXT = (f"seed {SEED_KIT}", f"seed {SEED_DISPLAY}", "juri_r2_v1")


def sha(p: Path) -> str:
    return T1.sha(Path(p))


def iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    st = {"missing_text": [s for s in REQUIRED_TEXT if s not in flat],
          "sha256": sha(PREREG) if PREREG.exists() else None,
          "preregistration": git_status(PREREG),
          "scripts": {n: git_status(SCRIPT / n) for n in P2_SCRIPT},
          "r1_modules": {n: git_status(SCRIPT / n) for n in R1_MODULE if (SCRIPT / n).exists()}}
    st["ready"] = (not st["missing_text"] and st["preregistration"]["clean"]
                   and all(v["clean"] for v in st["scripts"].values())
                   and all(v["clean"] for v in st["r1_modules"].values() if v["tracked"]))
    return st


def require_preregistration() -> dict:
    st = preregistration_status()
    if not st["ready"]:
        raise SystemExit("ABORT: the p2 rules must be in the pre-registration, and the pre-registration and "
                         "every p2 script must be committed without later edits (tracked R1 and r2 modules "
                         "unchanged) before this step:\n"
                         + json.dumps(st, indent=1, ensure_ascii=False))
    return st


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(Path(p).resolve())


def read_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(b) for b in p.read_text(encoding="utf-8").split("\n") if b.strip()]


def write_json(p: Path, content) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(content, indent=1, ensure_ascii=False), encoding="utf-8")
