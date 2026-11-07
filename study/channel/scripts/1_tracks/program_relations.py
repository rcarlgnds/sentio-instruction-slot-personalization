import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
AI = ROOT / "system"
OUT = ROOT / "study/channel/results/tables/program_relations.json"
MANIFEST = ROOT / "study/channel/results/raw/program_relations.manifest.json"
LOCK = ROOT / "study/_shared/config_lock.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sys.path.insert(0, str(AI))
    os.chdir(AI)
    from app.storage.postgres_client import PostgresAdapter
    from scripts.seed_department_relations import MATRIX, CHIN_NOTE

    pg = PostgresAdapter()
    rel = pg.fetch_department_relations()
    with pg._conn() as c:
        with c.cursor() as cur:
            cur.execute("select id, name from departments where deleted_at is null")
            code = {str(r["id"]): r["name"] for r in cur.fetchall()}

    payload = json.dumps(sorted((a, b, round(float(v), 4)) for (a, b), v in rel.items()))
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:12]
    ref = json.loads(LOCK.read_text())["dept_relations_fingerprint"]
    if fingerprint != ref:
        raise SystemExit(f"STOP: table fingerprint now {fingerprint}, configuration lock {ref}")

    pairs = []
    for (a, b), v in sorted(rel.items(), key=lambda x: (code.get(x[0][0]), code.get(x[0][1]))):
        ka, kb = sorted([code.get(a, a), code.get(b, b)])
        reason = (MATRIX.get((ka, kb)) or MATRIX.get((kb, ka)) or (None, None))[1]
        if reason is None and "CHIN" in (ka, kb):
            reason = CHIN_NOTE
        pairs.append({"prodi_a": ka, "prodi_b": kb, "nilai_m": round(float(v), 4), "alasan": reason})

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "_apa_ini": ("Relatedness values between programs that the proximity term (Prox) of DITS reads, "
                     "frozen from the department_relations table with the fingerprint that matches "
                     "config_lock."),
        "sidik_jari_dihitung_ulang": fingerprint,
        "sidik_jari_config_lock": ref,
        "cocok": True,
        "nilai_prodi_sama": 1.0,
        "tingkat_lintas_prodi": {"0.7": "strong", "0.5": "moderate", "0.3": "unrelated"},
        "sebaran_nilai": dict(Counter(str(p["nilai_m"]) for p in pairs)),
        "n_pasangan": len(pairs),
        "departemen": sorted(code.values()),
        "pasangan": pairs,
        "sumber_bukti": ("system/scripts/seed_department_relations.py: core-curriculum overlap in the B30 catalog "
                         "|A n B| / min(|A|,|B|), presence of codes across programs, corpus composition"),
        "label_source": "deterministic (read from the database, fingerprint matched)",
    }
    OUT.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    MANIFEST.write_text(json.dumps({
        "run_id": "program_relations_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/1_tracks/program_relations.py",
        "output": {"path": "results/tables/program_relations.json", "sha256": sha(OUT)},
        "label_source": "deterministic",
    }, indent=1))
    print(f"fingerprint {fingerprint} = configuration lock {ref}; {len(pairs)} pairs; values {output['sebaran_nilai']}")
    for p in pairs:
        if "CSCI" in (p["prodi_a"], p["prodi_b"]) and "STAT" in (p["prodi_a"], p["prodi_b"]):
            print("  paired-track pair:", p["prodi_a"], p["prodi_b"], p["nilai_m"])


if __name__ == "__main__":
    main()
