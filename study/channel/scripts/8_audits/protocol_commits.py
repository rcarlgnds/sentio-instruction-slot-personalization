import os
import glob
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
RAW = ROOT / "study/channel/results/raw"
OUT = ROOT / "study/channel/results/tables/protocol_commits.json"
MANIFEST = RAW / "protocol_commits.manifest.json"
DOCS = ROOT / "script/paper1_code_docs/protocol/COMMITS.md"
PACKAGE_NAME = {
    "PRAREGISTRASI_KAPPA_JUDGE.md": "judge_agreement",
    "PRAREGISTRASI_STRICT_TRAIN.md": "strict_train",
    "PRAREGISTRASI_LINTAS_ARSITEKTUR.md": "cross_architecture",
    "PRAREGISTRASI_LINTAS_ARSITEKTUR_ADENDUM2.md": "cross_architecture_addendum2",
    "PRAREGISTRASI_LINTAS_ARSITEKTUR_ADENDUM3.md": "cross_architecture_addendum3",
    "PRAREGISTRASI_LINTAS_ARSITEKTUR_ADENDUM4.md": "cross_architecture_addendum4",
    "PRAREGISTRASI_TANGGA_JARAK_INFERENSI.md": "marker_ladder",
    "PRAREGISTRASI_TANGGA_ADENDUM5_PERMUKAAN.md": "surface_ladder",
    "PRAREGISTRASI_ADENDUM_KESETARAAN.md": "scorer_equivalence",
    "PRAREGISTRASI_PROFILE_SHUFFLE.md": "profile_shuffle",
    "PRAREGISTRASI_AUTO_KURASI.md": "auto_curation",
}

CELL_COVERAGE = {
    "PRAREGISTRASI_LINTAS_ARSITEKTUR.md": "scores_[gcf]*_paired_*.manifest.json",
    "PRAREGISTRASI_LINTAS_ARSITEKTUR_ADENDUM2.md": "scores_[gcf]*_xpair_*.manifest.json",
    "PRAREGISTRASI_ADENDUM_KESETARAAN.md": "scores_zerank2official_*.manifest.json",
    "PRAREGISTRASI_TANGGA_JARAK_INFERENSI.md": "scores_*_tangga[0-9a-z]*[!P]*.manifest.json",
    "PRAREGISTRASI_TANGGA_ADENDUM5_PERMUKAAN.md": "scores_*_tanggaP*.manifest.json",
    "PRAREGISTRASI_LINTAS_ARSITEKTUR_ADENDUM3.md": "scores_*_paired_slot_vendor.manifest.json",
    "PRAREGISTRASI_LINTAS_ARSITEKTUR_ADENDUM4.md": ["scores_*_paired_slot_id.manifest.json",
                                                   "scores_*_paired_profile_id.manifest.json"],
}
NOT_CHANNEL = {"PRAREGISTRASI_KAPPA_JUDGE.md", "PRAREGISTRASI_STRICT_TRAIN.md",
               "PRAREGISTRASI_PROFILE_SHUFFLE.md", "PRAREGISTRASI_AUTO_KURASI.md",
               "PRAREGISTRASI_HOLDOUT_GENERASI.md"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def first_commit(path: Path):
    out = git("log", "--diff-filter=A", "--format=%H %aI", "--", str(path.relative_to(ROOT)))
    h, t = out.splitlines()[-1].split()
    return h, datetime.fromisoformat(t).astimezone(timezone.utc)


def first_cell(pattern):
    ts = []
    patterns = pattern if isinstance(pattern, list) else [pattern]
    for m in [x for pl in patterns for x in glob.glob(str(RAW / pl))]:
        if "smoke" in m:
            continue
        rid = json.loads(Path(m).read_text()).get("run_id") or ""
        g = re.search(r"(\d{8}T\d{6}Z)", rid)
        if g:
            ts.append((datetime.strptime(g.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc),
                       Path(m).name))
    return min(ts) if ts else (None, None)


def main():
    if not (ROOT / ".git").exists():
        raise SystemExit("STOP: protocol_commits.py reads the git history of the authors' working "
                         "repository, which does not ship with this package. The frozen "
                         "protocol_commits.json is left as it is.")
    remote = None
    try:
        remote = git("log", "-1", "--format=%H %aI", "origin/main")
    except subprocess.CalledProcessError:
        pass

    rows = []
    for p in sorted({q.resolve() for q in ROOT.glob("result/*/PRAREGISTRASI_*.md")}):
        h, t = first_commit(p)
        di_remote = bool(git("branch", "-r", "--contains", h)) if remote else False
        r = {"dokumen": p.name, "folder": p.parent.name, "commit": h, "commit_utc": t.isoformat(),
             "sudah_di_remote": di_remote}
        if p.name in CELL_COVERAGE:
            timestamp, name = first_cell(CELL_COVERAGE[p.name])
            r["sel_pertama_utc"] = timestamp.isoformat() if timestamp else None
            r["sel_pertama_manifest"] = name
            r["selang_menit"] = round((timestamp - t).total_seconds() / 60, 1) if timestamp else None
            r["protokol_mendahului_sel"] = bool(timestamp and t < timestamp)
        rows.append(r)

    if not rows:
        raise SystemExit("STOP: no protocol document found under result/*/; nothing is written.")
    failed = [r["dokumen"] for r in rows if r.get("protokol_mendahului_sel") is False]
    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "_apa_ini": ("First commit hash of every pre-registration document and, for the channel protocols, "
                     "proof that the commit precedes the first cell it governs."),
        "batas": [
            "git commit times are recorded by the authors' own machine, so they order events but do "
            "not certify them to a third party",
            "a row whose sudah_di_remote is false was not on the remote when it was committed. The last push to "
            "origin/main: "
            + (remote or "unreadable"),
        ],
        "protokol": rows,
        "protokol_kanal_yang_gagal": failed,
        "label_source": "deterministic (git log and the run_id of each manifest)",
    }
    OUT.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    MANIFEST.write_text(json.dumps({
        "run_id": "protocol_commits_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/8_audits/protocol_commits.py",
        "output": {"path": "results/tables/protocol_commits.json", "sha256": sha(OUT)},
        "label_source": "deterministic",
    }, indent=1))
    write_commits_md(rows, remote)
    for r in rows:
        extra = (f"  first cell +{r['selang_menit']} min" if r.get("selang_menit") is not None else "")
        print(f"{r['dokumen']:<46} {r['commit'][:10]} {r['commit_utc'][:16]}Z{extra}")
    print("channel protocols whose cells precede the commit:", failed or "none")


def write_commits_md(rows, remote):
    L = ["# Commit hashes of the protocols", "",
         "Each protocol below was committed to the authors' git repository before the",
         "measurements it governs. The hash is that of the commit that first added the",
         "document.",
         "For the seven protocols that govern channel cells,",
         "the last column gives how long before the first governed cell the commit was",
         "made, taken from the run_id in each cell's manifest.", "",
         "Two limits. Commit times are recorded by the authors' own machine, so they",
         "order events without certifying them to a third party. A row whose",
         "`sudah_di_remote` is false in `protocol_commits.json` was not on any public",
         "remote when it was committed.", "",
         "| protocol | commit | committed (UTC) | before first governed cell |",
         "| --- | --- | --- | --- |"]
    for r in rows:
        name = PACKAGE_NAME.get(r["dokumen"])
        if not name:
            continue
        if r.get("protokol_mendahului_sel"):
            before = f"yes, {r['selang_menit']:.0f} minutes"
        elif r.get("protokol_mendahului_sel") is False:
            before = "no"
        elif r["dokumen"] in NOT_CHANNEL:
            before = "governs no channel cell"
        else:
            before = "scope not mapped"
        L.append(f"| `{name}` | `{r['commit'][:12]}` | {r['commit_utc'][:16].replace('T', ' ')} | {before} |")
    L += ["", "Generated by `protocol_commits.py` from `protocol_commits.json`.", ""]
    DOCS.write_text("\n".join(L))


if __name__ == "__main__":
    main()
