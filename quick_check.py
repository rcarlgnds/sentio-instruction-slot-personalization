"""One command that checks the whole package. No GPU or corpus needed.

It checks MANIFEST.sha256 and the imports, reruns the 25 analysis scripts that work from the shipped
files, and requires every number in their 29 tables to match the saved tables. It then runs
verify_paper_numbers.py on the fresh tables and checks the demo pool.

    python quick_check.py
"""

import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RUNNABLE = [
    "study/channel/scripts/0_common/metrics.py",
    "study/channel/scripts/4_slot_vs_baseline/equivalence_bounds.py",
    "study/channel/scripts/3_slot_changes/flip_rate.py",
    "study/channel/scripts/4_slot_vs_baseline/mcnemar.py",
    "study/channel/scripts/4_slot_vs_baseline/aggregate_channel_board.py",
    "study/channel/scripts/8_audits/validity_audit.py",
    "study/channel/scripts/6_label_check/single_field_label.py",
    "study/channel/scripts/4_slot_vs_baseline/channel_tests_full.py",
    "study/channel/scripts/3_slot_changes/asker_hit_mrr.py",
    "study/channel/scripts/8_audits/gpu_cost.py",
    "study/channel/scripts/4_slot_vs_baseline/mined_pairs_architectures.py",
    "study/channel/scripts/5_why_slot_fails/slot_vendor_style.py",
    "study/channel/scripts/5_why_slot_fails/slot_indonesian.py",
    "study/channel/scripts/5_why_slot_fails/followir_calibration.py",
    "study/channel/scripts/4_slot_vs_baseline/cluster_bootstrap.py",
    "study/channel/scripts/4_slot_vs_baseline/aggregate_architecture_board.py",
    "study/channel/scripts/8_audits/pair_integrity_audit.py",
    "study/channel/scripts/8_audits/comparability_audit.py",
    "study/channel/scripts/6_label_check/inference_ladder.py",
    "study/channel/scripts/6_label_check/inference_ladder_surface.py",
    "study/channel/scripts/6_label_check/surface_ladder_p1_p2.py",
    "study/channel/scripts/9_revision/revision_analysis.py --bagian klaster",
    "study/channel/scripts/9_revision/revision_analysis.py --bagian paired",
    "study/channel/scripts/9_revision/revision_analysis.py --bagian seragam",
    "study/channel/scripts/9_revision/revision_analysis.py --bagian gerbang",
    "study/channel/scripts/9_revision/revision_analysis.py --bagian gemma",
    "study/channel/scripts/9_revision/revision_analysis.py --bagian ortogonal",
    "study/channel/scripts/9_revision/revision_analysis.py --bagian sensitivitas",
    "study/channel/scripts/10_rebuilt_and_extended/rebuilt_broad_analysis.py",
    "study/channel/scripts/10_rebuilt_and_extended/extended_paired_analysis.py",
    "study/channel/scripts/11_human_ratings/human_ratings_analysis.py",
]

TABLES = [
    "study/channel/results/tables/equivalence_bounds.json",
    "study/channel/results/tables/flip_rate.json",
    "study/channel/results/tables/mcnemar_paired.json",
    "study/channel/results/tables/channel_board_paired.json",
    "study/channel/results/tables/channel_board_xpair.json",
    "study/channel/results/tables/validity_audit.json",
    "study/channel/results/tables/single_field_label.json",
    "study/channel/results/tables/channel_tests_full.json",
    "study/channel/results/tables/asker_hit_mrr.json",
    "study/channel/results/tables/gpu_cost.json",
    "study/channel/results/tables/channel_board_xpair_architectures.json",
    "study/channel/results/tables/vendor_style_slot.json",
    "study/channel/results/tables/slot_id_paired.json",
    "study/channel/results/tables/followir_calibration.json",
    "study/channel/results/tables/cluster_bootstrap.json",
    "study/channel/results/tables/channel_board_paired_architectures.json",
    "study/channel/results/tables/marker_ladder.json",
    "study/channel/results/tables/surface_ladder.json",
    "study/channel/results/tables/surface_ladder_p1_p2.json",
    "study/channel/results/tables/r1_cluster_broad.json",
    "study/channel/results/tables/r1_cluster_paired.json",
    "study/channel/results/tables/r1_uniform_semester.json",
    "study/channel/results/tables/r1_gate_g0.json",
    "study/channel/results/tables/r1_gemma_vendor.json",
    "study/channel/results/tables/r1_slot_orthogonal.json",
    "study/channel/results/tables/r1_gold_sensitivity.json",
    "study/channel/results/tables/r2_broad.json",
    "study/channel/results/tables/p2_paired.json",
    "study/channel/results/tables/human_ratings.json",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def local_envs() -> list:
    return [p.parent for p in ROOT.rglob("pyvenv.cfg")]


def local_only(p: Path, envs) -> bool:
    inside = p.relative_to(ROOT).parts[:-1]
    return any(e == p or e in p.parents for e in envs) or any(x.startswith(".") for x in inside)


def skip_in_copy(folder, names) -> list:
    return [n for n in names if n == "__pycache__"
            or (n.startswith(".") and (Path(folder) / n).is_dir())
            or (Path(folder) / n / "pyvenv.cfg").is_file()]


def check_manifest() -> int:
    if not (ROOT / "MANIFEST.sha256").is_file():
        print("   MANIFEST.sha256 is missing. Use a released copy of the package, or rebuild it.")
        return 1
    manifest = ROOT / "MANIFEST.sha256"
    bad = 0
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        want, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*")
        listed.add(rel)
        p = ROOT / rel
        if not p.is_file():
            print(f"  MISSING {rel}")
            bad += 1
        elif sha256(p) != want:
            print(f"  MISMATCH {rel}")
            bad += 1
    envs = local_envs()
    extra = [
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file()
        and not local_only(p, envs)
        and p.relative_to(ROOT).as_posix() not in listed
        and p.name != "MANIFEST.sha256"
        and "__pycache__" not in p.parts
        and ".pytest_cache" not in p.parts
        and ".git" not in p.parts
        and p.name not in (".DS_Store", "Thumbs.db", "desktop.ini")
    ]
    for rel in extra:
        print(f"  NOT IN MANIFEST {rel}")
    print(f"  {len(listed)} files listed, {bad} problems, {len(extra)} unlisted")
    return bad + len(extra)


ALIAS = {"pymupdf": "fitz"}


def declared_packages() -> set:
    names = set()
    for rel in ("system/requirements.txt", "study/requirements-analysis.txt", "study/requirements-corpus.txt",
                "study/requirements-scoring.txt"):
        f = ROOT / rel
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            dist = re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip().lower()
            if not dist:
                continue
            names.add(ALIAS.get(dist, dist.replace("-", "_")))
    return names


def resolve_app(module: str, names=()) -> list:
    base = ROOT / "system" / Path(*module.split("."))
    if base.with_suffix(".py").is_file():
        return []
    init = base / "__init__.py"
    if not init.is_file():
        return [module]
    text = init.read_text(encoding="utf-8", errors="replace")
    return [f"{module}.{n}" for n in names
            if n != "*" and not (base / f"{n}.py").is_file()
            and not (base / n / "__init__.py").is_file()
            and not re.search(rf"\b{re.escape(n)}\b", text)]


def check_imports() -> int:
    third = {n.lower() for n in declared_packages()}
    std = {n.lower() for n in sys.stdlib_module_names}
    envs = local_envs()
    dirs = {d.name.lower() for d in ROOT.rglob("*")
            if d.is_dir() and "__pycache__" not in d.parts and not local_only(d, envs)}
    bad = 0
    scripts = sorted(p for p in ROOT.rglob("*.py")
                     if "__pycache__" not in p.parts and not local_only(p, envs))
    anywhere = {q.stem.lower() for q in scripts}
    for p in scripts:
        src = p.read_text(encoding="utf-8", errors="replace")
        local = ({q.stem.lower() for q in p.parent.glob("*.py")}
                 | (anywhere if "sys.path.insert" in src else set()))
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            print(f"  SYNTAX {p.relative_to(ROOT)}: {e}")
            bad += 1
            continue
        guarded = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Try) and n.handlers:
                for inner in ast.walk(n):
                    guarded.add(id(inner))
        names = set()
        app_imports = []
        for n in ast.walk(tree):
            if id(n) in guarded:
                continue
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.split(".")[0] == "app":
                        app_imports.append((a.name, ()))
                    else:
                        names.add(a.name.split(".")[0])
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                if n.module.split(".")[0] == "app":
                    app_imports.append((n.module, tuple(a.name for a in n.names)))
                else:
                    names.add(n.module.split(".")[0])
        for m in sorted(names):
            k = m.lower()
            if k in std or k in third or k in local or k in dirs:
                continue
            print(f"  UNRESOLVED IMPORT {p.relative_to(ROOT)}: {m}")
            bad += 1
        for module, imported in app_imports:
            for missing in resolve_app(module, imported):
                print(f"  UNRESOLVED IMPORT {p.relative_to(ROOT)}: {missing}")
                bad += 1
    print(f"  {len(scripts)} files parsed, {bad} unresolved")
    return bad


def numbers(obj, path=""):
    if isinstance(obj, bool):
        yield path, obj
    elif isinstance(obj, (int, float)):
        yield path, obj
    elif isinstance(obj, dict):
        for k in sorted(obj):
            yield from numbers(obj[k], f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from numbers(v, f"{path}[{i}]")


IGNORE = {
    "study/channel/results/tables/channel_board_paired_architectures.json": ("/waktu_gpu",),
}


def compare_numeric(frozen: Path, rerun: Path, ignore=()):
    a = dict(numbers(json.loads(frozen.read_text(encoding="utf-8"))))
    b = dict(numbers(json.loads(rerun.read_text(encoding="utf-8"))))
    diffs = []
    for k in sorted(set(a) | set(b)):
        if any(k.startswith(prefix) for prefix in ignore):
            continue
        if a.get(k) != b.get(k):
            diffs.append(f"{k}: frozen={a.get(k)} rerun={b.get(k)}")
    return diffs


def main() -> int:
    missing = [m for m in ("numpy", "scipy") if importlib.util.find_spec(m) is None]
    if missing:
        print(f"{' and '.join(missing)} not installed for {sys.executable}")
        print("   install with: python -m pip install -r study/requirements-check.txt")
        print("   (on Python 3.10, install numpy<2 and scipy<1.13 instead)")
        return 1
    print("1. manifest")
    bad = check_manifest()
    if bad:
        print("manifest FAILED")
        return 1
    print("   PASS")

    print("2. imports resolve")
    bad = check_imports()
    if bad:
        print("import check FAILED")
        return 1
    print("   PASS")

    print("3. rerun in a scratch copy")
    tmp = Path(tempfile.mkdtemp(prefix="quick_check_"))
    failures = 0
    try:
        work = tmp / "pkg"
        shutil.copytree(ROOT, work, ignore=skip_in_copy)
        env = {k: v for k, v in os.environ.items() if k != "SENTIO_ROOT"}
        for rel in RUNNABLE:
            r = subprocess.run(
                [sys.executable, str(work / rel.split()[0]), *rel.split()[1:]],
                cwd=work, capture_output=True, text=True, timeout=900, env=env,
            )
            status = "ok" if r.returncode == 0 else f"rc={r.returncode}"
            print(f"   {rel}: {status}")
            if r.returncode != 0:
                failures += 1
                tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
                for line in tail:
                    print(f"       {line}")
        if failures:
            print("   the scripts that failed did not rewrite their tables, so the comparison below "
                  "covers only the scripts that ran")
        for rel in TABLES:
            diffs = compare_numeric(ROOT / rel, work / rel, IGNORE.get(rel, ()))
            if diffs:
                failures += 1
                print(f"   NUMERIC DIFF {rel}")
                for d in diffs[:5]:
                    print(f"       {d}")
            else:
                skipped = f" ({', '.join(IGNORE[rel])} not compared)" if rel in IGNORE else ""
                print(f"   numbers identical: {rel}{skipped}")

        print("4. paper numbers on the rerun copy")
        r = subprocess.run(
            [sys.executable, str(work / "verify_paper_numbers.py"), str(work), str(tmp / "verify_report.md")],
            cwd=work, capture_output=True, text=True, timeout=900,
        )
        lines = r.stdout.strip().splitlines()
        print(f"   {lines[0].split('; report')[0] if lines else 'no output'}")
        if r.returncode != 0:
            failures += 1
            for line in lines[1:6] or (r.stderr or "").strip().splitlines()[-3:]:
                print(f"       {line}")

        print("5. demo pool of the scorers (no GPU)")
        r = subprocess.run(
            [sys.executable, str(work / "study/channel/scripts/2_scoring/demo_small_pool.py"), "--dry-run"],
            cwd=work, capture_output=True, text=True, timeout=300,
        )
        lines = r.stdout.strip().splitlines()
        print(f"   {lines[-1] if lines else 'no output'}")
        if r.returncode != 0:
            failures += 1
            for line in (r.stderr or "").strip().splitlines()[-3:]:
                print(f"       {line}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print(f"quick check FAILED, {failures} problems")
        return 1
    print("quick check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
