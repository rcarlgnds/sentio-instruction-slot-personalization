import hashlib

import sys
from pathlib import Path
for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
from score_arms import wrong_dept, source_label  # noqa: E402

FIXED_PROGRAMS_ALLOWED = ("CSCI", "STAT")
FIXED_SEMESTERS_ALLOWED = (1, 2)


def _dept(p: dict) -> str:
    return p.get("department") or p.get("department_name") or "unknown"


def _sem(p: dict):
    return p.get("current_semester") or p.get("semester") or "?"


def program_only_instruction(p: dict) -> str:
    return (f"Given a query from a {_dept(p)} student, retrieve passages from "
            f"that student's own study program that answer the query")


def semester_only_instruction(p: dict) -> str:
    sem = _sem(p)
    return (f"Given a query from a student in semester {sem}, retrieve passages "
            f"that answer the query at a level appropriate for semester {sem}")


def fixed_profile(p: dict, program: str | None = None,
                 semester: int | None = None) -> dict:
    if program is not None and program not in FIXED_PROGRAMS_ALLOWED:
        raise ValueError(f"fixed program must be one of {FIXED_PROGRAMS_ALLOWED}, got {program!r}")
    if semester is not None and semester not in FIXED_SEMESTERS_ALLOWED:
        raise ValueError(f"fixed semester must be one of {FIXED_SEMESTERS_ALLOWED}, got {semester!r}")
    q = dict(p)
    if program is not None:
        q["department"] = program
        q["department_name"] = program
    if semester is not None:
        q["current_semester"] = semester
        q.pop("semester", None)
    return q


def observed_semesters(meta_map: dict) -> list:
    return sorted({int(m["semester"]) for m in meta_map.values()
                   if m and m.get("semester") is not None})


def wrong_semester(orig_sem, text: str, all_sems: list) -> int:
    other = sorted(s for s in all_sems if s != int(orig_sem))
    if not other:
        raise ValueError("no alternative semester available")
    h = hashlib.sha256(("sem_salah_v1|" + text[:180]).encode()).hexdigest()
    return other[int(h[:8], 16) % len(other)]


def wrong_semester_source_label(m: dict, text: str, all_sems: list) -> str:
    fake = dict(m)
    fake["semester"] = wrong_semester(m.get("semester"), text, all_sems)
    return source_label(fake)


def wrong_both_source_label(m: dict, text: str, all_depts: list, all_sems: list) -> str:
    fake = dict(m)
    fake["department"] = wrong_dept(m.get("department"), text, all_depts)
    fake["semester"] = wrong_semester(m.get("semester"), text, all_sems)
    return source_label(fake)


def random_label(text: str, all_depts: list, all_sems: list) -> str:
    h = hashlib.sha256(("label_acak_v1|" + text[:180]).encode()).hexdigest()
    dept = sorted(all_depts)[int(h[:8], 16) % len(all_depts)]
    sem = sorted(all_sems)[int(h[8:16], 16) % len(all_sems)]
    return source_label({"department": dept, "semester": sem})


def uniform_label(p: dict) -> str:
    return source_label({"department": _dept(p), "semester": _sem(p)})


def revision_label_rules() -> dict:
    return {
        "frame": source_label({"department": "<DEPT>", "semester": "<N>"}),
        "semester_salah": "sha256('sem_salah_v1|' + text[:180]) modulo the sorted "
                          "list of observed semesters other than the passage's own",
        "dept_salah": "sha256('dept_salah_v1|' + text[:180]) modulo the sorted list "
                      "of observed programs other than the passage's own (frozen rule)",
        "acak": "program = sorted(programs)[int(h[0:8], 16) % n_programs], semester = "
                "sorted(semesters)[int(h[8:16], 16) % n_semesters], h = "
                "sha256('label_acak_v1|' + text[:180]); true values not excluded",
        "seragam": "label built from the asker's own profile, identical on every "
                   "candidate of the row",
    }
