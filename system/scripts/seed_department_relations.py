import argparse

from app.storage.postgres_client import PostgresAdapter

STRONG = 0.7
MODERATE = 0.5
UNRELATED = 0.3


MATRIX = {
    ("DTSC", "STAT"): (STRONG, "Core-curriculum overlap in B30 0.67, the highest of the 21 measured pairs. Data Science and Statistics-CS share a statistics and computing path."),
    ("MGMT", "MKTG"): (STRONG, "Core-curriculum overlap in B30 0.65. Both BBSUP and Global Business Marketing rest on a foundation of management courses."),
    ("BANA", "STAT"): (STRONG, "Core-curriculum overlap in B30 0.54. Business Analytics uses the same sequence of statistics courses."),
    ("BANA", "DTSC"): (STRONG, "Core-curriculum overlap in B30 0.54. Business Analytics uses the same sequence of data science courses."),
    ("ACCT", "MGMT"): (STRONG, "Codes starting with ACCT appear in the Management catalog and in none of the five non-business programs. Exact corpus count: 83.7 percent of the 9,067 chunks labeled ACCT come from the Entrepreneurship book (42.5 percent) and Principles of Management (41.2 percent), only 16.3 percent from accounting books."),

    ("CSCI", "DTSC"): (MODERATE, "Core-curriculum overlap in B30 0.28. Both are SOCS programs, and Data Science uses programming courses from Computer Science."),
    ("CSCI", "STAT"): (MODERATE, "Core-curriculum overlap in B30 0.26. The Statistics and Computer Science program combines both families."),
    ("BANA", "ISYS"): (MODERATE, "Core-curriculum overlap in B30 0.21. Both are programs of the School of Information Systems."),
    ("CSCI", "ISYS"): (MODERATE, "Measured core-curriculum overlap 0.02, but the ISYS study-plan extraction read only 58 courses and is probably truncated. On the full catalog the overlap is 0.38. A middle value was chosen because the two measurements disagree."),
    ("ACCT", "MKTG"): (MODERATE, "Codes starting with ACCT appear in the Global Business Marketing catalog. There is no support from the corpus side as for the ACCT-MGMT pair, so the evidence is thinner."),

    ("ISYS", "MKTG"): (UNRELATED, "Core-curriculum overlap in B30 0.05."),
    ("ISYS", "MGMT"): (UNRELATED, "Core-curriculum overlap in B30 0.03. The vision of the Information Systems program names an IT-business orientation, but its core courses barely overlap with Management."),
    ("BANA", "MGMT"): (UNRELATED, "Core-curriculum overlap in B30 0.03. Despite its name, the core curriculum of Business Analytics leans toward the analytics family, not management."),
    ("BANA", "MKTG"): (UNRELATED, "Core-curriculum overlap in B30 0.02."),
    ("CSCI", "MGMT"): (UNRELATED, "Core-curriculum overlap in B30 0.00. No core course is shared."),
    ("CSCI", "MKTG"): (UNRELATED, "Core-curriculum overlap in B30 0.00. No core course is shared."),
    ("BANA", "CSCI"): (UNRELATED, "Core-curriculum overlap in B30 0.00. No core course is shared."),
    ("ISYS", "STAT"): (UNRELATED, "Core-curriculum overlap in B30 0.00. No core course is shared."),
    ("MGMT", "STAT"): (UNRELATED, "Core-curriculum overlap in B30 0.00. No core course is shared."),
    ("MKTG", "STAT"): (UNRELATED, "Core-curriculum overlap in B30 0.00. No core course is shared."),
    ("DTSC", "ISYS"): (UNRELATED, "Core-curriculum overlap in B30 0.00. No core course is shared."),
    ("DTSC", "MGMT"): (UNRELATED, "Core-curriculum overlap in B30 0.00. No core course is shared."),
    ("DTSC", "MKTG"): (UNRELATED, "Core-curriculum overlap in B30 0.00. No core course is shared."),

    ("ACCT", "BANA"): (UNRELATED, "No code starting with ACCT in the Business Analytics catalog."),
    ("ACCT", "CSCI"): (UNRELATED, "No code starting with ACCT in the Computer Science catalog."),
    ("ACCT", "DTSC"): (UNRELATED, "No code starting with ACCT in the Data Science catalog."),
    ("ACCT", "ISYS"): (UNRELATED, "No code starting with ACCT in the Information Systems catalog."),
    ("ACCT", "STAT"): (UNRELATED, "No code starting with ACCT in the Statistics and Computer Science catalog."),
}

CHIN_NOTE = ("Chinese Literature has no program catalog in B30 and no chunk in the corpus, so there is no measurable evidence of relatedness.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write to the database")
    args = ap.parse_args()

    pg = PostgresAdapter()
    with pg._conn() as c:
        with c.cursor() as cur:
            cur.execute("select id, name from departments where deleted_at is null")
            rows = cur.fetchall()
    ids = {r["name"]: str(r["id"]) for r in rows}
    names = sorted(ids)

    plan = {}
    for (a, b), (score, note) in MATRIX.items():
        if a not in ids or b not in ids:
            print(f"SKIPPED: {a}-{b} is not in the database")
            continue
        plan[tuple(sorted([a, b]))] = (score, note)
    for other in names:
        if other != "CHIN" and "CHIN" in ids:
            plan[tuple(sorted(["CHIN", other]))] = (UNRELATED, CHIN_NOTE)

    expected = len(names) * (len(names) - 1) // 2
    print(f"{len(names)} departments -> {expected} pairs, the plan holds {len(plan)}")
    missing = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if (names[i], names[j]) not in plan:
                missing.append((names[i], names[j]))
    if missing:
        print(f"NOT YET COVERED ({len(missing)}): {missing}")
    print()
    for (a, b), (s, note) in sorted(plan.items(), key=lambda x: (-x[1][0], x[0])):
        print(f"  {a:5s} <-> {b:5s}  {s}   {note[:72]}")

    if not args.apply:
        print("\n(dry run. add --apply to write)")
        return

    with pg._conn() as c:
        with c.cursor() as cur:
            cur.execute("delete from department_relations")
            deleted = cur.rowcount
            for (a, b), (s, note) in plan.items():
                ida, idb = sorted([ids[a], ids[b]])
                cur.execute(
                    "insert into department_relations (dept_a_id, dept_b_id, relatedness_score, note) "
                    "values (%s,%s,%s,%s)", (ida, idb, s, note.strip()[:500]))
        c.commit()
    print(f"\n{deleted} old rows deleted, {len(plan)} rows written")


if __name__ == "__main__":
    main()
