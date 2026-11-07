import json
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import channel_cells as K  # noqa: E402
from equivalence_bounds import exact_interval, wald_interval  # noqa: E402
from exact_tests import holm, mcnemar_exact as mcnemar  # noqa: E402

OUT = K.TABLE / "channel_tests_full.json"
MANIFEST = K.RAW / "channel_tests_full.manifest.json"

CONTRASTS = {
    "paired": [("none", "slot"), ("none", "text"), ("slot", "both"), ("text", "both"),
               ("both", "wrong"), ("both", "prog"), ("both", "sem")],
    "xpair": [("none", "slot"), ("slot", "both"), ("both", "wrong")],
}


def main():
    wins = {t: {s: K.manuscript_gate(t, s) for s in K.PAPER[t]} for t in K.PAPER}

    rows = []
    for t, pair in CONTRASTS.items():
        for a, b in pair:
            for m in K.MODEL:
                A, B = wins[t][a][m], wins[t][b][m]
                q = sorted(set(A) & set(B))
                n10 = sum(1 for x in q if A[x] == 1 and B[x] == 0)
                n01 = sum(1 for x in q if A[x] == 0 and B[x] == 1)
                rows.append({"track": t, "kontras": f"{a}_lawan_{b}", "model": m,
                              "n": len(q), "menang_a": sum(A[x] for x in q),
                              "menang_b": sum(B[x] for x in q),
                              "n10_a_menang_b_kalah": n10, "n01_a_kalah_b_menang": n01,
                              "p_mentah_mcnemar_eksak": mcnemar(n10, n01)})

    for t, pair in CONTRASTS.items():
        for a, b in pair:
            group = [r for r in rows if r["track"] == t and r["kontras"] == f"{a}_lawan_{b}"]
            for r, ph in zip(group, holm([r["p_mentah_mcnemar_eksak"] for r in group])):
                r["p_holm_enam_model"] = ph
                r["m_enam_model"] = len(group)
    for r, ph in zip(rows, holm([r["p_mentah_mcnemar_eksak"] for r in rows])):
        r["p_holm_seluruh_tabel"] = ph
        r["m_seluruh_tabel"] = len(rows)
    for r in rows:
        for k in ("p_mentah_mcnemar_eksak", "p_holm_enam_model", "p_holm_seluruh_tabel"):
            r[k] = float(f"{r[k]:.6g}")

    bound = {}
    for t in K.PAPER:
        bound[t] = {}
        for r in rows:
            if r["track"] != t or r["kontras"] != "none_lawan_slot":
                continue
            b_slot, c_blind, n = r["n01_a_kalah_b_menang"], r["n10_a_menang_b_kalah"], r["n"]
            d, wl, wh = wald_interval(b_slot, c_blind, n)
            _, el, eh = exact_interval(b_slot, c_blind, n)
            lo, hi = min(wl, el), max(wh, eh)
            bound[t][r["model"]] = {
                "n": n, "diskordan_slot_menang": b_slot, "diskordan_buta_menang": c_blind,
                "titik_poin_persen": round(d * 100, 2),
                "wald95_poin_persen": [round(wl * 100, 2), round(wh * 100, 2)],
                "eksak95_poin_persen": [round(el * 100, 2), round(eh * 100, 2)],
                "dilaporkan95_poin_persen": [round(lo * 100, 2), round(hi * 100, 2)],
                "batas_gain_poin_persen": round(max(hi, 0.0) * 100, 2),
                "batas_rugi_poin_persen": round(abs(min(lo, 0.0)) * 100, 2),
                "max_abs_poin_persen": round(max(abs(lo), abs(hi)) * 100, 2),
                "sisi_max_abs": "gain" if abs(hi) >= abs(lo) else "rugi",
            }

    def summarize(t):
        b = bound[t]
        gain = max(b.values(), key=lambda v: v["batas_gain_poin_persen"])
        loss = max(b.values(), key=lambda v: v["batas_rugi_poin_persen"])
        return {
            "gain_terbesar_poin_persen": gain["batas_gain_poin_persen"],
            "model_gain_terbesar": next(m for m, v in b.items() if v is gain),
            "rugi_terbesar_poin_persen": loss["batas_rugi_poin_persen"],
            "model_rugi_terbesar": next(m for m, v in b.items() if v is loss),
            "gain_per_model": {m: v["batas_gain_poin_persen"] for m, v in b.items()},
            "rugi_per_model": {m: v["batas_rugi_poin_persen"] for m, v in b.items()},
            "max_abs_per_model_seperti_naskah": {m: v["max_abs_poin_persen"] for m, v in b.items()},
        }

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": "0a7b2e4de856",
        "_apa_ini": ("Every channel McNemar test with n10, n01, n, the raw p, and Holm in two families, plus "
                     "the slot bound split into its gain side and its loss side."),
        "definisi": {
            "n10": "rows the first condition wins and the second loses",
            "n01": "rows the first condition loses and the second wins",
            "menang": "the asker's gold ranks above every gold of the other profiles in its group",
            "keluarga_enam_model": "one contrast across six models, m = 6",
            "keluarga_seluruh_tabel": "every row of this table at once, the strictest reading",
            "selang": ("Wald and conditional exact Clopper-Pearson, the wider on each side; the functions are "
                       "imported from equivalence_bounds.py"),
            "batas_gain": "upper end of the interval, cut at zero",
            "batas_rugi": "absolute value of the lower end of the interval, cut at zero",
            "max_abs": "the number the paper calls 'at most', the wider of the two sides",
        },
        "gerbang": "all sixty-six cells reproduce the paper's counts before anything is computed",
        "zerank2": "official vendor path, as in the paper",
        "batas_slot": bound,
        "ringkasan_batas": {t: summarize(t) for t in bound},
        "uji": rows,
        "label_source": "deterministic (gold matching on frozen scores, no rater)",
    }
    OUT.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    MANIFEST.write_text(json.dumps({
        "run_id": "channel_tests_full_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/4_slot_vs_baseline/channel_tests_full.py",
        "output": {"path": "results/tables/channel_tests_full.json", "sha256": K.sha(OUT)},
        "label_source": "deterministic",
    }, indent=1))

    for t in bound:
        print(f"\n{t}: slot bound per model (percentage points)")
        print(f"  {'model':<22}{'point':>7}{'gain':>7}{'harm':>7}{'max|CI|':>9}  side")
        for m, v in bound[t].items():
            print(f"  {m:<22}{v['titik_poin_persen']:>7}{v['batas_gain_poin_persen']:>7}"
                  f"{v['batas_rugi_poin_persen']:>7}{v['max_abs_poin_persen']:>9}  {v['sisi_max_abs']}")
        r = output["ringkasan_batas"][t]
        print(f"  largest gain limit {r['gain_terbesar_poin_persen']} ({r['model_gain_terbesar']}), "
              f"largest harm limit {r['rugi_terbesar_poin_persen']} ({r['model_rugi_terbesar']})")
    print(f"\n{len(rows)} tests written to {OUT}")


if __name__ == "__main__":
    main()
