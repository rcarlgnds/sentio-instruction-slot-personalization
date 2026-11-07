"""95% limits on how large the instruction-slot effect could be.

A McNemar test that finds nothing does not say which effects it rules out. This script turns the saved
counts of disagreeing rows into intervals on the paired difference in resolved rows, and keeps the
wider of a Wald and an exact interval on each side. Writes results/tables/equivalence_bounds.json.
"""

import os
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
Q2 = ROOT / "study/channel/results"
SHARED = ROOT / "study/_shared"

CORE_SOURCE = Q2 / "tables/channel_board_paired.json"
BROAD_SOURCE = Q2 / "tables/channel_board_xpair.json"

CORE_PATH = ("mcnemar_keluarga_A_slot_lawan_buta", "kontras")
BROAD_PATH = ("kontras_mcnemar_suplemen", "kontras")

CORE_KEY = {
    "Qwen3-Reranker-0.6B": "C3 tanpa_profil vs C3 slot",
    "Qwen3-Reranker-4B": "C4 tanpa_profil vs C4 slot",
    "zerank-2": "zerank2 tanpa_profil vs zerank2 slot",
}
BROAD_KEY = {
    "Qwen3-Reranker-0.6B": "Qwen3-Reranker-0.6B tanpa_profil vs Qwen3-Reranker-0.6B slot",
    "Qwen3-Reranker-4B": "Qwen3-Reranker-4B tanpa_profil vs Qwen3-Reranker-4B slot",
    "zerank-2": "zerank-2 tanpa_profil vs zerank-2 slot",
}

CORE_TEXT_KEY = {
    "Qwen3-Reranker-0.6B": "C3_profil vs C3_docmeta_profil",
    "Qwen3-Reranker-4B": "C4_profil vs C4_docmeta_profil",
}
TEXT_SOURCE = Q2 / "tables/mcnemar_paired.json"

BROAD_TEXT_KEY = {
    "Qwen3-Reranker-0.6B": "Qwen3-Reranker-0.6B slot vs Qwen3-Reranker-0.6B slot_plus_teks",
    "Qwen3-Reranker-4B": "Qwen3-Reranker-4B slot vs Qwen3-Reranker-4B slot_plus_teks",
    "zerank-2": "zerank-2 slot vs zerank-2 slot_plus_teks",
}

Z95 = 1.959963984540054


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def betainv_lower(k, n, alpha):
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        s = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k, n + 1))
        if s < alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def betainv_upper(k, n, alpha):
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        s = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(0, k + 1))
        if s > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def wald_interval(b, c, n):
    d = (b - c) / n
    var = (b + c - (b - c) ** 2 / n) / n ** 2
    se = math.sqrt(max(var, 0.0))
    return d, d - Z95 * se, d + Z95 * se


def exact_interval(b, c, n):
    m = b + c
    d = (b - c) / n
    if m == 0:
        return d, -0.0, 0.0
    lo_pi = betainv_lower(b, m, 0.025)
    hi_pi = betainv_upper(b, m, 0.025)
    return d, (2 * lo_pi - 1) * m / n, (2 * hi_pi - 1) * m / n


def analysis(name, b, c, n):
    d, wl, wh = wald_interval(b, c, n)
    _, el, eh = exact_interval(b, c, n)
    lo, hi = min(wl, el), max(wh, eh)
    bound = max(abs(lo), abs(hi))
    return {
        "model": name,
        "n_berpasangan": n,
        "diskordan_slot_menang": b,
        "diskordan_buta_menang": c,
        "selisih_bersih": b - c,
        "selisih_proporsi": round(d, 6),
        "wald95": [round(wl, 6), round(wh, 6)],
        "eksak_bersyarat95": [round(el, 6), round(eh, 6)],
        "dilaporkan95": [round(lo, 6), round(hi, 6)],
        "metode_dilaporkan": "the wider interval on each side, of Wald and exact",
        "batas_atas_besaran_efek": round(bound, 6),
        "batas_atas_poin_persen": round(bound * 100, 2),
    }


def main():
    core_src = json.loads(CORE_SOURCE.read_text())
    broad_src = json.loads(BROAD_SOURCE.read_text())
    text_src = json.loads(TEXT_SOURCE.read_text())

    core_k = core_src[CORE_PATH[0]][CORE_PATH[1]]
    broad_k = broad_src[BROAD_PATH[0]][BROAD_PATH[1]]
    text_k = text_src["kontras_mcnemar"]

    core_result, broad_result = [], []
    for name, key in CORE_KEY.items():
        v = core_k[key]
        core_result.append(analysis(name, v["diskordan_a_kalah_b_menang"],
                                   v["diskordan_a_menang_b_kalah"], v["n_berpasangan"]))
    for name, key in BROAD_KEY.items():
        v = broad_k[key]
        broad_result.append(analysis(name, v["diskordan_a_kalah_b_menang"],
                                   v["diskordan_a_menang_b_kalah"], v["n_berpasangan"]))

    text = []
    for name, key in CORE_TEXT_KEY.items():
        v = text_k[key]
        text.append(analysis(name + " (slot -> slot+teks)",
                             v["diskordan_a_kalah_b_menang"],
                             v["diskordan_a_menang_b_kalah"], v["n_berpasangan"]))

    broad_text = []
    for name, key in BROAD_TEXT_KEY.items():
        v = broad_k[key]
        broad_text.append(analysis(name + " (slot -> slot+teks)",
                                  v["diskordan_a_kalah_b_menang"],
                                  v["diskordan_a_menang_b_kalah"], v["n_berpasangan"]))

    core_bound = max(h["batas_atas_besaran_efek"] for h in core_result)
    broad_bound = max(h["batas_atas_besaran_efek"] for h in broad_result)
    text_effect_min = min(abs(t["selisih_proporsi"]) for t in text)
    broad_text_effect_min = min(abs(t["selisih_proporsi"]) for t in broad_text)
    broad_text_effect_max = max(abs(t["selisih_proporsi"]) for t in broad_text)
    within_track_ratio = []
    for hb, tl in zip(broad_result, broad_text):
        within_track_ratio.append(round(abs(tl["selisih_proporsi"]) / hb["batas_atas_besaran_efek"], 1))

    print("INSTRUCTION SLOT EQUIVALENCE BOUNDS, computed from frozen discordant counts\n")
    print(f"{'track':10s} {'model':22s} {'b/c':>8s} {'diff':>9s} {'95% interval':>24s} {'bound':>8s}")
    for label, block in (("core n=96", core_result), ("broad n=1621", broad_result)):
        for h in block:
            lo, hi = h["dilaporkan95"]
            print(f"{label:10s} {h['model']:22s} {h['diskordan_slot_menang']}/{h['diskordan_buta_menang']:<5d} "
                  f"{h['selisih_proporsi']:+9.4f} [{lo:+.4f}, {hi:+.4f}]  {h['batas_atas_poin_persen']:6.2f} pp")

    print("\nTRACK CORE n=96")
    print(f"  worst-case slot bound    : {core_bound*100:.2f} percentage points")
    print(f"  smallest text effect     : {text_effect_min*100:.2f} percentage points")
    print("TRACK BROAD n=1621")
    print(f"  worst-case slot bound    : {broad_bound*100:.2f} percentage points")
    print(f"  text effect range        : {broad_text_effect_min*100:.2f} to "
          f"{broad_text_effect_max*100:.2f} percentage points")
    print(f"  ratio WITHIN track       : {min(within_track_ratio)}x up to "
          f"{max(within_track_ratio)}x")
    print("\nNote: the ratio is computed within one track. Dividing the broad track bound")
    print("by the core track text effect mixes two denominators and is not valid.")

    lock = json.loads((SHARED / "config_lock.json").read_text())
    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "_apa_ini": ("Upper bound on the effect size of the instruction-slot channel. A failure to reject is "
                     "not evidence of no effect, so this gives a bound instead of another p-value."),
        "metode": {
            "wald": "Var(d) = [b + c - (b-c)^2/n] / n^2, normal 95 percent interval",
            "eksak_bersyarat": ("b ~ Binomial(b+c, pi), Clopper-Pearson on pi, mapped to d = (2*pi - 1)*(b+c)/n"),
            "dilaporkan": "the wider side of the two methods, so it is never more optimistic",
            "margin_ekuivalensi": ("not chosen. Choosing a margin after seeing the data is the error the pre-registration "
                                   "of this project is there to prevent, so the interval itself is reported."),
        },
        "track_inti_n96": core_result,
        "track_luas_n1621": broad_result,
        "pembanding_saluran_teks_inti_n96": text,
        "pembanding_saluran_teks_luas_n1621": broad_text,
        "ringkas": {
            "_peringatan": ("the ratio must be computed within one track. The broad-track slot bound against the "
                            "core-track text effect mixes a denominator of 1621 with one of 96 and is not valid."),
            "inti_n96_batas_slot_pp": round(core_bound * 100, 2),
            "inti_n96_efek_teks_terkecil_pp": round(text_effect_min * 100, 2),
            "luas_n1621_batas_slot_pp": round(broad_bound * 100, 2),
            "luas_n1621_efek_teks_pp": [round(broad_text_effect_min * 100, 2),
                                        round(broad_text_effect_max * 100, 2)],
            "rasio_dalam_track_luas": within_track_ratio,
        },
        "label_source": "deterministic (arithmetic on the frozen discordant counts)",
    }
    out = Q2 / "tables/equivalence_bounds.json"
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    (Q2 / "raw/equivalence_bounds.manifest.json").write_text(json.dumps({
        "run_id": "q2_ekuivalensi_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/4_slot_vs_baseline/equivalence_bounds.py",
        "config_version": lock["config_version"],
        "label_source": "deterministic",
        "sumber": {
            "channel_board_paired.json": sha(CORE_SOURCE),
            "channel_board_xpair.json": sha(BROAD_SOURCE),
            "mcnemar_paired.json": sha(TEXT_SOURCE),
        },
        "output": {"path": "results/tables/equivalence_bounds.json", "sha256": sha(out),
                   "tingkat_reproduksi": "1_bit_identical"},
    }, indent=2, ensure_ascii=False))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
