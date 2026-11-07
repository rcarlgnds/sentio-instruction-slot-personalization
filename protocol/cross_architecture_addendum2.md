> Translation of the original Indonesian protocol, which is kept alongside this file. The original is the dated record, and this file is a reading aid. Table names in this protocol are those of the run. In this package the tables have English names, and their manifests record the run names.

# Cross-architecture pre-registration, addendum 2: packages G, E, F, and carrying out the equivalence consequence

Written 17 September 2026, 20:00 WIB, *before* a single cell below was run. The
state at that moment: the FollowIR-7B paired cells five of five finished, ctxl
two of six, gemma none, no Core17 yet for any of the three new models, no G, E
or F cells at all, and the official zerank-2 path existing for only two
conditions (default, profile) from the full equivalence test. The decision to
run all of it, including the equivalence consequence in full, was taken by the
first author on the evening of 17 September 2026. Sections 1 to 9 of the main
pre-registration and the equivalence addendum are unchanged.

## G. Single-field label cells for the three new lineages

The `both_program` and `both_semester` cells in `score_architectures.py`: the profile
in the instruction slot (identical to the `both` cell) and a document label
carrying only one field, rendered by the same `label_sumber(m, field)` as the
frozen `label_satu_field` cells:

    program  : [Source: <DEPT> degree program textbook]
    semester : [Source: degree program textbook, semester <N>]

Three models: bge-reranker-v2-gemma, ctxl-rerank-v2-instruct-multilingual-6b,
FollowIR-7B. Every other parameter is identical to the paired cells (bf16,
max-len 1536, a 500 MiB gap, automatic batch). Six cells, an estimated 35
minutes of GPU.

Reported per model: the win count out of 96 for both cells, set beside `none`,
`both`, and the lock of 48. Three exact McNemar contrasts per model: program
against both, semester against both, program against semester. Family G is
those nine contrasts (three models times three), Holm-corrected at m = 9.

Reading rule, fixed now: the frozen pattern is that semester-only comes close
to the full label (78/84/93 against 82/84/95) and program-only falls well below
it while staying above the lock (62/74/76). Replication means semester against
both is not significant and program against both is significant in the same
direction. Any other pattern is reported as it stands, as a difference between
lineages. No cell is discarded.

## E. The 1,621-row broad track for the three new lineages

Pool `pools_paired_expanded.jsonl` (sha256 65ac6f2fcdc6e662, 1,621 rows,
110,762 documents, 8 programs), tag `xpair`, four cells mirroring the frozen
broad-track cells: `none`, `slot`, `both`, `wrong` (frozen equivalents:
default, profile, profile_docmeta, profile_docmetasalah). Three models, twelve
cells, an estimated 16 hours of GPU (FollowIR 8.7 hours at 14 documents per
second, ctxl 5.4 hours, gemma 1.9 hours). There is no text-only cell, as on the
frozen track.

Reported per model: the win count out of 1,621 against the structural blind
lock of 600/1621; an exact McNemar of none against slot per model, family E2 at
m = 3 under Holm; the slot-effect equivalence bound under the same `analisis()`
function (the wider of Wald and exact); and the text-channel effect (both minus
none) per model *within* this track. Ratios between tracks are computed inside a
track, never mixing denominators.

Reading rule: replication means the slot sits within a few rows of the lock of
600 with an upper bound under 1 percentage point, and both sits far above it.
If any model moves through the slot on this track, that is reported as a
finding rather than hidden.

## F. News21 and Robust04 for the three new lineages

`followir_architectures.py --dataset News21` and `--dataset Robust04`, 1,000
candidates per query, p-MRR and nDCG@5 from the MTEB implementation already in
use, three models, six runs, an estimated 6 hours of GPU (News21 64 queries,
Robust04 104 queries). The positive-control criterion stays the one in the main
pre-registration for Core17. News21 and Robust04 are reported as additions in
the same table under no new criterion, so that the FollowIR table holds three
datasets for six models, except zerank-2 whose frozen data covers only Core17
(and which is completed below).

## (a) Carrying out the consequence of section 3 of the equivalence addendum

The pre-registered consequence is to rescore *every* zerank-2 cell on the
vendor's official path. How: a new scorer family `zerank2official` in
`score_arms.py`, additive, leaving the old family's behavior unchanged,
calling `sentence_transformers.CrossEncoder(...).predict()` (version 6.0.1 in
the `system/.st-libs` overlay, bf16, max_length 1536, batch 2, identical to the
full equivalence test). The whole input construction (instruction, slot format,
document label, wrong label, vendor memory, the 4,000-character cut) stays with
the same script, so each cell is repeated under exactly the command in its
frozen manifest with only the family name changed.

The cells repeated, every one that has a frozen manifest: control_default,
control_kondisi, control_profile (116 rows); holdout_default, holdout_profile
(283 rows, seal: the holdout pool was already opened for these frozen cells,
this is a reread of the same pool, zero tuning); paired default,
default_docmeta, profile, profile_docmeta, profile_docmeta_fieldprogram,
profile_docmeta_fieldsemester, profile_docmetasalah, profile_fmtlabeled,
profile_fmttag, profile_vmem (104 rows); traindev_default, traindev_profile
(2,493 rows); xpair default, profile, profile_docmeta, profile_docmetasalah
(1,621 rows); Core17; and the seven-strategy injection probe. Around 690,000
documents in all, an estimated 7 hours of GPU at 26 documents per second.

Reported for each cell: the frozen count, the official-path count, per-row
agreement, and the number of rows tied within precision (|gold score minus
counterpart score| <= 0.25). Following section 3 of the equivalence addendum,
the write-up's numbers move to the official-path values and *both* numbers are
reported in Appendix B. Expected in advance, and written here so it cannot read
as a defense invented after the numbers arrived: a cell with 6 to 8 tied rows
is expected to shift by zero to two rows, and a shift that size does *not* change
any conclusion, because every test uses paired discordant counts rather than
raw totals.

## Execution order and general rules

Order by value to the write-up: G, H (its own pre-registration in
`study/retrieval/PRAREGISTRASI_AUTO_KURASI.md`), (a), I (its own
pre-registration in `study/retrieval/PRAREGISTRASI_HOLDOUT_GENERASI.md`),
E, F. One GPU job at a time through `raw/.gpu.lock`, wait for an empty card
before loading, a 500 MiB gap (the owner's decision for this run). Every cell
has its own manifest. A finished cell is not repeated. A failed cell is retried
once and, if it fails again, is reported as failed. No cell is discarded
because of its result.
