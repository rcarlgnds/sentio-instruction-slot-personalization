# Protocol: switching the label map

English translation of `label_map.id.md` (original title "Praregistrasi pergantian peta label").
Script names are those of the authors' tree. `q2_peta_label.py` ships as
`study/channel/scripts/0_common/label_map.py` and `q2_gerbang_peta_label.py` as
`study/channel/scripts/1_tracks/label_map_gate.py`.

Written 2 October 2026, *before* any cell was rerun. It holds the predictions, the analysis rules,
and the list of things that may not change. Its purpose is that the results of the rerun cannot
be picked after the numbers are seen.

## 1. The defect being fixed

The reranker cells that insert a document label read the curriculum metadata through
`muat_peta_metadata()`. That function maps the first 180 characters of a candidate's text to the
corpus metadata and uses `setdefault`, so the first matching book wins. Two OpenStax books
overlap, Introductory Statistics 2e of the STAT program and Introductory Business Statistics of
MGMT, with 322 identical chunk openings and 110 identical chunks. As a result, some STAT passages
received an MGMT label.

Measured sizes, not estimates:

| what was checked | paired track | broad track |
|---|---|---|
| gold with a wrong label | 21 of 96 | 30 of 1,621 |
| non-gold candidates with a wrong label | 370 of 8,138 | 1,314 of 110,762 |
| rows with at least one wrongly labeled candidate | 94 of 104 | 437 of 1,621 |
| wrong semester | 0 | 0 |

The old map also lost 1,140 of 64,108 chunks because their prefixes collided, and
`n_kandidat_tak_terpetakan` in the old manifests still reads 0 because every prefix was indeed
found, only in the wrong book. The old gate could not see this defect.

## 2. What changes

`q2_peta_label.py` maps a candidate's identity to the corpus metadata, through `_id` in
`corpus.jsonl` plus the aliases in `uuid_to_canonical.json`. Identities cannot collide, so this
class of defect disappears rather than shrinks.

Seven scripts used the old map, and each gets a switch:

| script | switch | default |
|---|---|---|
| `q2_arm_score.py` | `--peta-label {id,teks}` | `id` |
| `q2_arch_score.py` | `--peta-label {id,teks}` | `id` |
| `q2_probe_injeksi.py` | `--peta-label {id,teks}` | `id` |
| `q2_strata_label_korup.py` | `--peta {teks,id}` | `teks` |
| `q2_strata_label_korup_arsitektur.py` | `--peta {teks,id}` | `teks` |
| `q2_label_salah_prodi.py` | `--peta {teks,id}` | `teks` |
| `q2_semester_gold_paired.py` | `--peta {teks,id}` | `teks` |

The four diagnostic scripts keep `teks` as their default because their task is to document the
defect the manuscript reports. Their `id` mode writes to files with the suffix `_petaid`, so a
frozen table cannot be overwritten.

## 3. What does *not* change, and this is binding

1. The choice of the wrong label in `dept_salah()` keeps using a sha256 over 180 characters of
   text. That is a deterministic salt, not a metadata lookup, so replacing it would shift the
   wrong-label cells for no reason.
2. The list `semua_dept` that drives that choice was verified identical under both maps: eight
   programs and five semesters, in the same order.
3. Gold, counterpart, candidate pools, DITS weights, model configuration, dtype, batch, maximum
   length, and instruction text are not touched.
4. The `test_holdout` seal stays closed.
5. Old result files are neither deleted nor overwritten. New results go to `results/raw_petaid/`.

## 4. Predictions, written before the numbers exist

On the 21 defective paired rows, the gold was given the label of a program that is not the
asker's. A reranker that reads the label will push the gold down in those rows. With the correct
label the gold should rise. Therefore:

1. The paired-track win count of the both-channel cell *rises*, by at most 21 rows. The old
   numbers are 82 to 96 of 96 for five models and 51 for bge-gemma.
2. The broad track *rises* by at most 30 rows from the old numbers 696, 639, 810, 837, 913, and 600.
3. Smaller movement in both directions may come from the 370 and 1,314 non-gold candidates whose
   label changes, because such a candidate can rise past a gold or fall below it.
4. The slot-only cells are not rerun, because the map is never read there, so the slot finding
   does not move at all.
5. The wrong-label cells move in a direction that is *not* predicted, because the wrong label is
   chosen relative to the correct label, and the correct label changes in the defective rows.

If the rise exceeds 21 on the paired track or 30 on the broad track, or if a number *falls*, then
something other than the label map has changed, and that cell must be investigated before it is
used.

## 5. Analysis rules

1. Stage r0 runs six representative cells with the *old* map into a new folder. It passes when zero
   rows flip and score differences stay below the dtype tolerance, 1e-6 for float16 and 0.30 for
   bfloat16, following the recorded zerank-2 noise floor. If r0 fails, the rerun does not proceed,
   because the new pipeline itself would then have shifted the numbers.
2. Cells in `id` mode must pass eleven checks, and the most binding is attribution: a row whose
   outcome flips must be a row whose label actually changed. The limit for flipped rows outside
   those is 0 for float16 and 8 for bfloat16.
3. Old and new numbers are reported side by side in the manuscript. Old numbers are not deleted
   from the repository.
4. If the direction of a finding changes, that is, if the slot channel comes alive or the text
   channel dies, it is reported as it is and the manuscript follows the data. No cell is dropped
   because its result is unwelcome.
5. One failing cell holds the whole queue. No cell continues on top of output that has not passed
   validation.

## 6. The gate that must pass before the first cell runs

`q2_gerbang_peta_label.py`, nine checks, no GPU. Results of 2 October 2026:

| gate | result |
|---|---|
| G1 identity coverage | 8,138 of 8,138 and 110,762 of 110,762 |
| G2 conflicts on unique prefixes | 0 and 0 |
| G3 reproduction of the defect size | 21 and 30, matching the manuscript |
| G4 holdout seal | no holdout query_id |
| G6 library overlay | transformers 4.55.0, sentence-transformers 6.0.1 |
| G7 map determinism | the same fingerprint twice |

G2 is the most convincing gate. On candidates whose text prefix is unique in the corpus, the old
and the new map *never* differ. The new map is therefore not a different map on its own: it is the
same map at every unambiguous place, and it differs only where the old one was ambiguous.

## 7. Budget

101 of 165 cells touch a document label, and only those are rerun. By the durations of the old
runs they total 1,333 GPU minutes, 22.2 hours: 83 paired-track cells at 423 minutes and 18
broad-track cells at 911 minutes. The highest peak VRAM among them is 14,330 MiB, so every cell
passes a 1,800 MiB free-VRAM check before it runs.

---

Notes added for this release (not part of the protocol): 97 cells were rescored and ship. Two
both-channel paired counts fell rather than rose, Qwen3-4B by one row and bge-gemma by two. In
the 73 rescored cells that have a pre-fix counterpart, every row whose outcome changed is one in
which the asker's or the rival's gold changed label. The paper prints only the new counts, so
rule 3 of section 5 (old and new side by side in the manuscript) was not followed. The old cells
stayed in the authors' archive.
