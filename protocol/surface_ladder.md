> Translation of the original Indonesian protocol, which is kept alongside this file. The original is the dated record, and this file is a reading aid. Table names in this protocol are those of the run. In this package the tables have English names, and their manifests record the run names.

# Pre-registration addendum 5: separating surface form from crossing a language

Written and committed *before* a single cell was run. The interpretation rules
below bind, predictions included.

Date: 20 September 2026
Parent: marker_ladder.id.md and KEPATUHAN_PRAREGISTRASI_TANGGA.md
Config lock: 0a7b2e4de856

## 1. The defect this repairs

The first ladder replaced the program code in the candidate text with the
Indonesian program name while the profile instruction stayed in English and
kept the code. That rung therefore changed *three* things at once: code to name,
English to Indonesian, and the model was now required to know that CSCI is an
institutional abbreviation for computer science. Its collapse cannot be
attributed to any one of the three.

The write-up already states that limitation, but the practical rule in the
conclusion was still written as though it were general. This addendum measures
which of the three is right.

## 2. English program names, and why the choice is not free

The NAMA_PRODI mapping in the script comes from production code, and that is
what protected the first ladder from the charge of picking favorable wording.
The repository holds *no* code-to-English-name mapping, so the mapping below was
*written* for this experiment, and that has to be acknowledged.

What limits the freedom: every code really is an abbreviation of its English
name, so the name is not chosen but read off the code itself.

| code | English name | the code is a prefix of the name |
| --- | --- | --- |
| ACCT | Accounting | yes |
| BANA | Business Analytics | no |
| CHIN | Chinese Literature | yes |
| CSCI | Computer Science | no |
| DTSC | Data Science | no |
| ISYS | Information Systems | no |
| MGMT | Management | no |
| MKTG | Marketing | no |
| STAT | Statistics | yes |

The paired track uses CSCI and STAT, so once again one program shares a
prefix and one does not. The same stratum split as tangga_substring.json *must*
be run again and its result reported.

## 3. The new rungs

### 3a. The program field, with the instruction using the English *name*

The profile instruction changes exactly once, from `a STAT student in semester
2` to `a Statistics student in semester 2`. The label in the candidate text is
then varied, and *only* the label changes between cells.

| cell | injected label | what it tests |
| --- | --- | --- |
| p1 | `[Source: Statistics degree program textbook]` | surface identical to the instruction |
| p2 | `[Source: statistika degree program textbook]` | same fact, different language |
| p3 | `[Source: STAT degree program textbook]` | the instruction names, the document codes |

The frozen anchor: code instruction plus code label, that is rung a of the
first ladder, 62 / 74 / 76 / 52 / 83 / 49 against the blind lock of 48.

### 3b. The semester field, with the instruction *unchanged*

The instruction stays `semester 2`. The label frame is held constant and only
the value changes, so that exactly one thing varies.

| cell | injected label | what it tests |
| --- | --- | --- |
| s1 | `[Source: degree program textbook, semester 2]` | *Already measured*, the semester-only control, 78 / 84 / 93 / 91 / 96 / 51 |
| s2 | `[Source: degree program textbook, semester two]` | the number as an English word |
| s3 | `[Source: degree program textbook, semester T-B]` | an opaque term code, the *negative control* |

Language is *not* varied on the semester field, because the word semester is the
same in Indonesian and in English, so no language contrast can be formed. The
opaque code uses letters rather than digits so that the 2 in the instruction
cannot leak into the label. The map is fixed, the nth letter of the alphabet
for semester n, and is recorded in the script.

## 4. Predictions, written before running

1. If the document channel works through surface form, p1 is high and p2 falls
   to around the lock. The size is not predicted.
2. p3 is lower than p1 on most models, because the instruction names while the
   document codes.
3. s2 falls below s1 on most models if the semester effect is number matching.
   If s2 holds, the semester effect carries meaning.
4. s3 lands on the lock on all six models.
5. Bge-reranker-v2-gemma stays near the lock in every cell, as in every cell
   before this.

## 5. Interpretation rules, binding

- The deciding contrast is p1 against p2. Both use an English instruction that
  names the program, and only the label's language differs. Exact paired
  McNemar, Holm within the family of six models.
- If p1 is high and p2 sits on the lock, the write-up *must* state that the
  document channel demands the same surface form, and the collapse of rung b of
  the first ladder is attributed to surface form rather than to crossing a
  language.
- If p1 and p2 are both high, that collapse was a cross-language failure. The
  write-up *must* withdraw the phrase "spelled the way the instruction spells it"
  and state language as the cause.
- If p1 also sits on the lock, then naming the program in the instruction is
  by itself enough to cut the channel, and every channel result applies
  specifically to the code form. Reported as it stands, including when it
  weakens the write-up.
- s2 against s1 decides whether the abstract's "shared string" claim may be
  extended to the semester field. If s2 holds, the claim is permanently
  restricted to the program field and the limiting sentence in the conclusion
  stays.
- No frozen cell is overwritten. No old number is changed.
- If any result weakens the write-up it still goes in the body, not into an
  appendix.

## 6. Artifacts

Cell script: study/channel/scripts/score_architectures.py, cells `label_en`,
`label_id_en`, `label_kode_en`, `sem_kata`, `sem_opak`.
Aggregator: study/channel/scripts/inference_ladder_surface.py.
Table: study/channel/results/tables/tangga_permukaan.json.
Stratum split: study/channel/scripts/inference_ladder_substring.py, extended.

## Operational note, added while the queue was running

*Do not* rebuild paper1_code or run quick_check while the GPU queue is still
writing score files. The validity audit gate reads every score file, and a file
being written reads as a cell with a smaller n_komparabel, so the uniformity
check on n fails. That is not a bug, that is the gate working. Real occurrence
on 20 September: the anomaly {'paired': [50, 96]} came from the gemma sem_opak
cell, which had filled 50 of 96 rows so far.

The correct order: wait for the queue to finish, run the aggregator *without*
--izinkan-hilang, and only then build the package and run its gates.

## Two queue incidents, recorded so they are not repeated

### 1. A batch size forced on a model with no room for it

Four FollowIR-7B cells died of OOM because `--batch 4` was set explicitly. That
model's weights take 13,813 MiB on a 16 GB card, leaving only 718 MiB, and the
`auto` probe would in fact have picked batch 2. The first ladder *never* set an
explicit batch for FollowIR, and that is exactly why it passed.

The lesson had been applied backwards. The lesson "set the batch explicitly"
came from a case where the probe picked too *large* a batch because it only
looked at the first query's pool, and that applies to ctxl. For FollowIR the
probe was already choosing correctly, and forcing it higher killed it. The
correct rule: set an explicit batch only where the probe has been shown to pick
too large a value for that model, not for every large model.

### 2. The wrong track's pool, which nearly produced fifteen invalid cells

Fifteen Qwen3 steps pointed at `results/raw/pools_paired_expanded.jsonl`. That
filename *misleads*. It holds 1,621 rows prefixed `xpair-`, that is the *broad*
*track* pool, not the paired one. The correct paired-track pool lives at
`study/retrieval/results/raw/pools_paired.jsonl`, 104 rows, all prefixed
`paired-`, sha256
ff0b146e3c75ed9f05a0080f52c28353c9ff353b0b1837413d1b87b602106483, and that sha
matches the one every first-ladder cell used.

Had they run, those fifteen cells would have produced 1,621-row numbers *not*
comparable with the 96-row anchor, and the table would have looked valid. This
is the same class of defect as the cwd and env incidents, an invalid
measurement that looks valid.

What saved it was luck: the pool path was written relative while the steps' cwd
was system, so all fifteen died with FileNotFoundError before touching the GPU.
My own smoke test did *not* catch it, because it verified the shape of the output
and the row count but not the query_id prefix.

New rule: every new cell must have its output's query_id prefix verified, not
only its row count and file shape. And every queue step must use absolute
paths, because a step's cwd is not the repository root.
