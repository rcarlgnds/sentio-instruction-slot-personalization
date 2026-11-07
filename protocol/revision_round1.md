# Pre-registration: revision analyses for IJIES round 1 (Paper No. 20266617)

Written 12 October 2026, before any new score, before the broad-track resample is
drawn, and before any analysis of new data. This file is committed to the
project repository before the first GPU cell runs; its sha256 is recorded in the
manifest of the resample and of every new cell. Any departure from it is logged
in section 11 with its reason, and nothing here is changed after results exist.

Submitted manuscript: build 33o (PDF sha256 1516c1ba...). Frozen inputs: the
canonical score files in `result/paper_q2/results/raw/` after the label-map
re-run of 2-3 October 2026, and the pools `pools_paired.jsonl` (paired track)
and `pools_paired_expanded.jsonl` (broad track).

## 1. What each reviewer item needs

| Item | Request | Answered by |
| --- | --- | --- |
| R1-1, R2-1 | Independent validation of gold correctness and answerability, agreement or error rate, sensitivity of the resolved counts, provenance of the 115 paired questions | Section 9 (provenance, validation kit, sensitivity) |
| R1-2, R2-2 | Broad sample not tied to gate version 1 and file order | Section 6 (new stratified random sample) and section 7 |
| R1-3 | Program and semester change together in the paired slot test | Section 4 (single-field and fixed-field slot cells) and section 7.3 |
| R1-4, R2-4 | Group-level inference for the broad track with multiplicity control | Section 7 |
| R1-5 | bge-gemma not tested at the vendor's documented prompt placement | Section 3 |
| R1-6 | 36-pair textual subset was defined after scoring | No new analysis; the subset is labelled exploratory in the text |
| R2-5 | Factorial label controls with matched negative controls | Section 5 |
| R1-8, editor | Public code and data identifiers | Section 10 |

## 2. Environment gate (G0)

The NVIDIA driver changed from 580.173.02 (October re-run) to 580.178.04 after
the reboot of 12 October; torch 2.11.0+cu128 and transformers 4.55.0 are
unchanged. The scorers received additive patches for the new cells (CPU check:
`q2_r1_uji_masukan.py`, 331 of 331 checks, every frozen cell type produces
byte-identical inputs before and after the patch).

Before any new cell, these existing cells are re-run with the patched scripts
into `results/raw_r1/_g0/` and compared with their canonical files:

- Qwen3-Reranker-0.6B and 4B, paired, profile with label (q2_arm_score.py)
- zerank-2 official path, paired, profile with label
- FollowIR-7B, ctxl-6B, bge-gemma, paired `both`; bge-gemma paired `slot_vendor`
- Qwen3-Reranker-0.6B and 4B, paired, default and profile without label, re-run
  with q2_arm_score.py and compared with `qwen_scores_{0.6b,4b}_paired_{default,profile}.jsonl`
  from t2_qwen_score.py (these two cells become the blind and slot references
  of the new Qwen3 paired cells)

Pass rule per cell: identical query set, zero rows whose win or loss flips, and
maximum absolute score difference at most 1e-6 for float16 or 0.30 for
bfloat16 (0.30 is the measured bfloat16 repeatability floor of this project).
A family whose gate cell fails runs no new cell until the failure is explained;
the other families continue. For the two Qwen3 reference cells, a failure only
means the new Qwen3 paired cells are compared with the re-run references, not
the old files.

## 3. R1-5: bge-gemma at the documented prompt placement (G1)

Vendor documentation (model card at revision 1787044f8b6f, Transformers
section): `get_inputs` builds `<bos>A: {query}\nB: {passage}\n{prompt}` and the
score is the logit of "Yes" at the last position. The default prompt ends with
"by providing a prediction of either 'Yes' or 'No'." In our `slot` construction
the profile sentence was appended after that prompt, so the last token was
profile text. Our existing `slot_vendor` cell replaces the prompt with a task
description that contains the profile and still ends with the documented
sentence, which is the documented placement.

New cells, bge-gemma only, explicit batch 8, gap 300 MiB:

| Cell | Paired | Broad (original 1,621 rows) |
| --- | --- | --- |
| `slot_vendor` | exists | new |
| `both_vendor` (vendor placement plus correct label) | new | new |
| `wrong_vendor` (vendor placement plus wrong-program label) | new | new |

Analysis (family = bge-gemma, Holm over the tests listed per track):
`slot_vendor` vs `none` (exact McNemar and the bound of section 7.1),
`both_vendor` vs `slot_vendor`, `wrong_vendor` vs `both_vendor`.

Reading rule. If `both_vendor` resolves significantly more rows than
`slot_vendor` on the paired track, the text says bge-gemma reads the label once
the prompt follows its documentation, and the label sensitivity check covers
six models. Otherwise the text keeps "five of six" and states that the label is
not read at either placement. The slot conclusion for bge-gemma is stated from
`slot_vendor` only.

Optional, last in the queue (G1b, about 2 GPU hours): the FollowIR positive
control for bge-gemma at the documented placement, with the prompt
"Relevance requirement: {instruction}\nGiven a query A and a passage B,
determine whether the passage contains an answer to the query and meets the
relevance requirement by providing a prediction of either 'Yes' or 'No'."
Reported beside the existing appended-prompt values, never in place of them.

## 4. R1-3: orthogonal slot design (G2), paired track, six models

Profiles on the paired track are CSCI semester 1 and STAT semester 2. New
instruction cells, everything else identical to each model's existing slot
cell (same pool, precision, batch, token budget):

| Cell | Instruction | What varies between the two rows of a pair |
| --- | --- | --- |
| program only | "Given a query from a {dept} student, retrieve passages from that student's own study program that answer the query" | program |
| semester only | "Given a query from a student in semester {sem}, retrieve passages that answer the query at a level appropriate for semester {sem}" | semester |
| semester fixed at 1 | full profile instruction, semester set to 1 for both rows | program |
| semester fixed at 2 | full profile instruction, semester set to 2 for both rows | program |
| program fixed at CSCI | full profile instruction, program set to CSCI for both rows | semester |
| program fixed at STAT | full profile instruction, program set to STAT for both rows | semester |

Analysis per cell and model: resolved rows out of 96, exact McNemar against the
blind reference of that model, the pair-level sign-flip test of section 7.2,
and the bound of section 7.1. Holm over the six models within each cell.

Broad-track complement without new GPU work: the 95 groups of the original
sample whose rows share one semester (218 rows) isolate the program; the slot
and label contrasts are recomputed on that subset (section 7.3).

Reading rule. If no cell differs from the blind reference after Holm, the text
states that neither field, alone or with the other held fixed, moves the slot.
Any significant cell is reported as a finding with its direction.

## 5. R2-5: factorial label design (G3), paired track, six models

The slot carries the full profile in every cell. Existing cells: correct
program and semester (`both`), wrong program with correct semester (`wrong`).
New cells:

| Cell | Program in the label | Semester in the label |
| --- | --- | --- |
| semester wrong | correct | wrong |
| both wrong | wrong | wrong |
| random (matched negative control) | drawn independently of the book | drawn independently of the book |
| uniform (presence control) | the asker's own, on every candidate | the asker's own, on every candidate |

Wrong values follow the frozen wrong-program rule: a deterministic choice among
the other observed values (programs ACCT, BANA, CSCI, DTSC, ISYS, MGMT, MKTG,
STAT; semesters 1 to 5), keyed by sha256 of a fixed salt and the first 180
characters of the passage. The random label uses the same hash construction
over the full value sets, true values included. Exact rules are in
`q2_r1_label.py` and are copied into every manifest. Because the rule excludes
only the passage's own value, some wrong labels equal the other student's
value; the analysis reports how many gold and rival labels coincide that way and
repeats each contrast without those rows.

Contrasts, each with exact McNemar per model and Holm over six models:

1. program correct vs wrong, semester correct: `both` vs `wrong`
2. program correct vs wrong, semester wrong: `semester wrong` vs `both wrong`
3. semester correct vs wrong, program correct: `both` vs `semester wrong`
4. semester correct vs wrong, program wrong: `wrong` vs `both wrong`
5. random label vs slot only
6. uniform label vs slot only

Reading rule. A field is read for its content in a model when correcting it
raises resolution significantly at both levels of the other field (contrasts 1
and 2 for program, 3 and 4 for semester). The controls are clean when contrasts
5 and 6 are not significant. The 2 x 2 table of resolved rows is reported for
every model whatever the tests say.

## 6. R1-2, R2-2: independent broad sample (G4)

Population: the 2,493 training and development queries of both gate versions;
test_holdout is never read. Eligibility, partner rule and pool rule are the
frozen ones (gold in the asker's own program; per other program the most
similar chunk among its top 20 with cosine in [0.66, 0.89] and word Jaccard at
most 0.36; at most two partners; pool of the 71 nearest neighbours of the gold
plus every gold). `q2_r1_tambang_luas.py --mode lama` re-ran the original miner
and reproduced the frozen pool: 1,621 of 1,621 rows, 2,200 eligible queries,
874 read, 600 groups; 1,615 rows byte-identical and 6 differing only in the
order of two golds appended at the end of the pool (the original appended them
in Python set order, which does not affect any ranking because rankings sort by
score and break ties by passage id).

Sampling: partners are computed for every eligible query first. From the
groupable queries, 600 groups are drawn by stratified random sampling without
replacement, strata = gate version x asker program, proportional allocation by
largest remainder, numpy default_rng (PCG64) with seed 20261012. The draw is
made once; the script refuses to overwrite it.

Cells: `none`, `slot` and `both` for all six models, with each model's existing
broad-track settings (zerank-2 on the official path).

Reported: resolved rows against the blind baseline (one row per group); slot vs
none and both vs slot with the analyses of section 7; the same per gate stratum;
the same on the groups whose source query the original miner never read;
DITS and the hard filter on the new pool; overlap with the original sample.

On the original sample, the slot vs none contrast is also reported per tertile
of the source query's gate-1 score (`qc.top_cps`, tertiles over the 600 groups)
and per source program, with the analyses of section 7, as a description of
whether the bound depends on how the original sample was admitted. No test of
heterogeneity is planned; the per-stratum intervals are reported as they come.

Reading rule. The text reports the larger of the original and the new
bound as the broad-track bound. If the new sample shows a significant slot
effect after Holm, the null claim for the broad track is withdrawn.

## 7. Inference (R1-4, R2-4), applied to every broad contrast in the text and to all new contrasts

7.1 Row-level bound, unchanged from the manuscript: paired difference d = (a -
b) / n with the Wald interval of Eq. (4) and the exact interval of Eq. (5); the
wider limit on each side is kept.

7.2 Group level, reported as the primary robustness check:
- group-level sign-flip randomization test: per group g, D_g = sum over its rows
  of (win in condition B - win in condition A); statistic T = sum of D_g; under
  the null the condition labels are exchangeable within a group, so the sign of
  each D_g is flipped independently; exact enumeration when at most 20 groups
  have D_g != 0, otherwise 100,000 Monte Carlo flips with seed 20261012;
  two-sided p = (1 + #{|T*| >= |T|}) / (1 + B)
- cluster bootstrap: 10,000 resamples of whole groups with replacement, seed
  20261012, percentile 95% interval of d
- cluster-robust (CR1) Wald interval of d
- Holm over the six models within each contrast family
- the bound quoted in the text is the largest gain-side limit among the
  row-level bound, the cluster bootstrap limit and the CR1 limit; a family-wise
  version uses the same constructions at level 0.05 / 6

On the paired track the same group-level test is computed with the pair as the
group (48 pairs) beside the row-level McNemar test.

7.3 The 95 single-semester groups of the original broad sample are analysed with
the same code as a pre-specified subset.

## 8. Order and safety of the GPU queue

G0, then the paired cells of G1, G2 and G3, then the broad cells of G1, then G4
(fastest model first), then the optional G1b. Each cell is validated as soon as
it ends: row count and query set equal to the pool, rankings complete and
aligned with scores, no NaN or infinity, candidate sets and golds equal to the
pool, manifest consistent, every blind cell giving bit-identical rankings to the
rows of each group (and therefore exactly one win per group), and every profile
cell giving different inputs to the rows of every pair. A cell that fails is
retried once automatically and then held for the operator. The VRAM floor is the
project rule (1,800 MiB free), with the owner's 300 MiB exception for
FollowIR-7B and ctxl-6B from 2 October 2026, renewed for this queue only by the
owner's GO.

## 9. R1-1, R2-1: gold validity

9.1 Provenance. The workflow journal of 29 July 2026 (`wf_d095d1b4-baf`, kept in
`/home/io/bukti_asal_paired_29jul2026` with a sha256 manifest) logs every step:
150 topic pairs sent to writer agents; 115 questions written and 35 declined
with a stated reason; 115 judged once each against four tests; 52 passed and 63
failed with reasons; the 52 passing questions equal the released
`paired_queries.jsonl` text exactly. A table derived from the journal is
released. The manuscript sentence saying the 115 cannot be traced to the 150 is
corrected.

9.2 Human validation. Paired: the existing 104-item kit (both gold passages of
every row). Broad: 55 groups of the original sample drawn with seed 20261012,
stratified by the program of the source query with proportional allocation,
every gold of each drawn group rated (about 150 rows). Two raters who are not
authors, blind to program, semester and book. Per gold passage: answers the
question fully, partly or not at all; per question: clear and natural (yes or
no). A gold is valid when it answers fully or partly. Disagreements are
settled by discussion between the two raters, or by a third non-author.
Reported: raw agreement, Cohen's kappa (valid vs not, and three classes), Gwet's
AC1, the error rate (share of rows whose gold is not valid) with a cluster
bootstrap interval by group, and the share of groups whose golds are all valid.

9.3 Sensitivity. (a) Analytic: if a share e of rows has an invalid gold that
carries no signal, a true effect is attenuated by (1 - e); if invalid golds
reverse the signal, by (1 - 2e). Bounds are rescaled with the upper limit of e.
(b) Empirical: every contrast of section 7 is recomputed on rows whose gold is
valid and on groups whose golds are all valid. Until the sheets return, (a) is
reported on a grid e = 0.05, 0.10, 0.20.

## 10. Release (R1-8, editor)

A public repository built from the analysis package with: scoring and analysis
scripts, exact model revisions and prompts, query texts and gold and candidate
chunk identifiers with their sha256 anchors, group assignments, gate-version
labels, seeds, all pre-registrations, and machine-readable per-row outcomes
for every table and figure. Book text from the 12 conditionally licensed
documents is not redistributed; identifiers and anchors allow reconstruction.

## 11. Deviations

### 11.1 Amendment of 12 October 2026, committed before any new cell

Logged after two internal reviews of the scripts (operations; statistics) and
before any new cell ran. At that time only three G0 gate cells had run
(Qwen3-Reranker-0.6B, bge-gemma and zerank-2, paired `both`), each identical to
its canonical file (zero flips, maximum score difference 0). Nothing below
depends on a new score.

Design
- (a) bge-gemma placement. Section 3 bases gemma's slot conclusion on
  `slot_vendor` only, but its G2, G3 and G4 cells use the appended placement of
  the frozen paper cells. Two G4 cells are added: gemma `slot_vendor` and
  `both_vendor` on the new broad sample (99 cells in total). Every R1 statement
  about gemma's slot rests on vendor-placement cells only (G1 and these two).
  Its G2 and G3 rows and its appended G4 cells are reported as secondary and
  labelled with their placement.
- (b) Coincident wrong labels in G3. The wrong-value rule excludes only the
  passage's own value. A wrong semester can therefore equal the other gold's
  semester, which reverses the cue on that row. The primary analysis uses all
  comparable rows, as the frozen `wrong` cell does. The same contrasts without
  coincident rows are a pre-specified sensitivity analysis. A claim that label
  content matters needs the primary contrast to be significant and the
  sensitivity contrast to point the same way.

Inference (refines sections 4, 5 and 7)
- (c) Decisive test. Every "differs" or "significant" statement in sections 4
  to 6 is decided by the row-level exact McNemar test with Holm over the six
  models, as in the manuscript. The group-level test is reported beside it as
  the robustness check. No statement that a cell or control does not differ
  rests on non-significance alone. It is stated as a bound: the family-wise
  bound of section 7.2, computed for every contrast family (G1, G2, G3, G4, the
  uniform-semester subset and the strata), not only for slot versus none.
- (d) The group-level sign-flip test is approximate. D_g is not symmetric under
  the null, because a blind cell has exactly one win per group. It is labelled
  approximate and reported with the number of groups whose D_g is nonzero. When
  fewer than 20 groups are nonzero, the group-level interval is a t interval
  with G - 1 degrees of freedom on the cluster-robust standard error instead of
  the percentile bootstrap.
- (e) When a contrast has no discordant rows, the conditional exact interval of
  Eq. (5) collapses to zero. The bound then also uses the Newcombe hybrid score
  interval for a paired difference of proportions (method 10), which does not
  collapse. The wider limit is kept.
- (f) Shared passages. CR1 and the cluster bootstrap are also computed with the
  book set of a group's golds as the cluster.
- (g) Gold-error rescaling (section 9.3).
  - 1/(1 - e) and 1/(1 - 2e) are reported as models. The worst case
    d <= (U + e) / (1 - e) is reported beside them.
  - e is the share of rows touched by any invalid gold of their group. It is
    estimated per track from that track's own kit, with an exact one-sided 95%
    upper limit (Clopper-Pearson), so zero errors in n items gives
    1 - 0.05^(1/n), not zero.
  - Section 9.3(b), the contrasts on rows whose golds are all valid, is
    computed when the rater data exist.
- (h) Exploratory tables. These five tables on existing data were built before
  this file was committed: `r1_klaster_luas`, `r1_klaster_paired`,
  `r1_semester_seragam`, `r1_sensitivitas_gold` and `r1_asal_paired`. They are
  exploratory, not confirmatory. The same code rebuilds them at the end of the
  queue.

Gate and queue
- (i) The float16 tolerance of section 2 becomes 1.5e-6. Scores are stored with
  six decimals, so 1e-6 equals one storage quantum, and a one-quantum rounding
  difference failed the check through floating-point representation. The new
  tolerance allows one quantum, not two.
- (j) VRAM.
  - The scorers' in-run guard (`--gap-mib`, `--min-free-mib`) is 300 MiB for
    every FollowIR-7B and ctxl-6B cell. Their GPU-cap decision then loads both
    models fully, as in the October re-run.
  - The guard is also 300 MiB for the bge-gemma vendor cells, as section 3
    says. zerank-2 keeps its canonical `--min-free-mib 300`.
  - The section 8 floor is enforced by the runner before each cell: free VRAM
    must be at least the measured peak plus 1,800 MiB, or plus 300 MiB for
    FollowIR-7B and ctxl-6B. Their peaks are raised to the measured process
    totals of 14,800 and 14,750 MiB.
  - The guard changes no score unless a model is offloaded. Every manifest
    records the cap decision.
- (k) Queue mechanics.
  - Every attempt starts from an empty output: an earlier attempt's partial
    output is moved to `_karantina/`, so no cell mixes two runs.
  - A cell that cannot start for lack of VRAM is deferred to the second round
    instead of blocking the cells behind it.
  - G4 runs only on a resample pool whose row count and sha256 match its
    manifest. The pool is written atomically after a second check that every
    source query is in train or dev.
  - The runner writes this file's sha256 and commit into the manifest of
    every passed cell, as the preamble promises.
  - The manifest of `both_duasalah` no longer carries the legacy note "only
    the department field is replaced". Its R1 block describes the cell.

### 11.2 Correction of 11.1, committed during G0, before any new cell

- (e) The interval named in 11.1 (e) is wrong. Newcombe's method 10 also
  collapses to zero when the two proportions are equal and fully correlated,
  as with no discordant rows at a resolved rate of 0.5. The unconditional
  interval is therefore Tango's score interval for a paired difference
  (Tango 1998, Statistics in Medicine 17:891-908). With no discordant rows its
  limit is z^2 / (n + z^2), 3.8 pp at n = 96.
- (g) The Clopper-Pearson limit is taken on the share of invalid golds among a
  kit's items. It is converted to rows as 1 - (1 - e_gold)^k for a group with k
  golds, averaged over the track's rows.
