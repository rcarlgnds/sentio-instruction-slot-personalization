> Translation of the original Indonesian protocol, which is kept alongside this file. The original is the dated record, and this file is a reading aid. Table names in this protocol are those of the run. In this package the tables have English names, and their manifests record the run names.

# Pre-registration addendum: following up the zerank-2 scorer equivalence test

Written 17 September 2026, *after* the section 7 test of
`PRAREGISTRASI_LINTAS_ARSITEKTUR.md` finished and *before* its follow-up ran.
This addendum records that test's result as it stands, states that its
criterion was misspecified, and locks the follow-up question and its criterion
before the numbers exist.

## 1. The section 7 result, verbatim from `tables/zerank2_equivalence.json`

| measure | value |
|---|---|
| Spearman rho, 100 pairs | 0.998466 |
| order inversions between paths | 16 (14 on queries with 71 candidates, 2 on queries with 29) |
| absolute difference A against B | median 0.0625, p90 0.125, max 0.25; 35 of 100 identical |
| difference between path A and the frozen file of the same cell | max 0.2909; 29 of 100 identical |
| pre-registered criterion (zero inversions and max <= 0.05) | **FAILED** |

The FAILED verdict stands as written. It is not loosened.

## 2. Why the criterion was misspecified, and what the data does show

Raw zerank-2 logits on this pool run from -13.19 to 13.94. At magnitudes of 8
to 16 the bfloat16 resolution (one ulp) is 0.0625. The median difference of
0.0625 is therefore exactly one ulp, and the maximum of 0.25 is four ulps. The
tolerance of 0.05 I wrote sits *below* one ulp at the logit magnitudes that
actually occur. It was set without checking the logit scale first, and that was
my mistake.

Two facts decide how to read it:
- The *same* project scorer, rerun on the same input, differs from its own frozen
  output by up to 0.2909. Run-to-run noise within one piece of code is *already*
  as large as the difference against the vendor path.
- All 16 inversions are candidate pairs whose score gap is <= 0.1875 on path A
  and <= 0.25 on path B, that is, pairs that are tied within bf16 precision.
  Not one inversion occurs on a pair separated by more than four ulps.

The conclusion that may be drawn: there is no *systematic* difference between the
project scorer and the vendor's official path, only bf16 rounding noise that
also lives between two runs of the same scorer. The conclusion that may *not* be
drawn: that the original criterion passed. Both are written into Appendix B of
the write-up.

## 3. The follow-up question, locked now

What matters for the write-up is not per-pair logit agreement but whether the
vendor's official path reproduces the *two frozen* zerank-2 *cells* the write-up
cites: 48 of 96 (generic instruction) and 47 of 96 (profile in the slot).

- The official path `CrossEncoder.predict()` (sentence-transformers 6.0.1 in
  the `system/.st-libs` overlay, bf16, max_length 1536) scores the *whole* paired
  pool, 104 rows, under both conditions, with the query text formatted exactly
  as in the frozen cell (plain slot format: instruction, blank line,
  "Query: ...").
- The output is written in the standard cell format and read by `metrics` and
  `mcnemar` without modification.
- Reported: the win counts of the official path against the frozen one for each
  condition, how many of the 96 rows carry the same verdict, the list of rows
  that differ, and an exact McNemar of generic against profile on the official
  path.
- Pass criterion, fixed now: win counts *identical* to the frozen cell under *both*
  conditions (48 and 47). Per-row agreement is reported but is not a condition,
  because a row tied within precision can flip without changing the count.
- If it does not pass: every zerank-2 cell is rescored through the official
  path, the write-up's numbers are updated, and both results (section 7 and
  this one) are reported. Hiding either is not an option.

Script: `scripts/zerank2_equivalence_full.py`. Output:
`tables/zerank2_equivalence_full.json`,
`raw/scores_zerank2official_zerank-2_paired_{default,profile}.jsonl`.

## 4. Follow-up result, written 17 September 2026 *after* the numbers existed

Sections 1 to 3 above are unchanged, word for word. This section records the
result of `tables/zerank2_equivalence_full.json` (543.3 s of GPU,
sentence-transformers 6.0.1, transformers 4.55.0) and the status of its
consequence.

| condition | official path | frozen cell | counts identical | same verdict per row |
|---|---|---|---|---|
| generic (default) | 48/96 | 48/96 | yes | 96 of 96 |
| profile in the slot | 48/96 | 47/96 | *No* | 95 of 96 |

The section 3 criterion (identical counts under *both* conditions): **FAILED**,
by one row, `paired-91-csci`, under the profile condition.

The anatomy of that row. In the frozen cell the gold scores 3.40625 and its
counterpart gold 3.578125 (a gap of -0.17, gold loses). On the official path
the gold scores 3.609375 and the counterpart 3.59375 (a gap of +0.02, gold
wins). Both gaps sit inside the run-to-run noise of the *same* scorer recorded in
section 2 (up to 0.29). Under the profile condition, 8 of the 96 comparable
rows have |gold score - counterpart score| <= 0.25 on both paths, and one of
them flips. Under the generic condition, 6 rows (frozen) and 8 rows (official)
sit in that band and none flips. Each cell's win count therefore carries an
uncertainty the size of its tied-within-precision row count, roughly 6 to 8 of
96, and that holds for either path.

What does not change: McNemar of generic against profile on the official path
is 8 against 8 discordant, p = 1.0; on the frozen cells 7 against 8, p = 1.0.
The slot-effect equivalence bound at 8/8 discordant is 8.45 percentage points,
against the frozen zerank-2 bound of 8.98. Official-path nDCG@5 is 0.5710
(generic) and 0.5342 (profile) against the frozen 0.5769 and 0.5313.

The consequence pre-registered in section 3 is to rescore *every* zerank-2 cell
on the official path. The zerank-2 cells the write-up cites: 6 paired-track
cells, 2 vendor-format cells, 2 single-field label cells, 7 injection-probe
cells, 4 broad-track cells over 1,621 rows, and 1 Core17 run, so 22 cells. Cost
estimated from today's measurement: a paired-sized cell takes about 4.5 minutes
(543 s for two cells), a broad-track cell 15.6 times that, and Core17 took
2,800 s on the frozen run, so roughly 6 to 7 GPU hours in all.

Status as this section is written: that consequence has *not* been carried out.
The decision is left to the first author, with two reasons stated openly: the
GPU cost above, and the judgment that rescoring trades one draw of bf16 noise
for another, since the same scorer against itself already differs by as much as
the two paths differ. Whichever way it goes, the results of section 1 and
section 4 both enter the write-up. If the consequence is not carried out, the
write-up states it as a deviation from the pre-registration together with the
reasons above, and the frozen numbers stay quoted as they are alongside the
official-path numbers.
