> Translation of the original Indonesian protocol, which is kept alongside this file. The original is the dated record, and this file is a reading aid. Table names in this protocol are those of the run. In this package the tables have English names, and their manifests record the run names.

# Cross-architecture pre-registration, addendum 3: the instruction slot in each vendor's own sentence style

Written on the morning of 18 September 2026, *before* a single cell below was
run. The state at that moment: every paired cell, every G cell, and Core17 for
FollowIR-7B, ctxl-6B, and bge-reranker-v2-gemma were frozen; the zerank-2
official-path replay had only Core17 and the probe left; E and F had not
started. The trigger: the profile instruction for all three new models used the
same imperative sentence as Qwen3 ("Given a query from a <DEPT> student in
semester <N>, retrieve passages ..."), while each vendor documents a different
instruction style. The objection to be closed: "the slot is zero because the
sentence is not in that model's style".

## The cells

One cell per model, `--cell slot_vendor` in `score_architectures.py`, the paired
track of 96 comparable rows, the profile in the slot *only*, no document label,
identical to the `slot` cell except for the text of the instruction and where
it goes:

- FollowIR-7B: the narrative style of its README and its training data,
  appended to the query text (the README's convention of putting the
  instruction inside the query):
  `A relevant document would come from a textbook assigned to the <DEPT> degree
  program and would suit a student in semester <N>, rather than from a textbook
  of another degree program.`
- ctxl-6B: a short custom instruction in the style of the README's example
  ("Prioritize recent medical research"), appended after the query with a
  single space:
  `Prioritize passages from the <DEPT> degree program's own textbooks at a level
  suitable for a semester <N> student`
- bge-reranker-v2-gemma: the README's `prompt` argument in the form of a task
  description. The frozen `slot` cell appends the profile *after* the default
  prompt; this cell *replaces* the default prompt entirely with a task
  description that carries the profile:
  `Given a query A from a <DEPT> student in semester <N> and a passage B,
  determine whether the passage comes from that student's own degree program
  and contains an answer to the query at a level appropriate for semester <N>
  by providing a prediction of either 'Yes' or 'No'.`

Template, score, precision (bf16), the 1,536-token limit, the 4,000-character
cut, and the 300 MiB VRAM gap are identical to the frozen cells. One
`--limit 2` smoke run for gemma (the new `set_prompt_penuh` code path) runs
first and does not count.

## Hypothesis and analysis, locked now

H0: all three cells stay indistinguishable from the lock of 48/96. Expected
range: 44 to 52. Test: an exact McNemar of each cell against the same model's
frozen `none` cell (family V, Holm m=3) and, as a comparator, against the same
model's frozen `slot` cell. The slot effect bound is computed with the same
`analisis()` function (the wider of Wald and Clopper-Pearson).

Reading rule: if any cell clears Holm at 0.05 above the lock, the "slot is
zero" claim for that model is *withdrawn* from the write-up, its `text` and
`both` cells are rerun in the vendor style, and the vendor style becomes that
model's primary slot cell. If not, all three cells are reported in one sentence
in V-D (Not a Matter of Prompt Craft) with their text verbatim in the appendix.
Whatever the result, it is reported.

What is *not* changed: every test family of the main pre-registration, addendum
1, and addendum 2. These cells form a separate diagnostic family (V) and do not
join family E.
