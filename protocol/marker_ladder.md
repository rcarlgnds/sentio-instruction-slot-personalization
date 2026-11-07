> Translation of the original Indonesian protocol, which is kept alongside this file. The original is the dated record, and this file is a reading aid. Table names in this protocol are those of the run. In this package the tables have English names, and their manifests record the run names.

# Pre-registration: the inference-distance ladder on the document channel

Written and committed *before* a single cell was run. The analysis rules below
bind, predictions included, so that no result can be reinterpreted afterwards.

Date: 20 September 2026
Parent pre-registration: cross_architecture.id.md and addenda 2, 3, 4
Config lock: 0a7b2e4de856

## 1. The question

The write-up reports that the document channel works: the profile in the slot
plus a label in the candidate text carries five of six rerankers from 48 to
between 82 and 96 of 96. One objection is still unanswered. The injected label
reads `[Source: STAT degree program textbook, semester 2]` while the
instruction reads `a STAT student in semester 2`. Both carry exactly the same
tokens. The positive result could therefore be string matching rather than
matching of meaning.

The question: how far an inference distance between the asker's profile and the
marker in the document can a reranker still bridge?

## 2. Why the ladder runs on the *program-only* label

The frozen single-field control (label_satu_field.json and
papan_saluran_paired_arsitektur.json) shows a semester-only label almost
matching the full label, 78/84/93/91/96/51 against 82/84/95/90/96/51. The
semester dominates. A ladder run on the full label would sit flat near the
ceiling and measure nothing.

The program-only label has real range, 62/74/76/52/83/49 against the blind lock
of 48. That is where the effect of the program marker's surface form can show.

## 3. Four rungs

All use the 96-row paired track, six models, and the frozen English profile
instruction. What changes is *only* the surface form of the program marker in the
candidate text. The semester is not included in any rung.

| rung | injected label | program marker | status |
| --- | --- | --- | --- |
| a. literal code | `[Source: STAT degree program textbook]` | the code, identical to the instruction | *Already measured*, the "Program only" row of the single-field label table |
| b. program name | `[Source: statistika degree program textbook]` | domain name in Indonesian | new |
| c. natural header | `Program studi statistika.` | domain name, no Source frame | new |
| d. opaque code | `[Source: PRG-09 degree program textbook]` | institutional code with no semantic content | new |

The program names in rungs b and c are taken from `_DOMAIN_LABEL` in
`system/app/api/routes/query.py`, a mapping the deployed system already uses
for pre-OOD query expansion. That mapping was *not* built for this experiment, so
there is no freedom to pick favorable wording.

The opaque code in rung d has the form `PRG-<nn>`, with nn ascending by the
alphabetical order of the program codes in that mapping, fixed and recorded
in the script. The example in the table uses statistika, which falls on PRG-09.
That code appears in no corpus.

## 4. Predictions, written before running

Written so the result cannot be reframed once it is visible.

1. Rung d falls to the blind lock of 48 on all six models, by at most two rows.
   If it does not, there is a leak nobody knows about and the whole ladder
   cannot be interpreted.
2. Rung b falls below rung a on most models. The size is not predicted.
3. Bge-reranker-v2-gemma stays near the lock on all four rungs, because no cell
   has ever moved it.

## 5. Interpretation rules, binding

- Rung b is compared against a with an exact paired McNemar test on the same 96
  rows, Holm-corrected within the family of six models.
- If b falls to within two rows of the lock on most models, the write-up *must*
  state that the document channel works through lexical matching, and the first
  contribution is rewritten as it stands. This result may not be hidden or
  reported only in an appendix.
- If b holds at a distance not significantly different from a, the write-up may
  state that the document channel carries matching of meaning rather than of
  strings.
- If b falls significantly yet stays well above the lock, the write-up states
  both, a lexical component *and* a meaning component, with the numbers.
- Rung c is declared *exploratory* because it changes two things at once, the
  form of the marker and the frame of the sentence, so it is not used to
  conclude anything about lexical against meaning. Its value is practical: if c
  works, the write-up's recommendation changes from injecting a synthetic label
  to writing a program header inside the chunk.
- No frozen cell is overwritten. No old number is changed.

## 6. Artifacts

Script: study/channel/scripts/score_architectures.py, new cells `label_nama`,
`label_alami`, `label_opak`. Aggregator:
study/channel/scripts/inference_ladder.py. Table:
study/channel/results/tables/tangga_inferensi.json.
