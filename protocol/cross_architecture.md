> Translation of the original Indonesian protocol, which is kept alongside this file. The original is the dated record, and this file is a reading aid. Table names in this protocol are those of the run. In this package the tables have English names, and their manifests record the run names.

# Pre-registration: the cross-architecture test, the zerank-2 scorer equivalence test, and the bare-query cell

Written 17 September 2026, *before* a single new cell was run. When this file was
written, all that existed on disk for the new families was the model weights in
the Hugging Face cache. No score file, no table. The purpose of this file is to
lock the models, the cells, the inference contracts, the metrics, the tests,
and the reporting rules, so that no favorable selection can be made later.

Motivation (QC 17 September, Appendix F): the null "the instruction slot does
not carry personalization" currently rests on three models of one Qwen3 lineage
(Limitation 13). This test adds three other lineages. A result in either
direction is reported.

## 1. Population and instruments

- Paired track: `study/retrieval/results/raw/pools_paired.jsonl`, 104
  rows, 96 comparable (gold and counterpart both present in the pool). The
  pool's sha256 is recorded in every manifest. *There is no other pool*.
- External positive control: FollowIR Core17
  (`mteb/Core17InstructionRetrieval`), 40 queries, 1,000 candidates per query,
  p-MRR under the verbatim MTEB implementation already used by
  `followir.py`. The protocol is identical to the frozen Qwen3 and zerank-2
  cells.
- Holdout seal: the same `gerbang_segel` gate. The paired pool holds no
  test_holdout query_id.

## 2. The models, and why these three

| family | model | architecture (cached config.json) | lineage |
|---|---|---|---|
| gemma | `BAAI/bge-reranker-v2-gemma` | GemmaForCausalLM | Gemma (Google) |
| ctxl | `ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b` | MistralForCausalLM, 20 layers, hidden 5120 | Mistral |
| followir | `jhu-clsp/FollowIR-7B` | MistralForCausalLM, 32 layers, hidden 4096 | Mistral, tuned specifically to follow retrieval instructions |

The architecture is read straight from the cached snapshot's `config.json`, not
from the name. `ctxl-rerank-v2-instruct-multilingual-1b` and `-2b` are
*deliberately* not used, because their config is `Qwen3ForCausalLM`. No
quantization: this project's own kappa study already measured quantization
shifting a verdict by as much as a change of model family.

## 3. Inference contracts, copied from the README of the same snapshot as the weights

All three contracts are stored verbatim under `kontrak_inferensi` in each
cell's manifest.

gemma. The README's `get_inputs()` is copied verbatim (the `A: `/`B: `
prefixes, manual bos, the `\n` separator, the prompt at the tail,
`prepare_for_model` with `only_second` truncation, `pad_to_multiple_of=8`).
Score = the raw logit of the token `Yes` at the last position, no softmax. The
official instruction slot is the `prompt` argument. The vendor's default prompt
verbatim: "Given a query A and a passage B, determine whether the passage
contains an answer to the query by providing a prediction of either 'Yes' or
'No'."

ctxl. The official chat template (the snapshot's `chat_template.jinja`): a
`system` role (the instruction), `query`, `document`, bos from the template,
and the ` ??` tail. Score = the raw logit of token id 0 at the last position
(`1_LogitScore/config.json`: true_token_id 0, false_token_id null), exactly the
vendor's `predict()`. The official instruction slot is the `system` message,
which is the `prompt=` argument of `predict()`. The vendor default is *no*
instruction ("The prompt argument is optional").

followir. The README template verbatim (`<s> [INST] You are an expert Google
searcher ... Query: {query}\nDocument: {text}\nRelevant (only output one word,
either "true" or "false"): [/INST] `), left padding, pad = eos, score =
softmax(logit false, logit true)[true]. The README tokenizes a string that
already contains `<s>` with the tokenizer's default `add_special_tokens`, so
BOS appears twice. That is the vendor contract and is reproduced as is. The
instruction slot is appended to the query text ("Query: {query}
{instruction}"), FollowIR's own convention ("input of query + instruction
inside the template").

## 4. The cells that run, all of them reported

The profile instruction is verbatim the same as in every Qwen3 and zerank-2
cell (`instruksi_profil` in `score_arms.py`): "Given a query from a {dept}
student in semester {sem}, retrieve passages from that student's own study
program that answer the query at a level appropriate for semester {sem}". The
document text label is verbatim the same (`label_sumber`,
`label_sumber_salah`), attached *after* the text is cut to 4,000 characters,
exactly as in the frozen cells.

Per new model, five core T1 cells:

| cell | instruction slot | document text |
|---|---|---|
| none | vendor default (gemma: the default prompt; ctxl: no system; followir: a plain query) | plain |
| slot | the profile in that family's official slot | plain |
| text | vendor default | + the correct source label |
| both | the profile in the slot | + the correct source label |
| wrong | the profile in the slot | + a source label with the *wrong* program, semester still correct |

How the profile enters the slot per family: gemma prompt = default prompt + " "
+ profile instruction (the default prompt carries the Yes/No task frame the
score depends on, so it is added to rather than replaced); ctxl system = the
profile instruction; followir query = query + " " + profile instruction.

Secondary cells, fixed now rather than after seeing results:
- ctxl `none_generic`: system = the generic `TASK_DEFAULT` instruction ("Given
  a web search query, retrieve relevant passages that answer the query"), to
  check whether the vendor's "none" (no system) and the paper's convention (a
  generic instruction) give the same lock. Reported side by side.
- zerank-2 `bare`: a bare query with no `TASK_DEFAULT` prefix, same family and
  same frozen `SkorerZerank2` with no change. This closes a reviewer's question
  about the zerank-2 "None" cell, which until now used the generic prefix.

T2, per new model: FollowIR Core17, protocol identical to `followir.py`.

## 5. Technical parameters, fixed before running

- dtype bfloat16 for all three models (ctxl and followir are released in bf16;
  the gemma README states no dtype, and bf16 is chosen to match zerank-2).
  Recorded in the manifest.
- Maximum length 1,536 tokens, the same as zerank-2. Truncation is applied to
  the *document tokens* so that the template frame (instruction, query, template
  tail) is never cut, and the number of truncated documents is recorded per
  cell (`n_dokumen_terpotong`). The document text is cut to 4,000 characters
  first, as in every frozen cell.
- `logits_to_keep=1`: only the last position's logits are computed. That saves
  memory and does not change the value of that logit.
- The batch size is searched automatically from large to small until free VRAM
  stays above the set limit, and the final batch value is recorded. Scores do
  not depend on the batch (every document is scored with left padding and an
  attention mask).
- The free-VRAM limit for this run is 500 MiB, on the machine owner's
  instruction of 17 September, replacing the project's 1,800 MiB floor *for this*
  *run only*. The runtime guard aborts the cell, not the desktop, if free memory
  falls below the limit.
- ctxl-6b (14,240 MiB of weights) does not fit whole under that limit, so it is
  loaded with `accelerate`'s `device_map` and a GPU memory cap. Layers that do
  not fit are held in RAM and streamed to the GPU during the forward pass. The
  arithmetic is unchanged, only where the weights are stored. The number of
  offloaded modules is recorded in the manifest.
- One model on the GPU at a time, under the same `.gpu.lock` file lock as every
  earlier cell.
- Each cell's output has the *identical* format to the frozen cells (`query_id,
  gold, counterpart, ranking, scores, latency_s`), ordered by `(-score, id)`,
  plus a manifest with the sha256 of the pool and of the output, the
  `config_version` lock `0a7b2e4de856`, the duration, peak VRAM, the batch
  size, and the truncation count.

## 6. Metrics and tests, fixed before the numbers exist

- Metrics: `metrics.metrik_paired` and `mcnemar.hasil_per_item`, *without*
  modification. The replication gate `metrics.gerbang_replikasi()` must pass
  before a single cell is read (it reproduces 0.5000 / 0.5104 / 0.5000 / 0.5208
  from the published C3/C4 cells).
- The structural blind lock of 48/96 applies as before: a cell without a
  profile is *locked* at 48 algebraically and is *not* tested against a null of
  0.5.
- Exact McNemar contrasts per model: none vs slot; slot vs both; text vs both;
  wrong vs both. Holm correction within family E = the three none-vs-slot
  contrasts (m = 3), kept apart from the published family so that published
  p_holm values do not change.
- The slot-effect equivalence bound per model: the `analisis()` function from
  `equivalence_bounds.py` (the wider of the Wald and the exact conditional
  Clopper-Pearson interval), reported as an interval rather than tested against
  a margin of the authors' choosing.
- Positive control: Core17 p-MRR per model. A model with a p-MRR below zero is
  still reported. It only changes the reading (a model that does not follow its
  instruction even on an external instrument), it is not dropped.
- *Reporting rule*: every cell, every contrast, every model is reported as it
  stands. A model that leaves the lock through the slot is a finding that
  changes the write-up's claim, not a reason to rerun or drop a cell. A cell
  that fails or is aborted is reported with its reason. No cell is rerun under
  different settings except for the batch size (which does not change scores),
  and that is recorded.
- The new tables are written to *new* files,
  `papan_saluran_paired_arsitektur.json` and
  `followir_Core17_<family>_<model>.json`. The frozen
  `papan_saluran_paired.json` is not touched.

## 7. The zerank-2 scorer equivalence test

The claim under test: the project's `SkorerZerank2` scores (a reimplementation
through `AutoModelForCausalLM` + chat template + the true-token logit) are
equivalent to the vendor's official
`sentence_transformers.CrossEncoder.predict()`.

- Library: sentence-transformers 6.0.1 installed into a separate overlay
  `system/.st-libs` (`--no-deps --target`). The venv and the transformers 4.55
  overlay are unchanged. Versions are recorded.
- 100 pairs: iterate `pools_paired.jsonl` in file order and take
  (query_text, document_text) pairs in pool candidate order until there are
  exactly 100. query_text is formatted exactly as in the frozen `default` cell:
  `TASK_DEFAULT + "\n\nQuery: " + query`, and the document is cut to 4,000
  characters. There is no other pair selection.
- Two paths on identical input: (a) `SkorerZerank2.skor` in bf16 at max_len
  1536, (b) `CrossEncoder("zeroentropy/zerank-2", model_kwargs dtype bf16,
  max_length 1536).predict()`.
- Reported: Spearman rho over all 100 pairs; per query, the number of order
  inversions between the two paths (candidate pairs whose order differs); the
  maximum and median absolute difference; the number of pairs whose difference
  exceeds the bf16 tolerance of 0.05. Plus a check that path (a) reproduces the
  scores in the frozen
  `scores_zerank2_zerank-2_paired_default.jsonl` for the same pairs (the
  maximum difference is recorded).
- Pass criterion, fixed now: zero inversions across every query *and* a maximum
  absolute difference <= 0.05. If it does not pass, the result is still
  reported, and the consequence (rescoring the zerank-2 cells) is decided
  afterwards in the open rather than hidden.

## 8. What is *not* run here

T3 (the 1,621-row xpair replication for the new families) is not run. The
reason is recorded in QC 17 September, Appendix F.5. If it is ever run, it uses
the same scripts and cells with no change of settings.

The profile-shuffle control on the main benchmark will get its own
pre-registration addendum *before* it runs. It uses the same evaluation harness
as the frozen arm B and must first pass the replication gate of 0.8113 nDCG@5
on unshuffled profiles.

## 9. Execution order and timekeeping

Order: the zerank-2 equivalence test, the zerank-2 bare cell, T1 gemma, T1
followir, T1 ctxl, T2 gemma, T2 followir, T2 ctxl. Every cell records its
duration and peak VRAM. Total GPU time is reported from the manifests, not from
an estimate. Running status is written to `results/raw/arsitektur_status.json`
and read by `scripts/architecture_status.py`.
