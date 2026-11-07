> Translation of the original Indonesian protocol, which is kept alongside this file. The original is the dated record, and this file is a reading aid. Table names in this protocol are those of the run. In this package the tables have English names, and their manifests record the run names.

# Cross-architecture pre-registration, addendum 4: the profile instruction in the language of the query

Written 18 September 2026 at 08:30 WIB, *before* a single cell below was run. The
trigger: every slot cell used an English profile instruction over an Indonesian
query (the combination the Qwen3-Reranker model card recommends), while the
textual control that proves the slot can work used English queries. The
objection to be closed: "the slot is zero because the instruction's language is
not the query's language".

## The cells

One cell per model, six models, the paired track of 96 comparable rows, the
profile in the slot *only*, no document label, placed exactly as in each model's
frozen `slot` cell (the Qwen3 instruct field, plain text in front of the query
for the zerank-2 official path, after the query for FollowIR-7B and ctxl-6B,
appended to the default prompt for bge-reranker-v2-gemma). The instruction
text, a direct rendering of the frozen English sentence:

    Diberikan pertanyaan dari mahasiswa program studi <DEPT> semester <N>,
    ambil bagian teks dari program studi mahasiswa itu sendiri yang menjawab
    pertanyaan tersebut pada tingkat yang sesuai untuk semester <N>

Commands: `score_arms.py --instruction profile_id` (qwen3rr 0.6B batch 8
float16, 4B batch 4 float16, zerank2official batch 2 bf16 max-len 1536) and
`score_architectures.py --cell slot_id` (followir, ctxl, gemma, bf16 1536, gap 300).
Two `--limit 2` smoke runs (gemma slot_id, qwen3rr 0.6B profile_id) come first
and do not count.

## Hypothesis and analysis, locked now

H0: all six cells stay indistinguishable from the lock of 48/96. Expected
range: 44 to 52. Test: an exact McNemar of each cell against the same model's
frozen `none` cell (for Qwen3 0.6B and 4B, the novelty_1 default cells the
channel board cites; for zerank-2, the official-path default cell), family ID,
Holm m=6; as a comparator, against the same model's frozen English slot cell;
and the effect bound with `analisis()`.

Reading rule: if any cell clears Holm at 0.05 above the lock, the "slot is
zero" claim for that model is *withdrawn*, its `text` and `both` cells are rerun
with the Indonesian instruction, and Limitation 3 is rewritten. If not, one
sentence goes into V-D and the Limitation 3 sentence changes from "not yet
tested" to the result. Whatever the result, it is reported. The main
pre-registration and addenda 1 to 3 are unchanged.
