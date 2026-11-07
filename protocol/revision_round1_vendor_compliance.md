# How each reranker was called, against its vendor documentation (IJIES R1-5, R1-8)

Checked on 12 October 2026 against the model cards cached at the exact revisions
of Table 2, by tokenizing on CPU with the same tokenizers (transformers 4.55.0,
Hugging Face offline). No weights were loaded for these checks. The check
scripts are kept with the revision evidence
(`<home>/r1_alat/cek_vendor/cek_gemma.py`, `cek_lain.py`, `cek_panjang.py`).

| Model (revision) | Vendor input and score | Where an instruction goes, per vendor | Our construction |
| --- | --- | --- | --- |
| Qwen3-Reranker-0.6B, 4B (e61197ed4502, 22e683669bc0) | `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}` inside the card's fixed prefix and suffix; score = P(yes) from the yes/no logits at the last position | the `<Instruct>` field | token-identical to the card for default, profile and labelled inputs |
| FollowIR-7B (4d25d437e38b) | the card's template; score = softmax(false, true)[true] at the last position | appended to the query text, as in the card's example | template string-identical; the card's double BOS is reproduced |
| ctxl-rerank-v2-instruct-multilingual-6b (f14ca1a2fc20) | `format_prompts` of the card; score = logit of token 0 at the last position | optional instruction after the query | equal to the card's `format_prompts` output for every cell type |
| zerank-2 (5eae30d5ee3c) | (query, document) pairs through `CrossEncoder.predict`; score = the "Yes" logit | not documented on the card; the query string is the only channel | all counts in the text come from the vendor's own scoring code (sentence-transformers 6.0.1); the instruction strings are ours; the card's bare-query input gives 48 of 96, the same as our generic prefix |
| bge-reranker-v2-gemma (1787044f8b6f) | `get_inputs` of the card: `<bos>A: {query}\nB: {passage}\n{prompt}`; score = the "Yes" logit at the last position; the default prompt ends "by providing a prediction of either 'Yes' or 'No'." | the `prompt` argument, which replaces the default prompt | see below |

## bge-reranker-v2-gemma

Our inputs are token-identical to the card's `get_inputs` for every prompt we
use. The difference the reviewer points to is where the profile sits:

- `none`: the default prompt; the "Yes" logit is read at the final "." of the
  documented request. As documented.
- `slot` and every profile cell built on it (`both`, `wrong`, the single-field
  and wording cells, and the FollowIR control): the profile sentence is appended
  after the default prompt, so the last token is profile text ("... semester 1")
  and the "Yes" logit is read after the profile. Not the documented placement.
- `slot_vendor`, and the new `both_vendor` and `wrong_vendor`: the profile is
  written into a replacement prompt that ends with the documented request, so the
  last token is again the final "." This is the documented placement. Every
  revision statement about bge-gemma's slot rests on these cells (G1 and the two
  G4 cells on the new broad sample). Its orthogonal (G2) and factorial (G3)
  cells keep the appended placement of the frozen paper cells and are reported
  as secondary (pre-registration 11.1 a).

## Length and precision

On the paired and broad reranking tracks no input reached a length cap. Upper
bounds over the longest instruction and label variants: paired track 779 tokens
for Qwen3 (cap 2,048), 795 for zerank-2 (cap 1,536), 632 for bge-gemma query plus
passage; original broad track 916, 932 and 786. Every FollowIR-7B, ctxl-6B and
bge-gemma manifest of those tracks records zero truncated documents, and the
runner raises an alert if a cell of the new broad sample records any. The
FollowIR benchmark cells differ: long news documents are cut to the token budget,
and their control tables record how many (bge-gemma Robust04 22; FollowIR-7B
News21 84 and Robust04 64; ctxl-6B News21 2 and Robust04 34). Precision follows Table 2 (Qwen3 float16, the others bfloat16); where a
card loads another default precision or attention kernel, the difference changes
memory, not the input.
