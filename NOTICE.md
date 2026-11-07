# Licenses and third-party material

## Code

The authors' code is under the MIT license (`LICENSE`), copyright 2026 Rico Aurelio Gunadi Sastra and Ahmad
Nurul Fajar. It covers the `.py` and `.js` files, the requirements files, and `system/app/.env.example`, apart
from the third-party code below.

- `study/channel/scripts/2_scoring/followir.py` has three functions adapted from MTEB
  (https://github.com/embeddings-benchmark/mteb, file `mteb/_evaluators/retrieval_metrics.py`):
  `get_rank_from_dict`, `rank_score`, and `calculate_pmrr`. MTEB is under the Apache License 2.0, a copy of which
  is in `LICENSES/Apache-2.0.txt`. The comment above the functions lists what was changed. The metric is the same.
- `GemmaScorer.get_inputs()` in `study/channel/scripts/2_scoring/architecture_scorers.py` is adapted from the
  `get_inputs` function in the model card of BAAI/bge-reranker-v2-gemma
  (https://huggingface.co/BAAI/bge-reranker-v2-gemma, Apache License 2.0). The tokenizer is bound to the object,
  the caller passes the prompt, and a counter of truncated documents was added.
- The scorers reproduce the input templates documented by Qwen3-Reranker, zerank-2, FollowIR-7B, and
  ctxl-rerank-v2. The templates belong to those model releases, whose licenses are listed in
  `study/channel/results/tables/model_table.json`. Code copied or adapted from a model card keeps the license of
  its source. This includes `CtxlScorer` in `architecture_scorers.py`. It follows the Transformers example in the
  model card of ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b at revision f14ca1a2fc20
  (https://huggingface.co/ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b), which is under CC BY-NC-SA 4.0
  (https://creativecommons.org/licenses/by-nc-sa/4.0/), and keeps the card's prompt and score. Unlike the card, it
  trims each document to the token budget before it builds the prompt, and the batching is our own code. The run
  records call this path copied verbatim.

## Results, records, and documents

These are under the Creative Commons Attribution 4.0 International license
(https://creativecommons.org/licenses/by/4.0/). Please credit them as "Sastra and Fajar, Measuring student
profile personalization in instruction-following rerankers: code and results, 2026, CC BY 4.0", with a link to
this repository. The license does not cover the third-party material below.

The paired-track questions in `study/channel/provenance/paired_queries.jsonl`, and the extended paired questions
and writer reasons in `study/channel/results/tables/p2_population_public.json`, were written in Indonesian by
language-model agents under the authors' direction. They fall under the same license.

## Book text

The 40 textbooks are not included. `study/_shared/document_licenses.json` records each book's license and how it
was checked, and `study/_shared/book_sources.csv` adds its source and the sha256 of the PDF the study parsed. The
record lists *Project Management*, 2nd edition, by Adrienne Watt as CC BY 4.0 with some CC BY-SA items, but the
book marks chapters 5, 7, 9, and 11 CC BY-SA 4.0 and chapters 3, 4, 10, 12 to 16, and 18 CC BY-NC-SA 4.0. No
text of that book is in this repository.

Short excerpts appear in four places: the first 200 characters of each gold and counterpart passage in
`paired_queries.jsonl`, quotations of up to 32 words in the judges' reasons in
`study/channel/results/tables/r1_paired_provenance.json`, two quotations in `verify_paper_numbers.py`, and the
twenty gold passages of the demo pool (`study/channel/results/raw_p2/pools/demo_pool_p2.jsonl`). They are plain
text extracted from the PDFs, cut to a fixed length, with the extraction artifacts left in, and they keep the
license of their book:

- David M. Lane, *Online Statistics Education: A Multimedia Course of Study*, Rice University,
  http://onlinestatbook.com/. Public domain.
- Barbara Illowsky and Susan Dean, *Introductory Statistics 2e*, OpenStax, Rice University, December 2023 printing,
  https://openstax.org/details/books/introductory-statistics-2e. CC BY 4.0 in that printing, while newer versions
  of the title are CC BY-NC-SA 4.0. OpenStax and Rice University do not endorse this repository.
- The demo pool, with the file names in `document_licenses.json`:
  *Accounting Principles: A Business Perspective*, two volumes (`ACCT_Accounting_Principles_GlobalText_Hermanson_CCBY3.pdf`,
  `ACCT_AccountingPrinciplesVol2_Hermanson_CCBY3.pdf`), CC BY 3.0;
  *Java, Java, Java: Object-Oriented Problem Solving*, third edition (`CSCI_Java_Java_Java_3e_CCBY4.pdf`), CC BY 4.0;
  *Business Processes and Information Technology* (`ISYS_Business_Processes_and_IT_GlobalText_CCBY3.pdf`), CC BY 3.0;
  *Information Systems for Business and Beyond*, 2014 edition
  (`ISYS_Information_Systems_for_Business_and_Beyond_EDISI_2014_Bourgeois.pdf`), CC BY 3.0;
  *Financial Strategy for Public Managers* (`MGMT_Financial_Strategy_Public_Managers_CCBY4.pdf`), CC BY 4.0;
  *Introduction to Supply Chain and Operations Management* (`MGMT_Intro_SupplyChain_Operations_Mgmt_KU_CCBY4.pdf`), CC BY 4.0;
  *Introductory Business Statistics* by Mahbobi and Tiemann (`STAT_IntroBusinessStats_MahbobiTiemann_CCBY4.pdf`), CC BY 4.0;
  and David M. Lane, *Introduction to Statistics* (`STAT_Introduction_to_Statistics_Lane_PublicDomain.pdf`), public domain.

## Data and models not included

The FollowIR collections (Core17, News21, and Robust04, from the `mteb/*InstructionRetrieval` datasets) have
their own access terms. Core17 rests on the New York Times Annotated Corpus from the Linguistic Data Consortium,
and Robust04 on TREC Disks 4 and 5 from NIST, so only per-query metrics are included.

No model weights are included. Five of the six rerankers are under Apache 2.0, and
`ctxl-rerank-v2-instruct-multilingual-6b` is under CC BY-NC-SA 4.0, so running its cells means accepting
non-commercial terms. bge-reranker-v2-gemma is built on Google's gemma-2b, so the Gemma Terms of Use may also
apply.
