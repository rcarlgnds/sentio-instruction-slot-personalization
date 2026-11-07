# Measuring Student Profile Personalization in Instruction-Following Rerankers for University Course Question Answering

This repository contains the code, result tables, and per-row scores of the paper by Sastra and Fajar, which is under
review at the International Journal of Intelligent Engineering and Systems.

The paper tests whether a student profile helps when it is written in the instruction slot of an instruction-following
reranker. Six rerankers were tested: Qwen3-Reranker-0.6B and 4B, zerank-2, FollowIR-7B, ctxl-rerank-v2-6B, and
bge-reranker-v2-gemma. The design is 2 x 2, with the profile in the instruction or not and a label line in the passage
or not. Each question is asked by two students from different programs, and each student's own passage is the gold. A
reranker that ignores the profile wins exactly one row of each pair or group, and this count is the blind baseline.

## Main results

| Track | Blind baseline | Profile in the slot | Gain bound (pp) | Label in the passage (five models) | Table | Script |
| --- | --- | --- | --- | --- | --- | --- |
| Extended paired, 480 question pairs | 480 of 960 | 476 to 494 | 4.40 | 691 to 929 | `p2_paired.json` | `10_rebuilt_and_extended/extended_paired_analysis.py` |
| Rebuilt broad, 26 groups | 26 of 55 | 25 to 28 | 12.50 | 33 to 48 | `r2_broad.json` | `10_rebuilt_and_extended/rebuilt_broad_analysis.py` |
| Original paired, 52 question pairs | 48 of 96 | 43 to 52 | 13.71 | 83 to 96 | `r1_cluster_paired.json` | `9_revision/revision_analysis.py --bagian paired` |

The gain bound is the largest gain that the 95 percent interval allows across the six models. On all three tracks, no
slot cell differs from the baseline after Holm correction. The two earlier broad samples are in
`r1_cluster_broad.json` and `r1_random_broad.json`.

Five lecturers who are not authors rated all 104 golds of the original paired track, all 55 golds of the rebuilt broad
track, and 120 sampled golds of the extended paired track. After disagreements were settled, they found no invalid
gold on the two paired tracks and one on the rebuilt broad track. Their ratings are in
`study/channel/results/human_ratings/` under rater codes, and `11_human_ratings/human_ratings_analysis.py` recomputes
Table 6 of the paper from them.

## Repository map

| Path | Contents |
| --- | --- |
| `quick_check.py` | all checks in one command |
| `verify_paper_numbers.py` | 284 numbers of the paper, each checked against a named table, record, or setting |
| `study/channel/scripts/` | the analysis, one folder for each step of the paper |
| `study/channel/results/` | result tables, per-row scores with their run manifests, and the lecturers' ratings |
| `study/channel/provenance/` | the paired-track questions and the workflow that wrote them |
| `study/retrieval/` | the DITS results on the original paired track and the benchmark composition |
| `study/_shared/` | config lock, corpus snapshot, book licenses and sources, and the names used in the runs |
| `system/` | the SentIO prototype: retrieval, the DITS scorer, the parser and chunker, and the benchmark generator |
| `protocol/` | the protocols that fixed the analyses in advance, with English translations of the Indonesian originals |

The per-row scores are in `raw/` (original paired track and first broad sample), `raw_r1/` (slot variants, label
factorial, vendor format, resampled broad sample), `raw_r2/` (rebuilt broad), and `raw_p2/` (extended paired). Most
files in the repository are these run outputs. To check the paper, start with `quick_check.py`.

Some code can only be read here. The benchmark generator and the bound calibration need the vector store of the
corpus. The scripts in `10_rebuilt_and_extended/` that built the two new tracks need the corpus, the judge server, or
the writer transcripts, so only the two analysis scripts of that folder run here.

## Verify the numbers

Python 3.11 or newer, no GPU. On Python 3.10, install `numpy<2` and `scipy<1.13` instead of the pinned versions.

```
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r study/requirements-check.txt
python quick_check.py
```

The check takes under a minute and should end with `quick check PASSED`. It compares every file with
`MANIFEST.sha256`, reruns the 25 analysis scripts that need only the files here, and compares every number in the 29
tables they write with the saved ones. Then it runs `verify_paper_numbers.py` on the rerun copy and checks the demo
pool. On Windows, keep the folder on a short path, since the longest path inside is 121 characters.

Tables whose inputs are not here are read as saved, for example the label factorial (Table 5) and the textual control
(row (c) of Table 4). The analysis scripts overwrite their tables, so run a single script in a copy of the repository.

## Run the rerankers

The scoring scripts need Linux, a CUDA GPU, and the model weights. We used one RTX 5080 with 16 GB.

```
pip install -r system/requirements.txt
pip install -r study/requirements-scoring.txt -r study/requirements-analysis.txt
```

Each scorer loads its model at the commit used in the study (`2_scoring/model_pins.py`). For a quick test,
`demo_small_pool.py` scores ten question pairs from public-domain and CC BY books and compares the result with the
released scores. The smallest model needs a few GB of GPU memory.

```
python study/channel/scripts/2_scoring/demo_small_pool.py --model Qwen3-0.6B
python study/channel/scripts/2_scoring/demo_small_pool.py --all
```

The full cells need the candidate pools as text. Because the text comes from 40 textbooks, the pools are kept here as
chunk ids. The pools of the extended paired and rebuilt broad tracks can be rebuilt as follows.

1. Get the PDFs listed in `study/_shared/book_sources.csv`, which gives each book's license, source, and sha256. Nine
   OpenStax titles have changed license or edition since then, so their CC BY printings may have to come from the
   Internet Archive.
2. Rebuild the corpus and the pool on Linux, with `EMBEDDING_MODEL_NAME` set to a local copy of BAAI/bge-m3 at commit
   5617a9f61b028005a4858fdac845db406aefb181. Every chunk is checked against its hash, and the script lists the books
   whose chunks do not match.

   ```
   pip install -r system/requirements.txt -r study/requirements-corpus.txt
   python study/channel/scripts/1_tracks/rebuild_corpus.py --books <pdf folder> --out corpus.jsonl
   python study/channel/scripts/0_common/rebuild_text_pool.py --track p2 --corpus corpus.jsonl --out pools_p2.jsonl
   ```

3. Run a cell with the command saved in its manifest (`perintah_persis` in
   `study/channel/results/raw_p2/scores_*.manifest.json`), with `--pools` set to the new file and `--out-dir` set to a
   folder of your own. `study/_shared/script_names.csv` gives the file here for each script name of the runs.

Use `--track r2` for the rebuilt broad track. Scores can move in the last decimals on other hardware, so compare
resolved counts and bounds rather than raw scores. The cells with a label also need the program and semester of each
chunk from the study's corpus file, which is not here, so the rebuild covers only the cells without a label.

## Data

Not included: the textbooks and the pools in text form, the 2,776-query benchmark and its qrels, the writer transcripts
of the extended paired track, the per-item scores of the textual control, the per-row DITS scores of the original
paired track and of the two earlier broad samples, the questions of the two earlier broad samples, the model weights,
and the FollowIR collections. The rating workbooks are not included because they name the raters.

The protocols are in `protocol/`, and its README lists them. The run manifests and `protocol_commits.json` record
which protocol each pre-registered analysis followed. The questions, the partner passages, and the judge
verdicts were produced with language models, and the run manifests and `study/channel/provenance/` name the model or
workflow of each step. The judge labels are not a gold standard.

The study ran in Indonesian, so table keys, command-line flags, status words (`LOLOS` pass, `GAGAL` fail,
`PERINGATAN` warning), prompts, questions, and some notes are in Indonesian. The rerankers' instruction is in English,
except in the control that tests an Indonesian one. The run records keep the names and paths of the runs, where
`result/paper_q2/` is `study/channel/`, `result/novelty_1_dits/` is `study/retrieval/`, and `src/ai/` is `system/`.
`study/_shared/script_names.csv` and `study/_shared/table_names.csv` give the file here for each of these names.

Some table notes and records differ from the paper or from the code here. `study/channel/results/tables/NOTES.md`
lists them.

## License

Code is MIT (`LICENSE`). Results and documents are CC BY 4.0. Three functions adapted from MTEB and one from the
bge-reranker-v2-gemma model card keep Apache 2.0, and the ctxl-rerank-v2 scoring path follows a model card under CC
BY-NC-SA 4.0. Book excerpts keep their own licenses. `NOTICE.md` has the details.

## Citation

See `CITATION.cff`.
