# Provenance of the paired-track questions

`paired_set_generation_workflow.js` wrote the 52 question pairs of the paired track (`paired_queries.jsonl` in
this folder, 104 rows). It ran as a Claude Code workflow in three phases.

First, language-model subagents take batches of the cross-book chunk pairs that `system/scripts/build_paired_set.py`
picked by embedding similarity alone (`topic_pairs.jsonl`, not included). For each pair they write one neutral question that both
the CSCI passage (Lane, public domain) and the STAT passage (OpenStax Introductory Statistics 2e, CC BY 4.0
printing) answer. The script offered the first 150 rows in six batches of 25 (lines 71 to 78), and because
`build_paired_set.py` sorts by descending cosine before cutting at 400, these are the 150 most similar of the 400
pairs. Second, other subagents check each question for lexical leakage toward one book, answerability by both
passages, text quality, and plausibility, and reject it when in doubt. Third, each accepted question becomes two
rows, one under the CSCI profile and one under the STAT profile, each with its own gold.

Language models wrote and checked the questions, and no person took part in the run. The first author
spot-checked the gold passages. The run produced 115 candidates and 52 accepted pairs. A batch whose agent returned
nothing would have been dropped without a count (`filter(Boolean)` at lines 123 and 163), so this folder alone
cannot confirm the 115. The run's journal accounts for all 150 topic pairs: 115 questions and 35 declines with a
reason (`study/channel/results/tables/r1_paired_provenance.json`).

The database password in the script's two `psql` commands is replaced by `<redacted>`, and the local path prefix by
`<workstation>`.

`extended_paired_writer_workflow.js` wrote and screened the questions of the extended paired track in six rounds.
Its name in the runs was `q2_p2_alur_tulis.js`, the name `extended_paired_store.py` still looks for. Before each
round, `extended_paired_topic_pairs.py --tahap ronde --ronde N` wrote batches of up to 25 topic pairs. One writer per
batch wrote one neutral question per pair, and a screener that had not written that batch applied the four tests
above. `extended_paired_store.py` stored the results of each round. `study/channel/results/tables/p2_population_public.json`
lists every topic pair sent to a writer, with its question and the writer's reason.
