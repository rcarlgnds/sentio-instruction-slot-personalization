# Notes on the tables and records

Some notes in the tables and run records differ from the paper or from the code in this repository. The list below
gives each case. Paths start at the root of the repository. `verify_paper_numbers.py` names the file and key behind
every number it checks.

## Tables and the paper

- A few notes inside the tables describe the paper and do not match its text, as in `followir_calibration.json`,
  `injection_probe_*.json`, and `channel_tests_full.json`.
- For zerank-2, the counts in `channel_board_paired.json`, `channel_board_xpair.json`, and `equivalence_bounds.json`,
  including those under `verifikasi_naskah`, come from our own scorer. The paper prints the counts of the scoring path
  in the model card: 48 of 96 for the paired slot cell, and 597, 819, and 705 for the three cells of the first broad
  sample. `zerank2_official_replay.json` holds both, and they differ by one row at most.
- `equivalence_bounds.json` takes three models and the wider side of each interval. The paper's bounds for the original
  paired track and the first broad sample, 13.71 and 0.89, are the gain side over six models, as in
  `channel_tests_full.json`.
- The `what` field of `p2_paired.json` says that both passages were judged full answers. The groups were built with
  the rule in `sample_design.rules.group`, which also accepts two partial answers, and 436 of the 480 groups are
  partial. The 43 groups judged full under both wordings form the strict subset.
- Table 3 prints the Qwen3-0.6B slot cell of the original paired track as first scored, 49 of 96.
  `r1_slot_orthogonal.json` and `r1_label_factorial.json` compare with the same cell scored again, which resolves 50
  because one row flipped (`r1_gate_g0.json`).
- `r1_uniform_semester.json` is labeled exploratory, although `protocol/revision_round1.md` plans this subset. The
  table was built before that file was committed, which section 11.1 (h) gives as the reason.
- Most bounds are stored with three decimals, and the paper rounds once from the unrounded value, so 6.425 in
  `r2_broad.json` is 6.42 in Table 3. In `human_ratings.json`, the share e comes from an upper limit rounded to four
  decimals, so a recomputation from the unrounded limit can move a cell of Table 6 by 0.01.
- `marker_ladder.json`, `surface_ladder.json`, `surface_ladder_p1_p2.json`, and `mcnemar_paired.json` compute Holm p
  from exact p values rounded to six decimals, so a Holm p of 0.0 is an exact p that rounded to zero.

## Controls and screening runs

- The run manifest of the worked-example control, `control_textual.manifest.json`, says in Indonesian that its four
  title rules were fixed before any score was read. The cells had been scored before the rules were written, so the
  paper and `control_textual.json` treat the 36 filtered pairs as exploratory. The two rule texts in that table are
  English versions of these rules without the timing clause, and `fakta_struktural_default` reports a check that the
  script does not run. `study/channel/scripts/1_tracks/build_control_track.py` built the pool of the control, 58
  topics and 116 rows.
- The format sweep tables, `injection_probe_*.json`, call the sweep a screening run whose numbers are not results of
  the paper. The paper reports one result from it: every format that leaves the passage unchanged stays within two rows
  of the baseline. Their two formats with a label, K4 and K5, used the label map from before `protocol/label_map.md`.
- The surface ladder protocol expected its negative control s3 to land on the blind baseline of 48 on all six models.
  In `surface_ladder.json`, ctxl-6B resolves 56 rows on it. The reading rules of that protocol do not depend on s3.

## Pools and samples

- The pool manifests of the second broad sample and the rebuilt broad track give `pool_size` 71, while their pools hold
  68 to 70 candidates. The rule takes the chunks nearest the first gold, 71 less the number of golds, and then adds the
  golds that are not among them. In the manifest of the rebuilt broad track, the `exclusions` block also counts
  `groupable`, the 26 groups that were kept.
- The pools without a rebuild step have a hash anchor for each chunk in `study/channel/results/raw_r1/pools/`, except
  the pool of the worked-example control. On Windows, the PDF parser of the rebuild cannot reopen its temporary file
  and falls back to a flat parse.

## Benchmark and system

- `benchmark_provenance.json` names `qc.py` and `templates.py` in `system/app/benchmark/rads_synthesis/` as the gate
  and the prompt template. The files here are the last versions, gate 2 and template rads-t1.4. The code of gate 1,
  which admitted the first 1,412 queries, and the templates rads-t1.2 and rads-t1.3 are not here.
- `study/_shared/split_manifest.json` records a split made in two rounds, 1,412 queries with seed 20260729 and 1,364
  with seed 20260803. `finalize_splits()` in `system/app/benchmark/rads_synthesis/pipeline.py` splits all queries in
  one round with seed 42, so it does not give the released split.
- `study/_shared/env_versions.json` records the setup of the SentIO system. The judge candidate it names, phi4, is not
  used in this paper, and the test it cites, `s0_verify_ingest.py`, is not here.
- The DITS scorer read the program relations and the course prerequisites from a database that is not here.
  `system/data/department_relations.json` records the program relations it used, and `system/calibration.json` the
  similarity bounds. The relations file names each program by its code, while the database used program ids, so the
  `fingerprint` in `study/_shared/config_lock.json` cannot be recomputed from the file.
- `study/channel/scripts/0_common/posthoc_holdout_metrics.py` and the held-out and train/dev cells of
  `zerank2_official_replay.json` belong to the project's earlier retrieval study, and the paper uses none of them. Some
  settings in `system/app/config.py`, such as those for Redis and MinIO, are read by no code here.

## Run records

- The manifests list the sha256 of the files each run read or wrote. For this release we removed most code comments
  and cleaned or translated the notes in some tables, data files, and manifests. Every recorded sha256 of a data file
  or table is the one of the file as it ships here. The recorded sha256 of a script is the one of the code as it ran,
  so it differs where comments were removed, as for `system/app/config.py` in `deployment_envelope.json`. No number
  changed.
- `system/app/fingerprint.py` no longer gives the `config_version` in `study/_shared/config_lock.json`, because it
  hashes the cleaned modules in `system/app/generation/` and a router centroid file that is not here.
- In the score manifests, `status_arm` marks each cell as post hoc, outside the pre-registered family of eight. That
  family is the eight comparisons pre-registered for the project's earlier retrieval study, and the field only says
  that a cell's tests stay out of its Holm correction.
- Some paths in the run records, two scripts, and one protocol note begin with `<workstation>` or `<home>`, which stand
  for local folders on the study's machine.
- The FollowIR scripts load their collections from the Hugging Face Hub without a fixed revision, and the runs did not
  record which revision they read.
- Some tables cite sections of the protocols, such as 11.1 c. Most tables have an `_apa_ini` or `what` field that says
  what they hold, and most have a manifest in `study/channel/results/raw*/` that names the script that wrote them.
- SentIO answers Indonesian students, so its prompts (`system/app/generation/prompts.py`), the example questions of
  its router (`system/app/routing/semantic_router.py`), and its calibration queries
  (`system/data/calibration_queries.json`) are in Indonesian, kept exactly as they ran.
