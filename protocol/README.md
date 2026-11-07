# Protocols

Most of these files fixed an analysis before its scores existed. Most state when they were written and that no score
of their analysis existed yet, and parts added later say so. `revision_round1_vendor_compliance.md` records a check of
how each model was called. They are working documents, written for the project's own use. They are terse and point
to internal notes of that time, such as a QC report or a numbered limitation.

The channel protocols and the label-map protocol were written in Indonesian. Each original ends in `.id.md`, and
the file with the same name ending in `.md` is its English translation. The two `revision_round1` files were
written in English.

Each channel protocol was committed before the first cell it covers.
`study/channel/results/tables/protocol_commits.json` lists each commit, the first cell, and the minutes between
them. The times come from the authors' own machine, so they show the order of events but do not prove it. Its note that a commit was
not yet on the remote holds only for the rows whose `sudah_di_remote` is false. The cells of
`cross_architecture` did not run in the order of its section 9, which changes no score. The
label-map protocol has no commit record. The Qwen3 channel cells, the own-harness zerank-2 cells, and the
worked-example control were scored before any of these documents, so they are post hoc.

| File | Fixed in advance |
| --- | --- |
| `cross_architecture` | three more rerankers beside Qwen3 and zerank-2, how each is called, five cells per new model, a test of the zerank-2 scorer, and the bound |
| `cross_architecture_addendum2` | three more model families on the broad track, one-field labels, News21 and Robust04, and a replay of zerank-2 on the vendor's path |
| `cross_architecture_addendum3` | the vendor-style instruction cells |
| `cross_architecture_addendum4` | the Indonesian instruction cells |
| `scorer_equivalence` | a test of the project's zerank-2 scorer against the vendor's |
| `surface_ladder` | program-only labels in English and Indonesian, and the semester labels |
| `marker_ladder` | program-only labels from the code to an opaque code, with a control |
| `label_map` | the rescoring of every cell that carries a label |
| `revision_round1` | the follow-up analyses of this paper |
| `revision_round1_vendor_compliance` | how each reranker was called, checked against its model card |

`revision_round1.md` is the pre-registration of the follow-up analyses as committed before their first new cell.

Scripts, run manifests, and tables cite these documents by their names in the authors' repository.

| Cited as | File here |
| --- | --- |
| `PRAREGISTRASI_LINTAS_ARSITEKTUR.md` | `cross_architecture.id.md` |
| `PRAREGISTRASI_LINTAS_ARSITEKTUR_ADENDUM2.md` | `cross_architecture_addendum2.id.md` |
| `PRAREGISTRASI_LINTAS_ARSITEKTUR_ADENDUM3.md` | `cross_architecture_addendum3.id.md` |
| `PRAREGISTRASI_LINTAS_ARSITEKTUR_ADENDUM4.md` | `cross_architecture_addendum4.id.md` |
| `PRAREGISTRASI_ADENDUM_KESETARAAN.md` | `scorer_equivalence.id.md` |
| `PRAREGISTRASI_TANGGA_ADENDUM5_PERMUKAAN.md` | `surface_ladder.id.md` |
| `PRAREGISTRASI_TANGGA_JARAK_INFERENSI.md` | `marker_ladder.id.md` |
| `PRAREGISTRASI_PETA_LABEL.md` | `label_map.id.md` |
| `PRAREGISTRASI_R1_IJIES.md` | `revision_round1.md`, as committed before the first new cell |

`protocol_commits.json` also lists five other pre-registrations of the project, which are not here. Table names and
section names in the protocols date from when each was written, and names such as "Appendix B" point to a
draft of an earlier paper. In `cross_architecture_addendum4.md` and `marker_ladder.md`, the translation gives a
table's name where the original gives its number in that draft. In the seven Indonesian channel protocols, script
names and paths were changed to the names used in this repository, and a few of those files have since moved or are
not here. Nothing else in these originals differs from the authors' copies.
