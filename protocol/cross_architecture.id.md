# Praregistrasi uji lintas arsitektur, kesetaraan skorer zerank-2, dan sel kueri telanjang

Ditulis 17 Sep 2026, SEBELUM satu pun sel baru dijalankan. Saat berkas ini
ditulis, yang ada di disk untuk keluarga baru hanya bobot model di cache
Hugging Face; nol berkas skor, nol tabel. Tujuan berkas ini mengunci model,
sel, kontrak inferensi, metrik, uji, dan aturan pelaporan supaya pemilihan yang
menguntungkan tidak mungkin dilakukan belakangan.

Motivasi (QC 17 Sep, Lampiran F): null "slot instruksi tidak membawa
personalisasi" saat ini bertumpu pada tiga model satu keturunan Qwen3
(Limitasi 13). Uji ini menambah tiga keturunan lain. Hasil ke arah mana pun
dilaporkan.

## 1. Populasi dan instrumen

- Track berpasangan: `study/retrieval/results/raw/pools_paired.jsonl`,
  104 baris, 96 komparabel (gold dan counterpart sama-sama ada di pool).
  sha256 pool dicatat di tiap manifest. TIDAK ADA pool lain.
- Kontrol positif eksternal: FollowIR Core17 (`mteb/Core17InstructionRetrieval`),
  40 kueri, 1.000 kandidat per kueri, p-MRR dengan implementasi MTEB verbatim
  yang sudah dipakai `followir.py`. Protokol identik dengan sel Qwen3 dan
  zerank-2 yang sudah beku.
- Segel holdout: gerbang `gerbang_segel` yang sama; pool paired tidak memuat
  query_id test_holdout.

## 2. Model, dan kenapa ketiganya

| keluarga | model | arsitektur (config.json cache) | keturunan |
|---|---|---|---|
| gemma | `BAAI/bge-reranker-v2-gemma` | GemmaForCausalLM | Gemma (Google) |
| ctxl | `ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b` | MistralForCausalLM, 20 lapis, hidden 5120 | Mistral |
| followir | `jhu-clsp/FollowIR-7B` | MistralForCausalLM, 32 lapis, hidden 4096 | Mistral, di-tune khusus patuh instruksi retrieval |

Arsitektur dibaca langsung dari `config.json` snapshot di cache, bukan dari
nama. `ctxl-rerank-v2-instruct-multilingual-1b` dan `-2b` SENGAJA tidak
dipakai karena config-nya `Qwen3ForCausalLM`. Tidak ada kuantisasi: studi kappa
proyek ini sudah mengukur kuantisasi menggeser vonis sebesar beda keluarga.

## 3. Kontrak inferensi, disalin dari README snapshot yang sama dengan bobotnya

Ketiga kontrak disimpan verbatim ke `kontrak_inferensi` di manifest tiap sel.

gemma. Fungsi `get_inputs()` README disalin verbatim (prefiks `A: `/`B: `,
bos manual, pemisah `\n`, prompt di ekor, `prepare_for_model` truncation
`only_second`, `pad_to_multiple_of=8`). Skor = logit token `Yes` pada posisi
akhir, mentah, tanpa softmax. Slot instruksi resmi = argumen `prompt`.
Prompt bawaan vendor verbatim: "Given a query A and a passage B, determine
whether the passage contains an answer to the query by providing a prediction
of either 'Yes' or 'No'."

ctxl. Chat template resmi (`chat_template.jinja` snapshot): peran `system`
(instruksi), `query`, `document`, bos dari template, akhiran ` ??`. Skor =
logit mentah token id 0 pada posisi akhir (`1_LogitScore/config.json`:
true_token_id 0, false_token_id null), persis `predict()` vendor. Slot
instruksi resmi = pesan `system`, yaitu argumen `prompt=` di `predict()`.
Bawaan vendor = TANPA instruksi ("The prompt argument is optional").

followir. Template README verbatim (`<s> [INST] You are an expert Google
searcher ... Query: {query}\nDocument: {text}\nRelevant (only output one
word, either "true" or "false"): [/INST] `), padding kiri, pad = eos, skor =
softmax(logit false, logit true)[true]. README mentokenisasi string yang
sudah memuat `<s>` dengan `add_special_tokens` bawaan, sehingga BOS ganda;
itu kontrak vendor dan ditiru apa adanya. Slot instruksi = ditempel ke teks
query ("Query: {query} {instruksi}"), konvensi FollowIR sendiri ("input of
query + instruction inside the template").

## 4. Sel yang dijalankan, semuanya dilaporkan

Instruksi profil verbatim sama dengan seluruh sel Qwen3 dan zerank-2
(`instruksi_profil` di `score_arms.py`): "Given a query from a {dept}
student in semester {sem}, retrieve passages from that student's own study
program that answer the query at a level appropriate for semester {sem}".
Label teks dokumen verbatim sama (`label_sumber`, `label_sumber_salah`),
ditempel SESUDAH teks dipotong ke 4.000 karakter, persis sel beku.

Per model baru, lima sel inti T1:

| sel | slot instruksi | teks dokumen |
|---|---|---|
| none | bawaan vendor (gemma: prompt bawaan; ctxl: tanpa system; followir: query polos) | polos |
| slot | profil di slot resmi keluarga itu | polos |
| text | bawaan vendor | + label sumber benar |
| both | profil di slot | + label sumber benar |
| wrong | profil di slot | + label sumber prodi SALAH, semester tetap benar |

Bagaimana profil masuk slot per keluarga: gemma prompt = prompt bawaan + " " +
instruksi profil (prompt bawaan memuat kerangka tugas Yes/No yang skornya
bergantung padanya, jadi ditambah, bukan diganti); ctxl system = instruksi
profil; followir query = query + " " + instruksi profil.

Sel sekunder, sudah ditetapkan sekarang, bukan sesudah melihat hasil:
- ctxl `none_generic`: system = instruksi generik `TASK_DEFAULT` ("Given a web
  search query, retrieve relevant passages that answer the query"), untuk
  memeriksa apakah "none" vendor (tanpa system) dan konvensi paper (instruksi
  generik) memberi kunci yang sama. Dilaporkan berdampingan.
- zerank-2 `bare`: query telanjang tanpa prefiks `TASK_DEFAULT`, keluarga dan
  skorer beku `SkorerZerank2` tanpa perubahan. Menutup pertanyaan reviewer
  soal sel "None" zerank-2 yang selama ini memakai prefiks generik.

T2, per model baru: FollowIR Core17, protokol identik `followir.py`.

## 5. Parameter teknis, ditetapkan sebelum jalan

- dtype bfloat16 untuk ketiga model (ctxl dan followir dirilis bf16; gemma
  README tanpa dtype, bf16 dipilih karena sama dengan zerank-2). Dicatat di
  manifest.
- Panjang maksimum 1.536 token, sama dengan zerank-2. Pemotongan diterapkan
  pada TOKEN DOKUMEN supaya kerangka template (instruksi, query, ekor
  template) tidak pernah terpotong; jumlah dokumen yang terpotong dicatat per
  sel (`n_dokumen_terpotong`). Teks dokumen dipotong ke 4.000 karakter lebih
  dulu, sama dengan seluruh sel beku.
- `logits_to_keep=1`: hanya logit posisi akhir yang dihitung. Ini menghemat
  memori, tidak mengubah nilai logit posisi akhir.
- Batch dicari otomatis dari besar ke kecil sampai sisa VRAM tetap di atas
  batas yang ditetapkan; nilai batch akhir dicatat. Skor tidak bergantung pada
  batch (tiap dokumen diskor dengan padding kiri dan attention mask).
- Batas sisa VRAM untuk run ini 500 MiB, atas perintah pemilik mesin 17 Sep,
  menggantikan lantai 1.800 MiB proyek KHUSUS untuk run ini. Penjaga runtime
  membatalkan sel (bukan desktop) bila sisa jatuh di bawah batas.
- ctxl-6b (14.240 MiB bobot) tidak muat penuh dengan batas itu, jadi dimuat
  dengan `accelerate` `device_map` dan cap memori GPU; lapisan yang tidak muat
  disimpan di RAM dan dialirkan ke GPU saat forward. Aritmetika tidak
  berubah; hanya lokasi penyimpanan bobot. Jumlah modul yang di-offload
  dicatat di manifest.
- Satu model di GPU pada satu waktu, lock berkas `.gpu.lock` yang sama dengan
  seluruh sel sebelumnya.
- Keluaran per sel berformat IDENTIK dengan sel beku (`query_id, gold,
  counterpart, ranking, scores, latency_s`), urutan `(-skor, id)`, plus
  manifest dengan sha256 pool dan keluaran, `config_version` lock
  `0a7b2e4de856`, durasi, VRAM puncak, batch, dan jumlah pemotongan.

## 6. Metrik dan uji, ditetapkan sebelum angka ada

- Metrik: `metrics.metrik_paired` dan `mcnemar.hasil_per_item`, TANPA
  perubahan. Gerbang replikasi `metrics.gerbang_replikasi()` wajib lolos
  sebelum satu sel pun dibaca (mereproduksi 0,5000 / 0,5104 / 0,5000 / 0,5208
  dari sel C3/C4 terbit).
- Kunci buta struktural 48/96 berlaku sama: sel tanpa profil TERKUNCI di 48
  secara aljabar dan TIDAK diuji terhadap null 0,5.
- Kontras McNemar exact per model: none vs slot; slot vs both; text vs both;
  wrong vs both. Koreksi Holm dalam keluarga E = tiga kontras none-vs-slot
  (m = 3), terpisah dari keluarga terbit supaya p_holm yang sudah terbit tidak
  berubah.
- Batas ekuivalensi efek slot per model: fungsi `analisis()` dari
  `equivalence_bounds.py` (selang yang lebih lebar dari Wald dan eksak
  Clopper-Pearson bersyarat), dilaporkan sebagai selang, bukan diuji terhadap
  margin pilihan penulis.
- Kontrol positif: p-MRR Core17 per model. Model dengan p-MRR di bawah nol
  tetap dilaporkan; ia hanya mengubah bacaan (model tidak patuh instruksi di
  instrumen luar sekalipun), bukan dibuang.
- ATURAN PELAPORAN: seluruh sel, seluruh kontras, seluruh model dilaporkan
  apa adanya. Model yang bergerak keluar dari kunci lewat slot adalah temuan
  yang mengubah klaim naskah, bukan alasan mengulang atau membuang sel. Sel
  yang gagal atau dibatalkan dilaporkan dengan alasannya. Tidak ada sel yang
  diulang dengan setelan berbeda kecuali batch (yang tidak mengubah skor) dan
  itu dicatat.
- Tabel baru ditulis ke berkas BARU `papan_saluran_paired_arsitektur.json`
  dan `followir_Core17_<keluarga>_<model>.json`. Tabel beku
  `papan_saluran_paired.json` tidak disentuh.

## 7. Uji kesetaraan skorer zerank-2

Klaim yang diuji: skor `SkorerZerank2` proyek (reimplementasi lewat
`AutoModelForCausalLM` + chat template + logit token true) setara dengan
`sentence_transformers.CrossEncoder.predict()` resmi vendor.

- Pustaka: sentence-transformers 6.0.1 dipasang ke overlay terpisah
  `system/.st-libs` (`--no-deps --target`); venv dan overlay transformers
  4.55 tidak berubah. Versi dicatat.
- 100 pasangan: iterasi `pools_paired.jsonl` urut berkas, ambil pasangan
  (teks_query, teks_dokumen) dalam urutan kandidat pool sampai genap 100.
  teks_query diformat persis sel `default` beku: `TASK_DEFAULT + "\n\nQuery: "
  + query`; dokumen dipotong 4.000 karakter. Tidak ada pemilihan pasangan lain.
- Dua jalur pada input identik: (a) `SkorerZerank2.skor` bf16 max_len 1536,
  (b) `CrossEncoder("zeroentropy/zerank-2", model_kwargs dtype bf16,
  max_length 1536).predict()`.
- Yang dilaporkan: Spearman rho seluruh 100 pasangan; per query, jumlah
  inversi urutan antar jalur (pasangan kandidat yang urutannya berbeda);
  selisih absolut maksimum dan median; jumlah pasangan dengan selisih di atas
  toleransi bf16 0,05. Ditambah pemeriksaan bahwa jalur (a) mereproduksi skor
  di berkas beku `scores_zerank2_zerank-2_paired_default.jsonl` untuk
  pasangan yang sama (selisih maksimum dicatat).
- Kriteria lolos, ditetapkan sekarang: nol inversi di seluruh query DAN
  selisih absolut maksimum <= 0,05. Bila tidak lolos, hasilnya tetap
  dilaporkan, dan konsekuensinya (skor ulang sel zerank-2) diputuskan
  sesudahnya secara terbuka, bukan disembunyikan.

## 8. Yang TIDAK dijalankan di run ini

T3 (replikasi xpair 1.621 baris untuk keluarga baru) tidak dijalankan;
alasannya dicatat di QC 17 Sep Lampiran F.5. Bila suatu saat dijalankan,
ia memakai skrip dan sel yang sama tanpa perubahan setelan.

Kontrol profile-shuffle pada benchmark utama akan mendapat adendum
praregistrasi terpisah SEBELUM dijalankan; ia memakai harness evaluasi yang
sama dengan arm B beku dan wajib lolos gerbang replikasi nDCG@5 0,8113 pada
profil tak diacak lebih dulu.

## 9. Urutan eksekusi dan pencatatan waktu

Urutan: uji kesetaraan zerank-2, sel bare zerank-2, T1 gemma, T1 followir,
T1 ctxl, T2 gemma, T2 followir, T2 ctxl. Setiap sel mencatat durasi dan VRAM
puncak; total waktu GPU dilaporkan dari manifest, bukan dari perkiraan.
Status berjalan ditulis ke `results/raw/arsitektur_status.json` dan dibaca
`scripts/architecture_status.py`.
