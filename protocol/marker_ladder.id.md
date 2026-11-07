# Praregistrasi: tangga jarak inferensi pada kanal dokumen

Ditulis dan dikomit SEBELUM satu sel pun dijalankan. Aturan analisis di bawah
mengikat, termasuk prediksinya, supaya hasil apa pun tidak bisa ditafsir ulang
belakangan.

Tanggal: 20 September 2026
Praregistrasi induk: PRAREGISTRASI_LINTAS_ARSITEKTUR.md dan adendum 2, 3, 4
Config lock: 0a7b2e4de856

## 1. Pertanyaan

Naskah melaporkan bahwa kanal dokumen bekerja, yaitu profil di slot ditambah
label di teks kandidat membawa lima dari enam reranker dari 48 ke 82 sampai 96
dari 96. Satu bantahan belum terjawab. Label yang disuntikkan berbunyi
`[Source: STAT degree program textbook, semester 2]` sedangkan instruksinya
berbunyi `a STAT student in semester 2`. Keduanya memuat token yang sama persis.
Jadi hasil positif itu bisa saja pencocokan string, bukan pencocokan makna.

Pertanyaannya: seberapa jauh jarak inferensi yang masih bisa dijembatani
reranker antara profil penanya dan penanda di dokumen?

## 2. Kenapa tangganya dijalankan pada label PROGRAM-SAJA

Kontrol satu-bidang yang sudah beku (label_satu_field.json dan
papan_saluran_paired_arsitektur.json) menunjukkan label semester-saja hampir
menyamai label penuh, 78/84/93/91/96/51 lawan 82/84/95/90/96/51. Artinya
semester mendominasi. Tangga yang dijalankan pada label penuh akan datar di
dekat plafon dan tidak mengukur apa pun.

Label program-saja punya jangkauan nyata, 62/74/76/52/83/49 lawan kunci buta 48.
Di situlah efek bentuk permukaan penanda program bisa terlihat.

## 3. Empat anak tangga

Semua memakai track berpasangan 96 baris, enam model, dan instruksi profil
Inggris yang sudah beku. Yang berubah HANYA bentuk penanda program di teks
kandidat. Semester TIDAK disertakan di satu anak tangga pun.

| anak tangga | label yang disuntikkan | penanda program | status |
| --- | --- | --- | --- |
| a. kode literal | `[Source: STAT degree program textbook]` | kode, sama persis dengan instruksi | SUDAH DIUKUR, baris "Program only" Tabel 6 |
| b. nama prodi | `[Source: statistika degree program textbook]` | nama domain bahasa Indonesia | baru |
| c. header alami | `Program studi statistika.` | nama domain, tanpa kerangka Source | baru |
| d. kode opak | `[Source: PRG-09 degree program textbook]` | kode institusi tanpa isi semantik | baru |

Nama prodi pada anak tangga b dan c diambil dari `_DOMAIN_LABEL` di
`system/app/api/routes/query.py`, pemetaan yang sudah dipakai sistem terpasang
untuk mekanisme perluasan kueri pra-OOD. Pemetaan itu TIDAK dibuat untuk
eksperimen ini, jadi tidak ada kebebasan memilih kata yang menguntungkan.

Kode opak pada anak tangga d dibentuk `PRG-<nn>` dengan nn urut menaik menurut
urutan alfabet kode prodi pada pemetaan itu, tetap, dan tercatat di skrip. Contoh di
tabel memakai statistika yang jatuh pada PRG-09. Kode itu tidak muncul
di korpus mana pun.

## 4. Prediksi, ditulis sebelum menjalankan

Ditulis supaya hasil tidak bisa dibingkai ulang setelah terlihat.

1. Anak tangga d jatuh ke kunci buta 48 pada keenam model, selisih paling
   banyak dua baris. Kalau tidak, berarti ada kebocoran yang belum diketahui
   dan seluruh tangga ini tidak bisa ditafsirkan.
2. Anak tangga b turun dari anak tangga a pada mayoritas model. Besarnya tidak
   diprediksi.
3. Bge-reranker-v2-gemma tetap di sekitar kunci pada keempat anak tangga,
   karena sel mana pun tidak pernah menggerakkannya.

## 5. Aturan tafsir, mengikat

- Anak tangga b dibandingkan dengan a memakai McNemar eksak berpasangan pada
  96 baris yang sama, koreksi Holm dalam keluarga enam model.
- Kalau b jatuh ke dalam dua baris dari kunci pada mayoritas model, naskah
  WAJIB menyatakan bahwa kanal dokumen bekerja lewat kecocokan leksikal, dan
  kontribusi pertama ditulis ulang apa adanya. Hasil ini tidak boleh
  disembunyikan atau dilaporkan hanya di apendiks.
- Kalau b bertahan dalam jarak yang tidak berbeda signifikan dari a, naskah
  boleh menyatakan kanal dokumen membawa pencocokan makna, bukan string.
- Kalau b turun signifikan tetapi tetap jauh di atas kunci, naskah menyatakan
  keduanya, yaitu ada komponen leksikal DAN komponen makna, dengan angkanya.
- Anak tangga c dinyatakan EKSPLORATIF karena mengubah dua hal sekaligus,
  bentuk penanda dan kerangka kalimatnya, jadi dia tidak dipakai untuk
  menyimpulkan apa pun tentang leksikal lawan makna. Nilainya praktis: kalau c
  bekerja, rekomendasi naskah berubah dari menyuntikkan label buatan menjadi
  menulis header program studi di dalam chunk.
- Nol sel beku ditimpa. Nol angka lama diubah.

## 6. Artefak

Skrip: study/channel/scripts/score_architectures.py, sel baru `label_nama`,
`label_alami`, `label_opak`. Agregator:
study/channel/scripts/inference_ladder.py. Tabel:
study/channel/results/tables/tangga_inferensi.json.
