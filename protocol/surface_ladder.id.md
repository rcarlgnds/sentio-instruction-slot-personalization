# Praregistrasi adendum 5: memisahkan bentuk permukaan dari penyeberangan bahasa

Ditulis dan dikomit SEBELUM satu sel pun dijalankan. Aturan tafsir di bawah
mengikat, termasuk prediksinya.

Tanggal: 20 September 2026
Induk: PRAREGISTRASI_TANGGA_JARAK_INFERENSI.md dan KEPATUHAN_PRAREGISTRASI_TANGGA.md
Config lock: 0a7b2e4de856

## 1. Cacat yang diperbaiki

Tangga pertama mengganti kode prodi di teks kandidat dengan nama prodi
berbahasa Indonesia, sementara instruksi profil tetap berbahasa Inggris dan
tetap memakai kode. Anak tangga itu karena itu mengubah TIGA hal sekaligus,
yaitu kode menjadi nama, Inggris menjadi Indonesia, dan model dituntut tahu
bahwa CSCI adalah singkatan institusional untuk ilmu komputer. Keruntuhannya
tidak bisa diatribusikan ke salah satu dari ketiganya.

Naskah sudah menyatakan keterbatasan itu, tetapi aturan praktis di kesimpulan
tetap ditulis seolah umum. Adendum ini mengukur yang mana yang benar.

## 2. Nama prodi bahasa Inggris, dan kenapa pilihannya tidak bebas

Pemetaan NAMA_PRODI di skrip diambil dari kode produksi dan itulah yang
melindungi tangga pertama dari tuduhan memilih kata yang menguntungkan.
Repositori TIDAK memuat pemetaan kode ke nama Inggris, jadi pemetaan di bawah
DITULIS untuk eksperimen ini, dan itu wajib diakui.

Yang membatasi kebebasannya: setiap kode memang singkatan dari nama Inggrisnya,
jadi namanya tidak dipilih melainkan dibaca dari kodenya sendiri.

| kode | nama Inggris | kode adalah awalan namanya |
| --- | --- | --- |
| ACCT | Accounting | ya |
| BANA | Business Analytics | tidak |
| CHIN | Chinese Literature | ya |
| CSCI | Computer Science | tidak |
| DTSC | Data Science | tidak |
| ISYS | Information Systems | tidak |
| MGMT | Management | tidak |
| MKTG | Marketing | tidak |
| STAT | Statistics | ya |

Track berpasangan memakai CSCI dan STAT, jadi lagi lagi satu prodi berbagi
awalan dan satu tidak. Pemecahan stratum yang sama seperti
tangga_substring.json WAJIB dijalankan lagi, dan hasilnya dilaporkan.

## 3. Anak tangga baru

### 3a. Bidang program, instruksi memakai NAMA Inggris

Instruksi profil diubah satu kali saja, dari `a STAT student in semester 2`
menjadi `a Statistics student in semester 2`. Label di teks kandidat lalu
divariasikan, dan HANYA labelnya yang berubah antar sel.

| sel | label yang disuntikkan | yang diuji |
| --- | --- | --- |
| p1 | `[Source: Statistics degree program textbook]` | permukaan sama persis dengan instruksi |
| p2 | `[Source: statistika degree program textbook]` | fakta sama, bahasa beda |
| p3 | `[Source: STAT degree program textbook]` | instruksi menyebut nama, dokumen menyebut kodenya |

Jangkar yang sudah beku: instruksi kode plus label kode, yaitu anak tangga a
tangga pertama, 62 / 74 / 76 / 52 / 83 / 49 lawan kunci buta 48.

### 3b. Bidang semester, instruksi TIDAK diubah

Instruksi tetap `semester 2`. Kerangka label dijaga konstan dan hanya nilainya
berubah, supaya tepat satu hal bervariasi.

| sel | label yang disuntikkan | yang diuji |
| --- | --- | --- |
| s1 | `[Source: degree program textbook, semester 2]` | SUDAH DIUKUR, kontrol semester-saja, 78 / 84 / 93 / 91 / 96 / 51 |
| s2 | `[Source: degree program textbook, semester two]` | angka jadi kata Inggris |
| s3 | `[Source: degree program textbook, semester T-B]` | kode termin opak, KONTROL NEGATIF |

Bahasa TIDAK ikut divariasikan di bidang semester, karena kata semester sama
dalam bahasa Indonesia dan bahasa Inggris sehingga tidak ada kontras bahasa
yang bisa dibentuk. Kode opak memakai huruf, bukan angka, supaya angka 2 di
instruksi tidak bocor ke label. Petanya tetap, huruf ke-n alfabet untuk
semester n, dan tercatat di skrip.

## 4. Prediksi, ditulis sebelum menjalankan

1. Kalau kanal dokumen bekerja lewat permukaan, p1 tinggi dan p2 jatuh ke
   sekitar kunci. Besarnya tidak diprediksi.
2. p3 lebih rendah daripada p1 pada mayoritas model, karena instruksi menyebut
   nama sedangkan dokumen menyebut kode.
3. s2 turun dari s1 pada mayoritas model kalau efek semester adalah pencocokan
   angka. Kalau s2 bertahan, efek semester membawa makna.
4. s3 mendarat di kunci pada keenam model.
5. Bge-reranker-v2-gemma tetap di sekitar kunci di seluruh sel, seperti di
   setiap sel sebelumnya.

## 5. Aturan tafsir, mengikat

- Kontras penentu adalah p1 lawan p2. Keduanya memakai instruksi Inggris yang
  menyebut nama prodi, dan yang berbeda hanya bahasa labelnya. McNemar eksak
  berpasangan, Holm dalam keluarga enam model.
- Kalau p1 tinggi dan p2 di kunci, naskah WAJIB menyatakan kanal dokumen
  menuntut bentuk permukaan yang sama, dan keruntuhan anak tangga b tangga
  pertama diatribusikan ke bentuk permukaan, bukan ke penyeberangan bahasa.
- Kalau p1 dan p2 dua duanya tinggi, keruntuhan anak tangga b itu kegagalan
  lintas bahasa. Naskah WAJIB mencabut frasa "spelled the way the instruction
  spells it" dan menyatakan sebabnya bahasa.
- Kalau p1 juga di kunci, maka menyebut nama prodi di instruksi saja sudah
  memutus kanal, dan seluruh hasil kanal berlaku khusus untuk bentuk kode.
  Dilaporkan apa adanya, termasuk kalau itu melemahkan naskah.
- s2 lawan s1 menentukan apakah klaim "shared string" di abstrak boleh
  diperluas ke bidang semester. Kalau s2 bertahan, klaim itu dibatasi permanen
  ke bidang program dan kalimat pembatas di kesimpulan dipertahankan.
- Nol sel beku ditimpa. Nol angka lama diubah.
- Kalau hasil apa pun melemahkan naskah, ia tetap ditulis di badan naskah,
  bukan dipindah ke apendiks.

## 6. Artefak

Skrip sel: study/channel/scripts/score_architectures.py, sel `label_en`,
`label_id_en`, `label_kode_en`, `sem_kata`, `sem_opak`.
Agregator: study/channel/scripts/inference_ladder_surface.py.
Tabel: study/channel/results/tables/tangga_permukaan.json.
Pemecahan stratum: study/channel/scripts/inference_ladder_substring.py diperluas.

## Catatan operasional, ditambahkan saat antrean berjalan

JANGAN membangun ulang paper1_code atau menjalankan quick_check selagi antrean
GPU masih menulis berkas skor. Gerbang audit validitas membaca setiap berkas
skor, dan berkas yang sedang ditulis terbaca sebagai sel dengan n_komparabel
lebih kecil, sehingga pemeriksaan keseragaman n gagal. Itu bukan bug, itu
gerbangnya bekerja. Kejadian nyata 20 Sep: anomali {'paired': [50, 96]} berasal
dari sel gemma sem_opak yang baru terisi 50 dari 96 baris.

Urutan yang benar: tunggu antrean selesai, jalankan agregator TANPA
--izinkan-hilang, baru bangun paket dan jalankan gerbangnya.

## Dua insiden antrean, dicatat supaya tidak terulang

### 1. Batch dipaksa pada model yang tidak punya ruang

Empat sel FollowIR-7B gagal OOM karena `--batch 4` dipasang eksplisit. Bobot
model itu 13.813 MiB di kartu 16 GB, jadi sisanya cuma 718 MiB, dan probe `auto`
sebenarnya memilih batch 2. Tangga pertama TIDAK pernah memasang batch eksplisit
untuk FollowIR, dan justru karena itu ia lolos.

Pelajarannya dipakai terbalik. Pelajaran "pasang batch eksplisit" lahir dari
kasus probe memilih terlalu BESAR karena hanya melihat pool kueri pertama, dan
itu berlaku untuk ctxl. Untuk FollowIR probe sudah memilih dengan benar, dan
memaksanya naik justru mematikannya. Aturan yang benar: pasang batch eksplisit
hanya kalau probe terbukti memilih terlalu besar untuk model itu, bukan untuk
semua model besar.

### 2. Pool salah track, nyaris menghasilkan lima belas sel yang tidak sah

Lima belas langkah Qwen3 menunjuk `results/raw/pools_paired_expanded.jsonl`.
Nama berkas itu MENYESATKAN. Isinya 1.621 baris berprefiks `xpair-`, yaitu pool
TRACK LUAS, bukan track berpasangan. Pool track berpasangan yang benar hidup di
`study/retrieval/results/raw/pools_paired.jsonl`, 104 baris, semua
berprefiks `paired-`, sha256 ff0b146e3c75ed9f05a0080f52c28353c9ff353b0b1837413d1b87b602106483,
dan sha itu cocok dengan yang dipakai seluruh sel tangga pertama.

Kalau lolos, kelima belas sel akan menghasilkan angka 1.621 baris yang TIDAK
sebanding dengan jangkar 96 baris, dan tabelnya akan terlihat sah. Ini kelas
cacat yang sama dengan insiden cwd dan env, yaitu pengukuran tidak sah yang
kelihatan sah.

Yang menyelamatkannya kebetulan: path pool ditulis relatif sedangkan cwd langkah
adalah system, jadi kelima belas langkah mati dengan FileNotFoundError sebelum
menyentuh GPU. Uji asap gw sendiri TIDAK menangkapnya karena gw memverifikasi
bentuk keluarannya dan jumlah barisnya, tetapi tidak memverifikasi prefiks
query_id-nya.

Aturan baru: setiap sel baru wajib diverifikasi prefiks query_id keluarannya,
bukan cuma jumlah baris dan bentuk berkasnya. Dan setiap langkah antrean wajib
memakai path absolut, karena cwd langkah bukan akar repositori.
