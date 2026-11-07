# Praregistrasi pergantian peta label

Ditulis 2 Oktober 2026, SEBELUM satu pun sel diulang. Isinya prediksi, aturan
analisis, dan daftar hal yang tidak boleh berubah. Tujuannya supaya hasil
pengulangan tidak bisa dipilih-pilih setelah angkanya kelihatan.

## 1. Cacat yang diperbaiki

Sel reranker yang menyisipkan label dokumen membaca metadata kurikulum lewat
`muat_peta_metadata()`. Fungsi itu memetakan 180 karakter pertama teks kandidat
ke metadata korpus, dan memakai `setdefault` sehingga buku pertama yang cocok
menang. Dua buku OpenStax saling tumpang tindih, Introductory Statistics 2e
milik program STAT dan Introductory Business Statistics milik MGMT, dengan 322
pembuka chunk yang sama dan 110 chunk identik. Akibatnya sebagian paragraf STAT
menerima label MGMT.

Ukuran yang terukur, bukan taksiran:

| yang diperiksa | jalur berpasangan | jalur luas |
|---|---|---|
| gold salah label | 21 dari 96 | 30 dari 1.621 |
| kandidat bukan gold salah label | 370 dari 8.138 | 1.314 dari 110.762 |
| baris yang punya kandidat salah label | 94 dari 104 | 437 dari 1.621 |
| semester salah | 0 | 0 |

Peta lama juga kehilangan 1.140 chunk dari 64.108 karena prefiksnya bertabrakan,
dan `n_kandidat_tak_terpetakan` di manifest lama tetap menulis 0 karena setiap
prefiks memang ketemu, hanya ketemu buku yang salah. Gerbang lama memang tidak
bisa melihat cacat ini.

## 2. Yang diubah

`q2_peta_label.py` memetakan identitas kandidat ke metadata korpus, lewat `_id`
di `corpus.jsonl` ditambah alias di `uuid_to_canonical.json`. Identitas tidak
bisa bertabrakan, jadi kelas cacat ini hilang, bukan dikurangi.

Tujuh skrip memakai peta lama dan semuanya diberi saklar:

| skrip | saklar | bawaan |
|---|---|---|
| `q2_arm_score.py` | `--peta-label {id,teks}` | `id` |
| `q2_arch_score.py` | `--peta-label {id,teks}` | `id` |
| `q2_probe_injeksi.py` | `--peta-label {id,teks}` | `id` |
| `q2_strata_label_korup.py` | `--peta {teks,id}` | `teks` |
| `q2_strata_label_korup_arsitektur.py` | `--peta {teks,id}` | `teks` |
| `q2_label_salah_prodi.py` | `--peta {teks,id}` | `teks` |
| `q2_semester_gold_paired.py` | `--peta {teks,id}` | `teks` |

Empat skrip diagnostik bawaannya tetap `teks` karena tugasnya memang
mendokumentasikan cacat yang dilaporkan naskah. Mode `id` di skrip itu menulis
ke berkas bersuffiks `_petaid` supaya tabel beku tidak mungkin tertimpa.

## 3. Yang TIDAK diubah, dan ini mengikat

1. Pemilihan label salah di `dept_salah()` tetap memakai sha256 atas 180
   karakter teks. Itu garam deterministik, bukan pencarian metadata, jadi
   menggantinya akan menggeser sel label salah tanpa alasan.
2. Daftar `semua_dept` yang menyetir pemilihan itu sudah diverifikasi identik
   antara dua peta, delapan program dan lima semester, urutan sama.
3. Gold, counterpart, kolam kandidat, bobot DITS, konfigurasi model, dtype,
   batch, panjang maksimum, dan teks instruksi tidak disentuh.
4. Segel `test_holdout` tetap tertutup.
5. Berkas hasil lama tidak dihapus dan tidak ditimpa. Hasil baru masuk
   `results/raw_petaid/`.

## 4. Prediksi, ditulis sebelum angkanya ada

Pada 21 baris berpasangan yang cacat, gold diberi label program yang bukan
program penanya. Reranker yang membaca label akan menurunkan gold di baris itu.
Dengan label benar, gold seharusnya naik. Karena itu:

1. Jumlah menang jalur berpasangan pada sel dua kanal NAIK, paling banyak 21
   baris. Angka lama 82 sampai 96 dari 96 untuk lima model dan 51 untuk
   bge-gemma.
2. Jalur luas NAIK paling banyak 30 baris dari angka lama 696, 639, 810, 837,
   913, dan 600.
3. Gerakan dua arah yang lebih kecil boleh muncul dari 370 dan 1.314 kandidat
   bukan gold yang labelnya berubah, karena kandidat itu bisa naik melewati gold
   atau turun di bawahnya.
4. Sel slot saja tidak dijalankan ulang karena peta tidak pernah dibaca di sana,
   jadi temuan slot tidak bergerak sama sekali.
5. Sel label salah bergerak dengan arah yang TIDAK diprediksi, karena label
   salah dipilih relatif terhadap label benar dan label benar berubah di baris
   yang cacat.

Kalau kenaikan melewati 21 di jalur berpasangan atau 30 di jalur luas, atau
kalau angkanya TURUN, maka ada yang berubah selain peta label dan sel itu wajib
diselidiki sebelum dipakai.

## 5. Aturan analisis

1. Tahap r0 menjalankan enam sel wakil dengan peta LAMA ke folder baru. Lolos
   berarti nol baris berbalik dan selisih skor di bawah toleransi dtype, yaitu
   1e-6 untuk float16 dan 0,30 untuk bfloat16 sesuai lantai kebisingan zerank-2
   yang sudah tercatat. Kalau r0 tidak lolos, pengulangan tidak dilanjutkan
   karena berarti pipeline baru sendiri sudah menggeser angka.
2. Sel mode `id` wajib lolos sebelas pemeriksaan, dan yang paling mengikat
   adalah atribusi: baris yang hasilnya berbalik harus baris yang labelnya
   memang berubah. Batas baris berbalik di luar itu adalah 0 untuk float16 dan 8
   untuk bfloat16.
3. Angka lama dan angka baru dilaporkan berdampingan di naskah. Angka lama
   tidak dihapus dari repositori.
4. Kalau arah temuan berubah, yaitu kalau kanal slot jadi hidup atau kanal teks
   jadi mati, itu dilaporkan apa adanya dan naskah diubah mengikuti data. Tidak
   ada sel yang dibuang karena hasilnya tidak disukai.
5. Satu sel gagal menahan seluruh antrean. Tidak ada sel yang dilanjutkan di
   atas keluaran yang belum lolos validasi.

## 6. Gerbang yang wajib lolos sebelum sel pertama jalan

`q2_gerbang_peta_label.py`, sembilan pemeriksaan, nol GPU. Hasil 2 Oktober 2026:

| gerbang | hasil |
|---|---|
| G1 cakupan identitas | 8.138 dari 8.138 dan 110.762 dari 110.762 |
| G2 konflik pada prefiks unik | 0 dan 0 |
| G3 reproduksi ukuran cacat | 21 dan 30, sesuai angka naskah |
| G4 segel holdout | nol query_id holdout |
| G6 overlay pustaka | transformers 4.55.0, sentence-transformers 6.0.1 |
| G7 determinisme peta | sidik jari sama dua kali |

G2 adalah gerbang yang paling meyakinkan. Pada kandidat yang prefiks teksnya
unik di korpus, peta lama dan peta baru TIDAK PERNAH berbeda. Artinya peta baru
bukan peta yang berbeda sendirian, ia peta yang sama di setiap tempat yang tidak
ambigu, dan hanya berbeda di tempat yang memang ambigu.

## 7. Anggaran

101 sel dari 165 menyentuh label dokumen, dan hanya itu yang diulang. Menurut
durasi lari lama totalnya 1.333 menit GPU, yaitu 22,2 jam, terbagi 83 sel jalur
berpasangan 423 menit dan 18 sel jalur luas 911 menit. Puncak VRAM tertinggi di
antaranya 14.330 MiB, jadi tiap sel lewat pemeriksaan sisa VRAM 1.800 MiB
sebelum jalan.
