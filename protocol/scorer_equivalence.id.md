# Adendum praregistrasi: tindak lanjut uji kesetaraan skorer zerank-2

Ditulis 17 Sep 2026, SESUDAH uji bagian 7 `PRAREGISTRASI_LINTAS_ARSITEKTUR.md`
selesai dan SEBELUM tindak lanjutnya dijalankan. Adendum ini mencatat hasil
uji itu apa adanya, menyatakan bahwa kriterianya salah spesifikasi, dan
mengunci pertanyaan lanjutan beserta kriterianya sebelum angkanya ada.

## 1. Hasil uji bagian 7, verbatim dari `tables/zerank2_equivalence.json`

| ukuran | nilai |
|---|---|
| Spearman rho, 100 pasangan | 0,998466 |
| inversi urutan antar jalur | 16 (14 pada kueri berkandidat 71, 2 pada kueri berkandidat 29) |
| selisih absolut A lawan B | median 0,0625, p90 0,125, maks 0,25; 35 dari 100 sama persis |
| selisih jalur A lawan berkas beku sel yang sama | maks 0,2909; 29 dari 100 sama persis |
| kriteria pra-registrasi (nol inversi dan maks <= 0,05) | **GAGAL** |

Vonis GAGAL berdiri sebagaimana ditulis. Ia tidak dilonggarkan.

## 2. Kenapa kriterianya salah spesifikasi, dan apa yang dibuktikan datanya

Logit mentah zerank-2 pada pool ini terentang -13,19 sampai 13,94. Pada besaran
8 sampai 16, resolusi bfloat16 (ulp) adalah 0,0625. Jadi selisih median 0,0625
adalah tepat satu ulp, dan maksimum 0,25 adalah empat ulp. Toleransi 0,05 yang
saya tulis berada DI BAWAH satu ulp pada besaran logit yang sebenarnya; ia
ditetapkan tanpa memeriksa skala logit lebih dulu, dan itu kesalahan saya.

Dua fakta yang menentukan bacaannya:
- Skorer proyek yang SAMA, dijalankan ulang atas input yang sama, berselisih
  sampai 0,2909 dengan keluaran bekunya sendiri. Derau antar-run kode yang
  sama sudah SEBESAR selisih terhadap jalur vendor.
- Ke-16 inversi seluruhnya pasangan kandidat yang jarak skornya <= 0,1875 di
  jalur A dan <= 0,25 di jalur B, yaitu pasangan yang seri di dalam presisi
  bf16. Tidak ada satu pun inversi pada pasangan yang jaraknya melampaui
  empat ulp.

Kesimpulan yang boleh ditarik: tidak ada perbedaan SISTEMATIS antara skorer
proyek dan jalur resmi vendor; yang ada adalah derau pembulatan bf16 yang juga
hidup di antara dua run skorer yang sama. Kesimpulan yang TIDAK boleh ditarik:
bahwa kriteria awal lolos. Kedua hal itu ditulis di Appendix B naskah.

## 3. Pertanyaan lanjutan, dikunci sekarang

Yang penting bagi naskah bukan kesamaan logit per pasangan, melainkan apakah
jalur resmi vendor mereproduksi DUA SEL BEKU zerank-2 yang dikutip naskah:
48 dari 96 (instruksi generik) dan 47 dari 96 (profil di slot).

- Jalur resmi `CrossEncoder.predict()` (sentence-transformers 6.0.1 di overlay
  `system/.st-libs`, bf16, max_length 1536) menskor SELURUH pool paired, 104
  baris, pada kedua kondisi, dengan teks query diformat persis seperti sel
  beku (format slot polos: instruksi, baris kosong, "Query: ...").
- Keluaran ditulis dalam format sel standar dan dibaca `metrics` dan
  `mcnemar` tanpa perubahan.
- Yang dilaporkan: hitungan menang jalur resmi lawan beku untuk tiap kondisi,
  jumlah baris yang vonisnya sama dari 96, daftar baris yang berbeda, dan
  McNemar exact generik-lawan-profil di jalur resmi.
- Kriteria lolos, ditetapkan sekarang: hitungan menang IDENTIK dengan sel beku
  pada KEDUA kondisi (48 dan 47). Kesepakatan per baris dilaporkan tetapi tidak
  menjadi syarat, karena satu baris yang seri di dalam presisi bisa berbalik
  tanpa mengubah hitungan.
- Bila tidak lolos: seluruh sel zerank-2 diskor ulang memakai jalur resmi,
  angka naskah diperbarui, dan keduanya (hasil bagian 7 dan hasil ini)
  dilaporkan. Tidak ada opsi menyembunyikan salah satunya.

Skrip: `scripts/zerank2_equivalence_full.py`. Keluaran:
`tables/zerank2_equivalence_full.json`, `raw/scores_zerank2official_zerank-2_paired_{default,profile}.jsonl`.

## 4. Hasil tindak lanjut, ditulis 17 Sep 2026 SESUDAH angkanya ada

Bagian 1 sampai 3 di atas tidak diubah satu kata pun. Bagian ini mencatat
hasil `tables/zerank2_equivalence_full.json` (durasi GPU 543.3 s,
sentence-transformers 6.0.1, transformers
4.55.0) dan status konsekuensinya.

| kondisi | jalur resmi | sel beku | hitungan identik | vonis per baris sama |
|---|---|---|---|---|
| generik (default) | 48/96 | 48/96 | ya | 96 dari 96 |
| profil di slot | 48/96 | 47/96 | TIDAK | 95 dari 96 |

Kriteria bagian 3 (hitungan identik pada KEDUA kondisi): **GAGAL**, selisih
satu baris, `paired-91-csci`, pada kondisi profil.

Anatomi baris itu. Di sel beku, gold berskor 3,40625 dan gold pasangannya
3,578125 (selisih -0,17, gold kalah). Di jalur resmi, gold 3,609375 dan
pasangan 3,59375 (selisih +0,02, gold menang). Kedua selisih berada di dalam
derau antar-run skorer yang SAMA yang tercatat di bagian 2 (sampai 0,29).
Pada kondisi profil, 8 dari 96 baris komparabel mempunyai |skor gold - skor
pasangan| <= 0,25 di kedua jalur; satu di antaranya berbalik. Pada kondisi
generik, 6 (beku) dan 8 (resmi) baris berada di pita yang sama dan tidak ada
yang berbalik. Angka menang tiap sel dengan demikian membawa ketidakpastian
sebesar jumlah baris seri-dalam-presisi, kira-kira 6 sampai 8 dari 96, dan
itu berlaku untuk jalur mana pun.

Yang tidak berubah: McNemar generik lawan profil di jalur resmi 8 lawan 8
diskordan, p = 1,0; di sel beku 7 lawan 8, p = 1,0. Batas ekuivalensi efek
slot dengan diskordan 8/8 adalah 8.45 poin persen; batas beku zerank-2
8.98 poin persen. nDCG@5 jalur resmi 0,5710 (generik) dan 0,5342
(profil) lawan beku 0,5769 dan 0,5313.

Konsekuensi yang dipra-registrasi di bagian 3 adalah menskor ulang SELURUH
sel zerank-2 dengan jalur resmi. Sel zerank-2 yang dikutip naskah: 6 sel
track paired, 2 sel format vendor, 2 sel satu-field label, 7 sel probe
injeksi, 4 sel track luas 1.621 baris, dan 1 run Core17; 22 sel. Taksiran
biaya dari pengukuran hari ini: sel seukuran paired sekitar 4,5 menit
(543 s untuk dua sel), sel track luas 15,6 kali itu, Core17 2.800 s pada
run beku; totalnya sekitar 6 sampai 7 jam GPU.

Status pada saat bagian ini ditulis: konsekuensi itu BELUM dijalankan.
Keputusannya diserahkan kepada penulis utama, dengan dua alasan yang
dinyatakan terbuka: biaya GPU di atas, dan penilaian bahwa skor ulang
mengganti satu undian derau bf16 dengan undian derau lain, karena skorer
yang sama lawan dirinya sendiri sudah berselisih sebesar selisih antar
jalur. Apa pun keputusannya, hasil bagian 1 dan bagian 4 keduanya masuk
naskah. Bila konsekuensi tidak dijalankan, naskah menyatakannya sebagai
penyimpangan dari pra-registrasi beserta alasan di atas, dan angka beku
tetap dikutip apa adanya bersama angka jalur resmi.
