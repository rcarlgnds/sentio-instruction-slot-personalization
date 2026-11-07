# Adendum 2 praregistrasi lintas arsitektur: paket G, E, F, dan eksekusi konsekuensi kesetaraan

Ditulis 17 Sep 2026, 20:00 WIB, SEBELUM satu pun sel di bawah ini dijalankan.
Keadaan saat ditulis: sel paired FollowIR-7B lima dari lima selesai, ctxl dua
dari enam, gemma nol, Core17 belum ada untuk ketiga model baru; nol sel G, E,
F; jalur resmi zerank-2 baru ada dua kondisi (default, profile) dari uji
kesetaraan penuh. Keputusan menjalankan semuanya, termasuk konsekuensi
kesetaraan secara penuh, diambil penulis utama pada 17 Sep 2026 malam.
Bagian 1 sampai 9 praregistrasi utama dan adendum kesetaraan tidak diubah.

## G. Sel label satu-field untuk tiga keturunan baru

Sel `both_program` dan `both_semester` di `score_architectures.py`: profil di slot
instruksi (identik dengan sel `both`) dan label dokumen yang hanya memuat
satu field, dirender oleh `label_sumber(m, field)` yang sama dengan sel beku
`label_satu_field`:

    program  : [Source: <DEPT> degree program textbook]
    semester : [Source: degree program textbook, semester <N>]

Tiga model: bge-reranker-v2-gemma, ctxl-rerank-v2-instruct-multilingual-6b,
FollowIR-7B. Parameter lain identik dengan sel paired (bf16, max-len 1536,
gap 500 MiB, batch otomatis). Enam sel, taksiran 35 menit GPU.

Dilaporkan per model: hitungan menang dari 96 untuk kedua sel, disandingkan
dengan `none`, `both`, dan kunci 48. Tiga kontras McNemar eksak per model:
program lawan both, semester lawan both, program lawan semester. Keluarga G
adalah sembilan kontras itu (tiga model kali tiga), koreksi Holm m = 9.

Aturan baca, ditetapkan sekarang: pola beku adalah semester-saja mendekati
label penuh (78/84/93 lawan 82/84/95) dan prodi-saja jauh di bawahnya tetapi
di atas kunci (62/74/76). Replikasi berarti semester lawan both tidak
signifikan dan program lawan both signifikan pada arah yang sama. Pola lain
apa pun dilaporkan apa adanya sebagai perbedaan keturunan; tidak ada sel
yang dibuang.

## E. Track luas 1.621 baris untuk tiga keturunan baru

Pool `pools_paired_expanded.jsonl` (sha256 65ac6f2fcdc6e662, 1.621 baris,
110.762 dokumen, 8 prodi), tag `xpair`, empat sel yang mencerminkan sel beku
track luas: `none`, `slot`, `both`, `wrong` (padanan beku: default, profile,
profile_docmeta, profile_docmetasalah). Tiga model, dua belas sel, taksiran
16 jam GPU (FollowIR 8,7 jam pada 14 dokumen per detik, ctxl 5,4 jam, gemma
1,9 jam). Tidak ada sel teks-saja, sama seperti track beku.

Dilaporkan per model: hitungan menang dari 1.621 lawan kunci buta struktural
600/1621; McNemar eksak none lawan slot per model, keluarga E2 m = 3 Holm;
batas ekuivalensi efek slot dengan `analisis()` yang sama (lebar terlebar
dari Wald dan eksak); efek kanal teks (both dikurangi none) per model DI
DALAM track ini. Rasio antar track dihitung per track, tidak pernah
mencampur penyebut.

Aturan baca: replikasi berarti slot di dalam beberapa baris dari kunci 600
dengan batas atas di bawah 1 poin persen, dan both jauh di atasnya. Bila
salah satu model bergerak lewat slot di track ini, itu dilaporkan sebagai
temuan, bukan disembunyikan.

## F. News21 dan Robust04 untuk tiga keturunan baru

`followir_architectures.py --dataset News21` dan `--dataset Robust04`, kandidat
1.000 per kueri, metrik p-MRR dan nDCG@5 dari implementasi MTEB yang sudah
dipakai, tiga model, enam run, taksiran 6 jam GPU (News21 64 kueri, Robust04
104 kueri). Kriteria kontrol positif tetap yang ada di praregistrasi utama
untuk Core17; News21 dan Robust04 dilaporkan sebagai tambahan di tabel yang
sama tanpa kriteria baru, supaya tabel FollowIR memuat tiga dataset untuk
enam model kecuali zerank-2 yang bekunya hanya Core17 (dan akan dilengkapi di
bawah).

## (a) Eksekusi konsekuensi bagian 3 adendum kesetaraan

Konsekuensi yang dipra-registrasi adalah menskor ulang SELURUH sel zerank-2
dengan jalur resmi vendor. Cara: keluarga skorer baru `zerank2official` di
`score_arms.py`, aditif, perilaku keluarga lama tidak berubah, memanggil
`sentence_transformers.CrossEncoder(...).predict()` (versi 6.0.1 di overlay
`system/.st-libs`, bf16, max_length 1536, batch 2, identik dengan uji
kesetaraan penuh). Seluruh konstruksi masukan (instruksi, format slot,
label dokumen, label salah, memori vendor, pemotongan 4.000 karakter)
tetap milik skrip yang sama, sehingga tiap sel diulang dengan perintah
persis dari manifest bekunya dan hanya nama keluarga yang diganti.

Sel yang diulang, semuanya yang bermanifest beku: control_default,
control_kondisi, control_profile (116 baris); holdout_default,
holdout_profile (283 baris, segel: pool holdout sudah pernah dibuka untuk
sel beku ini, ini pembacaan ulang pool yang sama, nol tuning); paired
default, default_docmeta, profile, profile_docmeta, profile_docmeta_fieldprogram,
profile_docmeta_fieldsemester, profile_docmetasalah, profile_fmtlabeled,
profile_fmttag, profile_vmem (104 baris); traindev_default, traindev_profile
(2.493 baris); xpair default, profile, profile_docmeta, profile_docmetasalah
(1.621 baris); Core17; probe injeksi tujuh strategi. Total sekitar 690.000
dokumen, taksiran 7 jam GPU pada 26 dokumen per detik.

Dilaporkan untuk tiap sel: hitungan beku, hitungan jalur resmi, kesepakatan
per baris, dan jumlah baris seri-dalam-presisi (|selisih skor gold lawan
pasangan| <= 0,25). Sesuai bagian 3 adendum kesetaraan, angka naskah beralih
ke nilai jalur resmi dan KEDUA angka dilaporkan di Appendix B. Diharapkan
sebelumnya, dan ditulis di sini supaya tidak terbaca sebagai pembelaan
sesudah angka ada: sel dengan 6 sampai 8 baris seri diperkirakan bergeser
nol sampai dua baris, dan pergeseran sebesar itu TIDAK mengubah kesimpulan
mana pun karena semua uji memakai diskordan berpasangan, bukan hitungan
mentah.

## Urutan eksekusi dan aturan umum

Urutan menurut nilai untuk naskah: G, H (praregistrasi terpisah di
`study/retrieval/PRAREGISTRASI_AUTO_KURASI.md`), (a), I (praregistrasi
terpisah di `study/retrieval/PRAREGISTRASI_HOLDOUT_GENERASI.md`), E, F.
Satu pekerjaan GPU sekali waktu lewat `raw/.gpu.lock`, tunggu kartu kosong
sebelum memuat, gap 500 MiB (keputusan pemilik untuk run ini). Setiap sel
punya manifest sendiri; sel yang selesai tidak diulang; sel yang gagal
diulang sekali dan kalau gagal lagi dilaporkan gagal. Tidak ada sel yang
dibuang karena hasilnya.
