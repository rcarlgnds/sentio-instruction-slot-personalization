# Praregistrasi lintas arsitektur, adendum 4: instruksi profil dalam bahasa kueri

Ditulis 18 September 2026 pukul 08:30 WIB, SEBELUM satu pun sel di bawah
dijalankan. Pemicu: seluruh sel slot memakai instruksi profil berbahasa
Inggris di atas kueri berbahasa Indonesia (kombinasi yang direkomendasikan
model card Qwen3-Reranker), sementara kontrol tekstual yang membuktikan slot
bisa bekerja memakai kueri berbahasa Inggris. Keberatan yang mau ditutup:
"slot nol karena bahasa instruksi tidak sama dengan bahasa kueri".

## Sel

Satu sel per model, enam model, track paired 96 baris komparabel, profil
HANYA di slot, nol label dokumen, penempatan identik dengan sel `slot` beku
tiap model (field instruct Qwen3, teks polos di depan query untuk zerank-2
jalur resmi, setelah query untuk FollowIR-7B dan ctxl-6B, ditempel ke prompt
bawaan untuk bge-reranker-v2-gemma). Teks instruksi, padanan langsung dari
kalimat Inggris beku:

    Diberikan pertanyaan dari mahasiswa program studi <DEPT> semester <N>,
    ambil bagian teks dari program studi mahasiswa itu sendiri yang menjawab
    pertanyaan tersebut pada tingkat yang sesuai untuk semester <N>

Perintah: `score_arms.py --instruction profile_id` (qwen3rr 0.6B batch 8
float16, 4B batch 4 float16, zerank2official batch 2 bf16 max-len 1536) dan
`score_architectures.py --cell slot_id` (followir, ctxl, gemma, bf16 1536, gap 300).
Dua smoke `--limit 2` (gemma slot_id, qwen3rr 0.6B profile_id) mendahului dan
tidak dihitung.

## Hipotesis dan analisis, dikunci sekarang

H0: keenam sel tetap tidak terbedakan dari kunci 48/96. Dugaan: 44 sampai 52.
Uji: McNemar eksak tiap sel lawan sel `none` beku model yang sama (untuk
Qwen3 0.6B/4B sel default novelty_1 yang dikutip Tabel 5; untuk zerank-2 sel
default jalur resmi), keluarga ID, Holm m=6; pembanding lawan sel slot Inggris
beku model yang sama; batas efek dengan `analisis()`.

Aturan baca: bila suatu sel lolos Holm 0,05 di atas kunci, klaim "slot nol"
untuk model itu DICABUT, sel `text` dan `both` diulang dengan instruksi
Indonesia, dan Limitasi 3 ditulis ulang. Bila tidak, satu kalimat di V-D dan
kalimat Limitasi 3 diperbarui dari "belum diuji" menjadi hasilnya.
Apa pun hasilnya dilaporkan. Praregistrasi utama dan adendum 1-3 tidak diubah.
