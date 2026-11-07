# Praregistrasi lintas arsitektur, adendum 3: slot instruksi dengan gaya kalimat masing-masing vendor

Ditulis 18 September 2026 pagi, SEBELUM satu pun sel di bawah dijalankan. Keadaan
saat ditulis: seluruh sel paired, G, dan Core17 untuk FollowIR-7B, ctxl-6B, dan
bge-reranker-v2-gemma sudah beku; replay jalur resmi zerank-2 tinggal Core17 dan
probe; E dan F belum mulai. Pemicu: kalimat instruksi profil untuk ketiga model
baru memakai satu kalimat imperatif yang sama dengan Qwen3
("Given a query from a <DEPT> student in semester <N>, retrieve passages ..."),
padahal tiap vendor mendokumentasikan gaya instruksi yang berbeda. Keberatan
yang mau ditutup: "slot nol karena kalimatnya bukan gaya model itu".

## Sel

Satu sel per model, `--cell slot_vendor` di `score_architectures.py`, track paired
96 baris komparabel, profil HANYA di slot, nol label dokumen, sama persis dengan
sel `slot` kecuali teks dan penempatan instruksinya:

- FollowIR-7B: gaya naratif README dan data latihnya, ditempel ke teks query
  (konvensi README yang menaruh instruksi di dalam query):
  `A relevant document would come from a textbook assigned to the <DEPT> degree
  program and would suit a student in semester <N>, rather than from a textbook
  of another degree program.`
- ctxl-6B: instruksi kustom pendek gaya contoh README ("Prioritize recent
  medical research"), ditempel setelah query dengan satu spasi:
  `Prioritize passages from the <DEPT> degree program's own textbooks at a level
  suitable for a semester <N> student`
- bge-reranker-v2-gemma: argumen `prompt` README berbentuk deskripsi tugas.
  Sel `slot` beku menempelkan profil SETELAH prompt bawaan; sel ini MENGGANTI
  prompt bawaan seluruhnya dengan deskripsi tugas yang memuat profil:
  `Given a query A from a <DEPT> student in semester <N> and a passage B,
  determine whether the passage comes from that student's own degree program
  and contains an answer to the query at a level appropriate for semester <N>
  by providing a prediction of either 'Yes' or 'No'.`

Template, skor, presisi (bf16), batas 1.536 token, pemotongan 4.000 karakter,
dan gap VRAM 300 MiB identik dengan sel beku. Satu smoke `--limit 2` untuk
gemma (jalur kode baru `set_prompt_penuh`) dijalankan dulu dan tidak dihitung.

## Hipotesis dan analisis, dikunci sekarang

H0: ketiga sel tetap tidak terbedakan dari kunci 48/96. Dugaan: 44 sampai 52.
Uji: McNemar eksak tiap sel lawan sel `none` beku model yang sama (keluarga V,
Holm m=3) dan, sebagai pembanding, lawan sel `slot` beku model yang sama.
Batas efek slot dihitung dengan `analisis()` yang sama (selang lebih lebar dari
Wald dan Clopper-Pearson).

Aturan baca: bila suatu sel lolos Holm 0,05 di atas kunci, klaim "slot nol"
untuk model itu DICABUT di naskah, sel `text` dan `both` diulang dengan gaya
vendor, dan gaya vendor menjadi sel slot utama model itu. Bila tidak, ketiga
sel dilaporkan satu kalimat di V-D (Not a Matter of Prompt Craft) dan
teksnya verbatim di apendiks. Apa pun hasilnya dilaporkan.

Yang TIDAK diubah: seluruh keluarga uji praregistrasi utama, adendum 1, dan
adendum 2. Sel ini keluarga diagnostik terpisah (V), tidak masuk keluarga E.
