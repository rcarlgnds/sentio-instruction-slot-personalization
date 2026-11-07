export const meta = {
  name: 'paired-set-generation',
  description: 'Generate neutral questions for paired CSCI/STAT statistics chunks, adversarially validate for lexical leakage',
  phases: [
    { title: 'Generasi', detail: 'pertanyaan netral per batch pasangan' },
    { title: 'Validasi', detail: 'deteksi kebocoran leksikal + kelayakan' },
    { title: 'Rakit', detail: 'susun set final berpasangan' },
  ],
}

const AI = '<workstation>/src/ai'
const PAIRS = `${AI}/data/paired_set/topic_pairs.jsonl`
const OUT = `${AI}/data/paired_set`

const RULES = `
KONTEKS: membangun set evaluasi BERPASANGAN untuk mengukur personalisasi pedagogis
pada sistem RAG pendidikan (tesis S2 SentIO, Binus).

Matkul "Basic Statistics" ditawarkan ke DUA prodi dengan BUKU BERBEDA:
  CSCI -> Introduction_to_Statistics (Lane, public domain)
  STAT -> Introductory Statistics 2e (OpenStax, CC BY)
Untuk topik yang dibahas KEDUA buku, SATU pertanyaan netral dipasangkan dengan DUA
profil mahasiswa. Jawaban benar BERBEDA tergantung prodi:
  mahasiswa CSCI -> gold = chunk buku Lane
  mahasiswa STAT -> gold = chunk buku OpenStax

INI INTI EKSPERIMENNYA: pertanyaannya identik, jadi cross-encoder dan similarity
murni BUTA (mereka tidak tahu mahasiswanya siapa). Hanya komponen yang membaca
profil mahasiswa yang punya informasi untuk membedakan.

SYARAT MUTLAK PERTANYAAN:
1. NETRAL SECARA LEKSIKAL. Tidak boleh memakai frasa, notasi, penamaan bab, atau
   contoh khas yang hanya muncul di salah satu buku. Kalau pertanyaan mengandung
   kata khas buku Lane, retrieval akan menemukan Lane karena kemiripan teks, bukan
   karena personalisasi, dan eksperimennya rusak.
2. DAPAT DIJAWAB DARI KEDUA chunk. Kalau hanya salah satu yang menjawab, pasangan
   itu tidak valid dan harus dibuang.
3. Bahasa INDONESIA informal-akademis, gaya mahasiswa bertanya, karena korpus
   berbahasa Inggris dan sistem ini memang cross-lingual (kueri Indonesia atas
   korpus Inggris). Ini karakteristik nyata sistem, pertahankan.
4. Panjang wajar 1-2 kalimat. Jangan menyalin kalimat dari chunk.
5. Jangan menyebut nama buku, penulis, nomor bab, atau nomor halaman.

ATURAN PENULISAN: DILARANG memakai em-dash, pakai "-". Bahasa Indonesia campur
istilah teknis Inggris.
`

const GEN_SCHEMA = {
  type: 'object',
  required: ['items'],
  properties: {
    items: {
      type: 'array',
      items: {
        type: 'object',
        required: ['pair_index', 'question', 'topic', 'usable', 'reason'],
        properties: {
          pair_index: { type: 'number' },
          question: { type: 'string' },
          topic: { type: 'string', description: 'nama topik singkat, mis. Central Limit Theorem' },
          usable: { type: 'boolean', description: 'false kalau pasangan tidak layak dipakai' },
          reason: { type: 'string', description: 'kalau usable false, jelaskan kenapa' },
        },
      },
    },
  },
}

phase('Generasi')

const BATCHES = [0, 1, 2, 3, 4, 5]

const generated = await parallel(BATCHES.map((bi) => () => agent(`${RULES}

TUGAS: baca file ${PAIRS} (format JSONL, tiap baris satu pasangan dengan field
"sim", "csci" {id,text,section,page}, "stat" {id,text,section,page}).

Ambil baris ke-${bi * 25} sampai ke-${bi * 25 + 24} (indeks mulai 0, jadi 25 pasangan).
Gunakan Read atau perintah shell seperti:
  sed -n '${bi * 25 + 1},${bi * 25 + 25}p' ${PAIRS}

Untuk TIAP pasangan:
1. Baca teks chunk CSCI dan chunk STAT.
2. Tentukan topik bersama yang dibahas keduanya.
3. Tulis SATU pertanyaan netral bahasa Indonesia yang bisa dijawab oleh KEDUANYA.
4. Kalau kedua chunk ternyata TIDAK membahas topik yang sama, atau salah satunya
   rusak (teks garbled, cuma rumus tanpa konteks, cuma daftar isi), set usable=false
   dan jelaskan. JANGAN memaksakan pertanyaan untuk pasangan yang jelek. Lebih baik
   sedikit tapi bersih.
5. pair_index = nomor baris absolut di file (mulai 0), yaitu ${bi * 25} + posisi dalam batch.

PENTING soal netralitas: sebelum menetapkan pertanyaan, cek sendiri apakah ada kata
di pertanyaanmu yang hanya muncul di salah satu chunk. Kalau ada, ganti dengan istilah
umum. Contoh buruk: kalau buku Lane menyebut "sampling distribution of the mean" dan
OpenStax menyebut "distribution of sample means", jangan pakai salah satunya persis;
pakai rumusan netral seperti "distribusi rata-rata sampel".

Kembalikan sesuai schema.`,
  { label: `gen:batch${bi}`, phase: 'Generasi', schema: GEN_SCHEMA, effort: 'high' })))

phase('Validasi')

const VAL_SCHEMA = {
  type: 'object',
  required: ['verdicts'],
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        required: ['pair_index', 'lolos', 'alasan'],
        properties: {
          pair_index: { type: 'number' },
          lolos: { type: 'boolean' },
          alasan: { type: 'string' },
          saran_perbaikan: { type: 'string' },
        },
      },
    },
  },
}

const kandidat = generated.filter(Boolean).flatMap((g) => g.items || []).filter((x) => x.usable)
log(`${kandidat.length} kandidat pertanyaan lolos tahap generasi`)

const CHUNKS = 4
const per = Math.ceil(kandidat.length / CHUNKS)
const validated = await parallel(
  Array.from({ length: CHUNKS }, (_, ci) => () => agent(`${RULES}

TUGAS VALIDASI ADVERSARIAL. Kamu SKEPTIS. Tugasmu MENJATUHKAN pertanyaan yang cacat,
bukan meloloskan sebanyak mungkin.

File pasangan: ${PAIRS} (baris ke-N sesuai pair_index, indeks mulai 0).

Kandidat yang harus kamu periksa:
${JSON.stringify(kandidat.slice(ci * per, (ci + 1) * per), null, 1)}

Untuk TIAP kandidat, ambil ulang teks kedua chunk dari file, lalu uji:

UJI 1 KEBOCORAN LEKSIKAL (paling penting). Hitung sendiri: berapa kata konten dari
pertanyaan yang muncul di chunk CSCI tapi TIDAK di chunk STAT, dan sebaliknya.
Kalau pertanyaan condong secara leksikal ke salah satu buku, sistem akan menemukan
buku itu karena kemiripan kata, BUKAN karena personalisasi. Itu merusak seluruh
eksperimen. TOLAK kalau condong.

UJI 2 KETERJAWABAN GANDA. Apakah KEDUA chunk benar-benar menjawab pertanyaan itu?
Kalau hanya satu yang menjawab, gold untuk prodi lain jadi salah. TOLAK.

UJI 3 KUALITAS TEKS. Kalau salah satu chunk garbled (banyak GLYPH, /uni, simbol rusak),
cuma potongan rumus tanpa konteks, atau cuma daftar isi, TOLAK.

UJI 4 KEWAJARAN. Apakah ini pertanyaan yang masuk akal diajukan mahasiswa? Kalau
terlalu artifisial atau menyalin kalimat chunk, TOLAK.

Default-mu adalah MENOLAK kalau ragu. Set lolos=false dan jelaskan. Kalau bisa
diperbaiki dengan mengganti beberapa kata, isi saran_perbaikan.`,
    { label: `validasi:${ci}`, phase: 'Validasi', schema: VAL_SCHEMA, effort: 'high' })))

phase('Rakit')

const verdicts = {}
validated.filter(Boolean).forEach((v) => (v.verdicts || []).forEach((x) => { verdicts[x.pair_index] = x }))
const lolos = kandidat.filter((k) => verdicts[k.pair_index]?.lolos)
log(`${lolos.length} dari ${kandidat.length} kandidat LOLOS validasi adversarial`)

const rakit = await agent(`${RULES}

TUGAS: rakit set evaluasi berpasangan final dan tulis ke ${OUT}/paired_queries.jsonl

Kandidat yang LOLOS validasi (${lolos.length} item):
${JSON.stringify(lolos, null, 1)}

Ambil id chunk dari ${PAIRS} sesuai pair_index (indeks baris mulai 0).

Department id yang harus dipakai (verifikasi sendiri lewat psql, JANGAN hardcode
kalau tidak cocok):
  PGPASSWORD=<redacted> psql -h localhost -p 5432 -U postgres -d postgres-db -t -A -F'|' \\
    -c "SELECT name, id FROM departments WHERE name IN ('CSCI','STAT');"
Semester mahasiswa: pakai semester course Basic Statistics di masing-masing prodi,
verifikasi lewat:
  PGPASSWORD=<redacted> psql -h localhost -p 5432 -U postgres -d postgres-db -t -A -F'|' \\
    -c "SELECT d.name, c.course_code, c.semester FROM courses c JOIN departments d ON c.department_id=d.id WHERE c.name='Basic Statistics';"

FORMAT tiap baris (satu baris = SATU kueri, jadi tiap pasangan menghasilkan DUA baris
dengan query_text IDENTIK tapi profil dan gold berbeda):
{
  "query_id": "paired-<pair_index>-<csci|stat>",
  "pair_id": "paired-<pair_index>",
  "class_label": "personalization_pair",
  "query_text": "<pertanyaan netral, IDENTIK untuk kedua baris>",
  "topic": "<topik>",
  "gold_relevant_chunks": ["<id chunk buku prodi ybs>"],
  "counterpart_chunks": ["<id chunk buku prodi LAWAN, yaitu jawaban SALAH untuk profil ini>"],
  "student_profile": {"department_id": "<uuid>", "department_name": "CSCI|STAT",
                      "current_semester": <int>, "enrolled_courses": []},
  "expected_answer_summary": "<ringkas 1 kalimat>",
  "annotation_status": "synthetic_unvalidated",
  "split": ""
}

CATATAN PENTING: enrolled_courses sengaja KOSONG supaya komponen Struct TIDAK aktif.
Set ini dirancang mengisolasi PROX (kecocokan prodi), bukan Struct. Tulis itu di
laporanmu.

Setelah menulis file, verifikasi dan laporkan:
1. Jumlah baris (harus 2x jumlah pasangan yang lolos).
2. Bahwa tiap pair_id punya TEPAT 2 baris dengan query_text identik.
3. Bahwa gold satu baris = counterpart baris pasangannya, dan sebaliknya.
4. Bahwa semua chunk id benar-benar ada di Qdrant. Cek dengan:
   curl -s http://localhost:6333/collections/sentio_knowledge/points -X POST \\
     -H 'Content-Type: application/json' -d '{"ids":["<id>"],"with_payload":["document_title","department_name"]}'
   dan pastikan document_title chunk gold cocok dengan prodi profilnya.
5. Sebaran topik.
Laporkan angka apa adanya. Kalau ada yang tidak konsisten, KATAKAN.`,
  { label: 'rakit:final', phase: 'Rakit', effort: 'high' })

return { n_generated: kandidat.length, n_lolos: lolos.length, rakit }
