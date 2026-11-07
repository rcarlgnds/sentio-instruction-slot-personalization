export const meta = {
  name: 'p2-tulis-saring',
  description: 'Track p2 round: writers write one neutral question per topic pair, separate screens apply the four tests of 29 July (pre-registration)',
  whenToUse: 'One round of track p2, after q2_p2_pasangan.py --tahap ronde N wrote the batch files',
  phases: [
    { title: 'Tulis', detail: 'one writer agent per batch of up to 25 topic pairs' },
    { title: 'Saring', detail: 'one screen agent per batch, never the agent that wrote it' },
  ],
}

const KONTEKS = `KONTEKS: membangun set evaluasi BERPASANGAN untuk mengukur personalisasi pedagogis
pada sistem RAG pendidikan (tesis S2 SentIO, Binus).

Topik yang sama dibahas di buku DUA prodi yang BERBEDA. Tiap pasangan topik berisi dua
chunk (Chunk 1 dan Chunk 2), masing-masing dari buku satu prodi; prodi dan judul bukunya
tertulis di tiap chunk. Untuk topik yang dibahas KEDUA buku, SATU pertanyaan netral
dipasangkan dengan DUA profil mahasiswa. Jawaban benar BERBEDA tergantung prodi:
  mahasiswa prodi Chunk 1 -> gold = Chunk 1
  mahasiswa prodi Chunk 2 -> gold = Chunk 2

INI INTI EKSPERIMENNYA: pertanyaannya identik, jadi cross-encoder dan similarity
murni BUTA (mereka tidak tahu mahasiswanya siapa). Hanya komponen yang membaca
profil mahasiswa yang punya informasi untuk membedakan.

SYARAT MUTLAK PERTANYAAN:
1. NETRAL SECARA LEKSIKAL. Tidak boleh memakai frasa, notasi, penamaan bab, atau
   contoh khas yang hanya muncul di salah satu buku. Kalau pertanyaan mengandung
   kata khas salah satu buku, retrieval akan menemukan buku itu karena kemiripan teks,
   bukan karena personalisasi, dan eksperimennya rusak.
2. DAPAT DIJAWAB DARI KEDUA chunk. Kalau hanya salah satu yang menjawab, pasangan
   itu tidak valid dan harus dibuang.
3. Bahasa INDONESIA informal-akademis, gaya mahasiswa bertanya, karena korpus
   berbahasa Inggris dan sistem ini memang cross-lingual (kueri Indonesia atas
   korpus Inggris). Ini karakteristik nyata sistem, pertahankan.
4. Panjang wajar 1-2 kalimat. Jangan menyalin kalimat dari chunk.
5. Jangan menyebut nama buku, penulis, nomor bab, atau nomor halaman.

ATURAN PENULISAN: DILARANG memakai em-dash, pakai "-". Bahasa Indonesia campur
istilah teknis Inggris.`

const BERKAS = (path, n) => `File pasangan: ${path}
Formatnya JSON: field "topic_pairs" berisi ${n} pasangan; tiap pasangan punya "tp_id",
"program_pair", dan "paragraf" berisi dua chunk dengan field "label" (Chunk 1 atau
Chunk 2), "program", "book_title", "section", "text". Baca berkas itu dengan Read.
Jangan menulis atau mengubah berkas apa pun dan jangan menjalankan perintah lain.`

function promptTulis(path, n) {
  return `
${KONTEKS}


TUGAS: tulis pertanyaan untuk setiap pasangan di berkas berikut.
${BERKAS(path, n)}

Untuk TIAP pasangan:
1. Baca teks Chunk 1 dan Chunk 2.
2. Tentukan topik bersama yang dibahas keduanya.
3. Tulis SATU pertanyaan netral bahasa Indonesia yang bisa dijawab oleh KEDUANYA.
4. Kalau kedua chunk ternyata TIDAK membahas topik yang sama, atau salah satunya
   rusak (teks garbled, cuma rumus tanpa konteks, cuma daftar isi), set usable=false
   dan jelaskan. JANGAN memaksakan pertanyaan untuk pasangan yang jelek. Lebih baik
   sedikit tapi bersih.
5. tp_id = tp_id pasangan itu, persis seperti di berkas. Setiap tp_id di berkas muncul
   tepat sekali di jawabanmu.

PENTING soal netralitas: sebelum menetapkan pertanyaan, cek sendiri apakah ada kata
di pertanyaanmu yang hanya muncul di salah satu chunk. Kalau ada, ganti dengan istilah
umum. Contoh buruk: kalau buku pertama menyebut "sampling distribution of the mean" dan
buku kedua menyebut "distribution of sample means", jangan pakai salah satunya persis;
pakai rumusan netral seperti "distribusi rata-rata sampel".

Kembalikan sesuai schema.`
}

function promptSaring(path, n, kandidat) {
  return `
${KONTEKS}


TUGAS VALIDASI ADVERSARIAL. Kamu SKEPTIS. Tugasmu MENJATUHKAN pertanyaan yang cacat,
bukan meloloskan sebanyak mungkin.

${BERKAS(path, n)}
Cari pasangan tiap kandidat lewat tp_id.

Kandidat yang harus kamu periksa:
${JSON.stringify(kandidat, null, 1)}

Untuk TIAP kandidat, ambil ulang teks kedua chunk dari file, lalu uji:

UJI 1 KEBOCORAN LEKSIKAL (paling penting). Hitung sendiri: berapa kata konten dari
pertanyaan yang muncul di Chunk 1 tapi TIDAK di Chunk 2, dan sebaliknya.
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
diperbaiki dengan mengganti beberapa kata, isi saran_perbaikan.

Setiap tp_id kandidat muncul tepat sekali di jawabanmu. Kembalikan sesuai schema.`
}

const SKEMA_TULIS = {
  type: 'object',
  properties: {
    items: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          tp_id: { type: 'string' },
          usable: { type: 'boolean' },
          question: { type: 'string' },
          topic: { type: 'string' },
          reason: { type: 'string' },
        },
        required: ['tp_id', 'usable', 'question', 'topic', 'reason'],
      },
    },
  },
  required: ['items'],
}

const SKEMA_SARING = {
  type: 'object',
  properties: {
    items: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          tp_id: { type: 'string' },
          lolos: { type: 'boolean' },
          alasan: { type: 'string' },
          saran_perbaikan: { type: 'string' },
        },
        required: ['tp_id', 'lolos', 'alasan', 'saran_perbaikan'],
      },
    },
  },
  required: ['items'],
}

if (!args || typeof args.ronde !== 'number' || !Array.isArray(args.batches) || !args.batches.length) {
  throw new Error('args must be {ronde: N, batches: [{name, path, n}]}')
}

phase('Tulis')
const hasil = await pipeline(
  args.batches,
  (b) => agent(promptTulis(b.path, b.n), {
    schema: SKEMA_TULIS, label: `tulis r${args.ronde} ${b.name}`, phase: 'Tulis', effort: 'high',
  }),
  (tulis, b) => {
    if (!tulis) {
      log(`${b.name}: writer returned nothing`)
      return { batch: b.name, tulis: null, saring: null }
    }
    if (tulis.items.length !== b.n) {
      log(`${b.name}: writer returned ${tulis.items.length} items for ${b.n} topic pairs`)
    }
    const kandidat = tulis.items.filter((x) => x.usable)
    if (!kandidat.length) {
      return { batch: b.name, tulis, saring: { items: [] } }
    }
    return agent(promptSaring(b.path, b.n, kandidat), {
      schema: SKEMA_SARING, label: `saring r${args.ronde} ${b.name}`, phase: 'Saring', effort: 'high',
    }).then((saring) => ({ batch: b.name, tulis, saring }))
  },
)

const ditulis = hasil.filter(Boolean).reduce((s, h) => s + (h.tulis ? h.tulis.items.filter((x) => x.usable).length : 0), 0)
const lolos = hasil.filter(Boolean).reduce((s, h) => s + (h.saring ? h.saring.items.filter((x) => x.lolos).length : 0), 0)
log(`round ${args.ronde}: ${args.batches.length} batches, ${ditulis} questions written, ${lolos} passed the screen`)
return { ronde: args.ronde, batches: hasil }
