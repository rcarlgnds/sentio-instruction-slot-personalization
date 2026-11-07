TEMPLATE_VERSION = "rads-t1.4"

NATURALISM = """GAYA NYATA MAHASISWA (WAJIB): tulis kueri persis seperti mahasiswa
Indonesia sungguhan mengetik ke asisten belajar, bukan seperti AI atau filsuf.
DILARANG KERAS frasa pembuka artifisial: "Mari kita renungkan", "Coba bantu aku
merenungkan", "Ajaklah saya", "Mari berpikir bersama", "Renungkanlah",
"Bantulah saya memikirkan", "Ajak saya memahami", dan sejenisnya. Kueri
reflektif tetap boleh, tapi dengan suara natural mahasiswa: "kok bisa ...
kenapa ya?", "sebenernya maksudnya apa sih ...?", "aku masih bingung kenapa ...".
"""

OUTPUT_CONTRACT = """SYARAT KUALITAS: kueri HARUS terjawab SPESIFIK dari chunk yang
diberikan, bukan pertanyaan umum yang kebetulan satu topik. Jangkarkan kueri ke
konsep/istilah/contoh yang benar-benar muncul di chunk.

Balas HANYA dengan JSON valid (tanpa markdown fence, tanpa teks lain) berskema:
{
  "query_text": "<kueri mahasiswa dalam bahasa Indonesia, natural, 1 kalimat tanya>",
  "expected_answer_summary": "<ringkasan jawaban benar 2-4 kalimat, bahasa Indonesia>",
  "reasoning_chain": ["<langkah penalaran 1>", "<langkah 2>", "..."],
  "gold_relevant_chunks": ["<chunk_id yang benar-benar dibutuhkan untuk menjawab, subset dari chunk yang diberikan>"],
  "difficulty": "easy|medium|hard"
}"""


def _difficulty_line(bundle) -> str:
    target = bundle.class_metadata.get("difficulty_target")
    if not target:
        return ""
    depth = {"easy": "definisional langsung",
             "medium": "butuh menghubungkan 2+ konsep",
             "hard": "butuh penalaran berlapis / kasus tepi"}[target]
    return f"TARGET tingkat kesulitan: {target} ({depth}). Set field difficulty sesuai target ini kalau kuerimu memang mencapainya; kalau tidak, isi jujur.\n"


def _style_line(bundle) -> str:
    style = bundle.class_metadata.get("style_target")
    if not style:
        return ""
    guide = {"formal": "bahasa Indonesia akademik yang LUGAS dan to the point; "
                       "kalimat tanya biasa, tanpa gaya filosofis atau retorika berbunga",
             "informal": "bahasa chat mahasiswa sehari-hari: boleh singkatan lazim "
                         "(yg, gmn, tp, krn, udh), partikel (sih, dong, deh, kok), "
                         "dan maksimal satu typo ringan yang wajar; langsung ke intinya"}[style]
    return f"REGISTER bahasa kueri: {style}, yaitu {guide}. Register ini variabel eksperimen terkontrol, patuhi.\n{NATURALISM}"


def _chunk_block(chunks) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[chunk_id: {c.chunk_id}]\n[course: {c.course_code} {c.course_name} | program: {c.department_name} | semester: {c.semester}]\n{c.raw_text}")
    return "\n\n---\n\n".join(parts)


def build_prompt(bundle) -> str:
    builder = {
        "multi_disciplinary_ambiguity": _class1,
        "prerequisite_chain_depth": _class2,
        "temporal_scope_drift": _class3,
        "adversarial_routing": _class4,
    }[bundle.class_label]
    return builder(bundle)


def _class1(bundle) -> str:
    term = bundle.class_metadata["ambiguous_term"]
    depts = ", ".join(bundle.class_metadata["departments"])
    return f"""Kamu membuat kueri benchmark untuk menguji sistem RAG kurikulum universitas.

KELAS: Multi-Disciplinary Terminology Ambiguity.
Istilah "{term}" dipakai dengan makna BERBEDA di beberapa program studi ({depts}).
Di bawah ini chunk dari masing-masing program yang memakai istilah itu.

TUGAS: buat SATU kueri mahasiswa berbahasa Indonesia yang memakai istilah "{term}"
secara AMBIGU: tanpa konteks program, kueri ini bisa dijawab dari lebih dari satu
program. Kueri WAJIB memuat kata "{term}" itu sendiri (atau padanan Indonesianya
yang lazim) secara verbatim - tanpa istilah itu, kelas ambiguitas gagal.
Kueri harus natural (bukan soal ujian), seperti mahasiswa bertanya ke
asisten belajar. PENJANGKARAN: kueri harus tetap ambigu lintas program, TAPI
sertakan 1-2 kata konteks topikal yang benar-benar muncul di chunk program
target (bukan nama programnya), supaya kueri tetap terjawab spesifik dan
menonjol saat retrieval. expected_answer_summary ditulis untuk SALAH SATU program (pilih
yang paling cocok dengan frasa kueri) dan sebutkan program itu di dalamnya.
gold_relevant_chunks = chunk dari program terpilih saja.

{_chunk_block(bundle.chunks)}

{_difficulty_line(bundle)}{_style_line(bundle)}{OUTPUT_CONTRACT}"""


def _class2(bundle) -> str:
    hop = bundle.class_metadata["hop_distance"]
    pair = " -> ".join(bundle.class_metadata["course_pair"])
    return f"""Kamu membuat kueri benchmark untuk menguji sistem RAG kurikulum universitas.

KELAS: Prerequisite Chain Depth ({hop} hop pada graf prasyarat pengetahuan: {pair}).
Dua chunk di bawah berasal dari course yang terhubung rantai prasyarat pengetahuan
sepanjang {hop} hop (fondasi -> lanjutan).

TUGAS: buat SATU kueri mahasiswa berbahasa Indonesia yang untuk dijawab dengan
benar MEMBUTUHKAN kedua sisi rantai: konsep fondasi DAN konsep lanjutan
(misal: "kenapa X (fondasi) penting untuk memahami Y (lanjutan)?", tapi jangan
pakai pola itu mentah-mentah, variasikan). reasoning_chain harus menunjukkan
jembatan dari fondasi ke lanjutan. gold_relevant_chunks = KEDUA chunk.

{_chunk_block(bundle.chunks)}

{_difficulty_line(bundle)}{_style_line(bundle)}{OUTPUT_CONTRACT}"""


def _class3(bundle) -> str:
    meta = bundle.class_metadata
    return f"""Kamu membuat kueri benchmark untuk menguji sistem RAG kurikulum universitas.

KELAS: Temporal Scope Drift.
Chunk di bawah berasal dari konten SEMESTER {meta["content_semester"]}.
Profil penanya: mahasiswa SEMESTER {meta["student_semester"]}
(drift {meta["drift"]:+d}; konten {"di atas" if meta["drift"] < 0 else "di bawah"} level penanya).

TUGAS: buat SATU kueri berbahasa Indonesia yang WAJAR ditanyakan mahasiswa
semester {meta["student_semester"]} tapi jawabannya ada di konten level
semester {meta["content_semester"]} ini. Kueri tidak boleh menyebut semester
secara eksplisit. expected_answer_summary harus menjawab pada level konten
(bukan disederhanakan). gold_relevant_chunks = chunk yang diberikan.

{_chunk_block(bundle.chunks)}

{_difficulty_line(bundle)}{_style_line(bundle)}{OUTPUT_CONTRACT}"""


def _class4(bundle) -> str:
    style = bundle.class_metadata["content_style"]
    target = bundle.class_metadata["target_boundary"]
    form = bundle.class_metadata.get("query_form", "long")
    instruction = (
        "buat kueri yang BERBUNYI reflektif/terbuka padahal jawabannya "
        "faktual-definisional dan langsung ada di chunk. Reflektifnya harus "
        "bersuara mahasiswa penasaran ('kok bisa gitu ya, kenapa?', 'sebenernya "
        "apa sih bedanya...'), BUKAN bahasa filsuf"
        if target == "sounds-socratic-needs-factual" else
        "buat kueri yang BERBUNYI faktual-singkat (seperti minta definisi/angka) "
        "padahal menjawabnya dengan benar butuh penalaran konseptual dari chunk"
    )
    form_line = ""
    if form == "short":
        form_line = ("BENTUK: kueri HARUS SANGAT PENDEK, 2-6 kata, gaya potongan chat "
                     "yang miskin konteks dan boleh sarat singkatan (pola: 'bedanya X "
                     "sama Y?', 'cara itung X yg bener?', 'X itu wajib ga sih?'). "
                     "Justru keterbatasan konteksnya yang menguji router; tapi kueri "
                     "tetap harus terjawab dari chunk.\n")
    return f"""Kamu membuat kueri benchmark untuk menguji ROUTER factual-vs-socratic sistem RAG.

KELAS: Adversarial Routing (kasus batas). Konten chunk bergaya {style}.
TUGAS: {instruction}. Satu kueri, bahasa Indonesia, natural.
{form_line}PENTING: syarat penjangkaran ke chunk TIDAK boleh menghilangkan tension kasus
batasnya - pertahankan gaya bertanya yang menipu router (bunyi vs kebutuhan
jawaban harus tetap bertolak belakang).
reasoning_chain menjelaskan kenapa kueri ini kasus batas untuk router.
gold_relevant_chunks = chunk yang diberikan.

{_chunk_block(bundle.chunks)}

{_difficulty_line(bundle)}{_style_line(bundle)}{OUTPUT_CONTRACT}"""
