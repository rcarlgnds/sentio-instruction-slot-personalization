"""Scores one cell with Qwen3-Reranker or zerank-2. Needs a GPU.

A cell is one model on one track with one input setup: the profile in the instruction slot, a program and
semester label in the passage, both, or neither (--help lists the options). It writes per-row scores and
a manifest with the exact prompt. The paper's zerank-2 numbers use the vendor's CrossEncoder path
(--family zerank2official).
"""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import sys  # noqa: E402
for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
from label_map import candidate_meta, load_id_label_map, source_summary
from model_pins import revision_for
ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
Q2 = ROOT / "study/channel/results"
SHARED = ROOT / "study/_shared"
SOURCE_PROFILE = ROOT / "system/data/rads_sample/queries.jsonl"

TASK_DEFAULT = ("Given a web search query, retrieve relevant passages that "
                "answer the query")
VENDOR_MEMORY_PREFIX = ("User memory: the reader is a student of the "
                        "{dept} degree program, currently in semester {sem}. "
                        "Prefer passages from that program's own course "
                        "materials. ")


def profile_instruction(p: dict) -> str:
    dept = p.get("department") or p.get("department_name") or "unknown"
    sem = p.get("current_semester") or p.get("semester") or "?"
    return (f"Given a query from a {dept} student in semester {sem}, retrieve "
            f"passages from that student's own study program that answer the "
            f"query at a level appropriate for semester {sem}")


def profile_instruction_id(p: dict) -> str:
    dept = p.get("department") or p.get("department_name") or "unknown"
    sem = p.get("current_semester") or p.get("semester") or "?"
    return (f"Diberikan pertanyaan dari mahasiswa program studi {dept} semester {sem}, "
            f"ambil bagian teks dari program studi mahasiswa itu sendiri yang menjawab "
            f"pertanyaan tersebut pada tingkat yang sesuai untuk semester {sem}")


def vendor_memory(p: dict) -> str:
    dept = p.get("department") or p.get("department_name") or "unknown"
    sem = p.get("current_semester") or p.get("semester") or "?"
    return VENDOR_MEMORY_PREFIX.format(dept=dept, sem=sem)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def manifest_output_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(Q2.parent))
    except ValueError:
        return str(p.resolve())


CORPUS = ROOT / "system/data/benchmark_release/corpus/corpus.jsonl"


def load_metadata_map() -> dict:
    mapping = {}
    for l in CORPUS.open():
        r = json.loads(l)
        k = (r.get("text") or "")[:180].strip()
        if k:
            mapping.setdefault(k, r["metadata"])
    return mapping


def source_label(m: dict, field: str = "both") -> str:
    if field == "program":
        return f"[Source: {m.get('department')} degree program textbook]"
    if field == "semester":
        return f"[Source: degree program textbook, semester {m.get('semester')}]"
    return (f"[Source: {m.get('department')} degree program textbook, "
            f"semester {m.get('semester')}]")


PROGRAM_NAMES = {
    "CSCI": "ilmu komputer", "DTSC": "data science", "STAT": "statistika",
    "ISYS": "sistem informasi", "BANA": "business analytics",
    "MGMT": "manajemen", "MKTG": "pemasaran", "ACCT": "akuntansi",
    "CHIN": "sastra china",
}

OPAQUE_CODES = {d: f"PRG-{i:02d}" for i, d in enumerate(sorted(PROGRAM_NAMES), start=1)}


PROGRAM_NAMES_EN = {
    "CSCI": "Computer Science", "DTSC": "Data Science", "STAT": "Statistics",
    "ISYS": "Information Systems", "BANA": "Business Analytics",
    "MGMT": "Management", "MKTG": "Marketing", "ACCT": "Accounting",
    "CHIN": "Chinese Literature",
}

SEMESTER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four",
                 5: "five", 6: "six", 7: "seven", 8: "eight"}

OPAQUE_SEMESTER_CODES = {n: f"T-{chr(ord('A') + n - 1)}" for n in range(1, 9)}


def profile_instruction_name_en(p: dict) -> str:
    dept = p.get("department") or p.get("department_name") or "unknown"
    sem = p.get("current_semester") or p.get("semester") or "?"
    name = PROGRAM_NAMES_EN.get(dept, dept)
    return (f"Given a query from a {name} student in semester {sem}, retrieve "
            f"passages from that student's own study program that answer the "
            f"query at a level appropriate for semester {sem}")


def label_program_en(m: dict) -> str:
    return f"[Source: {PROGRAM_NAMES_EN.get(m.get('department'), m.get('department'))} degree program textbook]"


def label_semester_style(m: dict, style: str) -> str:
    try:
        n = int(m.get("semester"))
    except (TypeError, ValueError):
        n = 0
    if style == "kata":
        value = SEMESTER_WORDS.get(n, str(m.get("semester")))
    elif style == "opak":
        value = OPAQUE_SEMESTER_CODES.get(n, "T-Z")
    else:
        raise ValueError(f"unknown semester style: {style}")
    return f"[Source: degree program textbook, semester {value}]"


def label_program_style(m: dict, style: str) -> str:
    dept = m.get("department")
    if style == "opak":
        return f"[Source: {OPAQUE_CODES.get(dept, 'PRG-00')} degree program textbook]"
    name = PROGRAM_NAMES.get(dept, dept)
    if style == "alami":
        return f"Program studi {name}."
    if style == "nama":
        return f"[Source: {name} degree program textbook]"
    raise ValueError(f"unknown label style: {style}")


def wrong_dept(original_dept: str, text: str, all_depts: list) -> str:
    other = sorted(d for d in all_depts if d and d != original_dept)
    if not other:
        return original_dept
    h = hashlib.sha256(("dept_salah_v1|" + text[:180]).encode()).hexdigest()
    return other[int(h[:8], 16) % len(other)]


def wrong_source_label(m: dict, text: str, all_depts: list) -> str:
    fake = dict(m)
    fake["department"] = wrong_dept(m.get("department"), text, all_depts)
    return source_label(fake)


def load_global_profiles() -> dict:
    prof = {}
    if SOURCE_PROFILE.exists():
        for line in SOURCE_PROFILE.open():
            r = json.loads(line)
            prof[r["query_id"]] = r.get("student_profile") or {}
    return prof


def seal_gate(query_ids, allow_holdout=False):
    assignment = json.loads((SHARED / "split_manifest.json").read_text())["assignment"]
    leak = [q for q in query_ids if assignment.get(q) == "test_holdout"]
    if leak and allow_holdout:
        print(f"SEAL OPENED AGAIN: {len(leak)} test_holdout query_id in the pool; "
              f"recorded in the manifest as segel_holdout.dibuka_kedua", flush=True)
        return len(leak)
    if leak:
        raise SystemExit(
            f"ABORT SEAL: {len(leak)} test_holdout query_id in the pool "
            f"(e.g. {leak[:3]}). A post-hoc arm may not touch the holdout.")
    return 0


class Zerank2Scorer:
    def __init__(self, name: str, dtype: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download
        self.torch = torch
        d = Path(snapshot_download(name, revision=revision_for(name)))
        cfg = json.loads((d / "1_LogitScore/config.json").read_text())
        self.true_id = cfg["true_token_id"]
        self.false_id = cfg.get("false_token_id")
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision_for(name), padding_side="left")
        kw = ({"torch_dtype": getattr(torch, dtype)} if dtype != "int8"
              else {"load_in_8bit": True})
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision_for(name), **kw)
        if dtype != "int8":
            self.model = self.model.cuda()
        self.model.eval()
        self.template_info = {
            "peran": "query -> system, document -> user, then the generation prompt",
            "true_token_id": self.true_id, "false_token_id": self.false_id,
            "sumber": "chat_template.jinja + 1_LogitScore/config.json of the official repo",
        }

    def prompt(self, query: str, docs: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "query", "content": query},
             {"role": "document", "content": docs}],
            tokenize=False, add_generation_prompt=True)

    def score(self, query: str, docs: list, batch: int, max_len: int) -> list:
        torch = self.torch
        output = []
        with torch.no_grad():
            for i in range(0, len(docs), batch):
                text = [self.prompt(query, d) for d in docs[i:i + batch]]
                enc = self.tok(text, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_len)
                enc = {k: v.cuda() for k, v in enc.items()}
                logits = self.model(**enc).logits[:, -1, :].float()
                if self.false_id is None:
                    s = logits[:, self.true_id]
                else:
                    duo = torch.stack([logits[:, self.false_id],
                                       logits[:, self.true_id]], dim=1)
                    s = torch.nn.functional.log_softmax(duo, dim=1)[:, 1].exp()
                output.extend(s.tolist())
        return output


class Qwen3RRScorer:
    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the "
              "requirements based on the Query and the Instruct provided. Note "
              "that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
              "<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, name: str, dtype: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision_for(name), padding_side="left")
        if dtype == "int8":
            self.model = AutoModelForCausalLM.from_pretrained(
                name, revision=revision_for(name), load_in_8bit=True)
        elif dtype == "int4":
            from transformers import BitsAndBytesConfig
            self.model = AutoModelForCausalLM.from_pretrained(
                name, revision=revision_for(name), quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"))
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                name, revision=revision_for(name), torch_dtype=getattr(torch, dtype)).cuda()
        self.model.eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suf = self.tok.encode(self.SUFFIX, add_special_tokens=False)
        self.instruction = TASK_DEFAULT
        self.template_info = {"format": "<Instruct>/<Query>/<Document>",
                              "skor": "softmax of the yes and no logits",
                              "sumber": "official Qwen3-Reranker README"}

    def score(self, query: str, docs: list, batch: int, max_len: int) -> list:
        torch = self.torch
        pairs = [f"<Instruct>: {self.instruction}\n<Query>: {query}\n"
                 f"<Document>: {d}" for d in docs]
        output = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch):
                chunk = pairs[i:i + batch]
                enc = [self.pre + self.tok.encode(p, add_special_tokens=False)[
                    :max_len - len(self.pre) - len(self.suf)] + self.suf
                    for p in chunk]
                m = max(len(e) for e in enc)
                ids = torch.full((len(enc), m), self.tok.pad_token_id,
                                 dtype=torch.long)
                att = torch.zeros((len(enc), m), dtype=torch.long)
                for j, e in enumerate(enc):
                    ids[j, m - len(e):] = torch.tensor(e)
                    att[j, m - len(e):] = 1
                logits = self.model(input_ids=ids.cuda(),
                                    attention_mask=att.cuda()).logits[:, -1, :]
                duo = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1)
                output.extend(torch.nn.functional.log_softmax(
                    duo.float(), dim=1)[:, 1].exp().tolist())
        return output


class Zerank2OfficialScorer:
    def __init__(self, name: str, dtype: str):
        import torch
        import sentence_transformers
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(name, revision=revision_for(name), model_kwargs={"torch_dtype": getattr(torch, dtype)},
                                  max_length=1536, device="cuda")
        self.template_info = {
            "jalur": "sentence_transformers.CrossEncoder.predict(), kode vendor",
            "versi_sentence_transformers": sentence_transformers.__version__,
            "sumber": "chat_template.jinja + 1_LogitScore/config.json of the official repo, applied by the "
                      "vendor library",
        }

    def score(self, query: str, docs: list, batch: int, max_len: int) -> list:
        if self.model.max_length != max_len:
            self.model.max_length = max_len
        return [float(x) for x in self.model.predict([(query, d) for d in docs],
                                                     batch_size=batch)]


FAMILY = {"zerank2": Zerank2Scorer, "qwen3rr": Qwen3RRScorer,
            "zerank2official": Zerank2OfficialScorer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(FAMILY), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pools", required=True)
    ap.add_argument("--instruction", choices=["default", "profile", "kondisi",
                                             "profile_id", "profile_nama_en",
                                             "profile_program", "profile_semester"],
                    default="default",
                    help="kondisi = use the instruksi_kondisi field of the pool (the worked-example control, "
                         "where the two rows of a pair get different instructions)")
    ap.add_argument("--doc-metadata", action="store_true",
                    help="put the source label (program and semester) into the text of each candidate. DITS reads "
                         "this metadata directly, and the flag moves it into the text that the reranker reads. If "
                         "the score rises, the model can use the label, so its failure without the flag comes "
                         "from where the label sits.")
    ap.add_argument("--doc-metadata-salah", action="store_true",
                    help="negative control for --doc-metadata: the label keeps the same format but names a wrong "
                         "department, picked from the sha256 of the text. On its own it keeps the right "
                         "semester. It separates what the label says from the fact that a label is there. A "
                         "rise that remains with this label can still come from the semester (Section 4.4 of "
                         "the paper). Use it with --doc-metadata.")
    ap.add_argument("--vendor-memory", action="store_true",
                    help="put a user-memory block in the vendor's style before the profile instruction, the setup "
                         "the zerank-2 vendor suggests")
    ap.add_argument("--slot-format", choices=["plain", "tag", "labeled"],
                    default="plain",
                    help="how zerank-2 gets the instruction. plain (default) = the instruction, a blank line, "
                         "then Query:. tag = the vendor format <query>..</query>\\n<instruction>..</instruction>. "
                         "labeled = the vendor's second format 'Query: ..\\nInstructions: ..'. Source: "
                         "zeroentropy.dev/articles/prompting-best-practices-for-instruction-following-rerankers "
                         "(22 Jan 2026).")
    ap.add_argument("--doc-label-gaya", choices=["nama", "alami", "opak"],
                    default=None,
                    help="inference-distance ladder: replace the program marker in the label with the program "
                         "name, a syllabus-style header, or an opaque institution code. The label holds the "
                         "program only, without the semester, so the ladder has room to move. Needs "
                         "--doc-metadata.")
    ap.add_argument("--doc-label-en", choices=["en", "id", "kode"], default=None,
                    help="surface ladder on the program field, run with --instruction profile_nama_en so the "
                         "instruction names the program in English. en = the same English name in the label, id = "
                         "the Indonesian name, kode = the department code. Program only. Needs --doc-metadata.")
    ap.add_argument("--doc-label-sem", choices=["kata", "opak"], default=None,
                    help="surface ladder on the semester field, with the instruction unchanged. kata = the number "
                         "written as an English word, opak = a lettered term code as the negative control. "
                         "Semester only. Needs --doc-metadata.")
    ap.add_argument("--doc-metadata-field", choices=["both", "program",
                    "semester"], default="both",
                    help="what the source label holds in the single-field control: both (default) = the full "
                         "label, program = no semester, semester = no program name. This shows which field of the "
                         "label the reranker reads.")
    ap.add_argument("--profil-tetap-prodi", choices=["CSCI", "STAT"], default=None,
                    help="orthogonal design: hold the program in the instruction at one "
                         "value for both rows of a pair, so only the semester "
                         "varies. Paired track and --instruction profile only.")
    ap.add_argument("--profil-tetap-semester", type=int, choices=[1, 2], default=None,
                    help="orthogonal design: hold the semester in the instruction at one "
                         "value for both rows of a pair, so only the program "
                         "varies. Paired track and --instruction profile only.")
    ap.add_argument("--doc-metadata-salah-semester", action="store_true",
                    help="factorial labels: the label keeps its frame but the "
                         "semester is wrong (rule copied from --doc-metadata-salah). "
                         "Together with --doc-metadata-salah both values are wrong.")
    ap.add_argument("--doc-label-acak", action="store_true",
                    help="matched negative control: program and semester "
                         "drawn independently of the passage's book.")
    ap.add_argument("--doc-label-seragam", action="store_true",
                    help="presence control: every candidate carries the "
                         "asker's own program and semester.")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "int8", "int4"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-holdout", action="store_true",
                    help="score test_holdout queries; records a second seal opening in the manifest")
    ap.add_argument("--min-free-mib", type=int, default=1800)
    ap.add_argument("--peta-label", choices=["id", "teks"], default="id",
                    help="source of the label metadata: the candidate's own record (default) or a match on the "
                         "first 180 characters of its text")
    ap.add_argument("--out-dir", default=None,
                    help="output folder for the jsonl and manifest; default raw/")
    args = ap.parse_args()

    import subprocess
    free = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True).split()[0])
    print(f"free VRAM before load: {free} MiB")
    if free < args.min_free_mib + 2000:
        raise SystemExit(f"ABORT VRAM: {free} MiB free is too tight "
                         f"(project rule leaves {args.min_free_mib} MiB).")

    pools = [json.loads(l) for l in Path(args.pools).open() if l.strip()]
    n_holdout = seal_gate([r["query_id"] for r in pools], args.allow_holdout)
    if args.limit:
        pools = pools[:args.limit]
    global_profile = load_global_profiles() if args.instruction in (
        "profile", "profile_id", "profile_nama_en",
        "profile_program", "profile_semester") else {}
    if args.instruction == "kondisi" and not any(
            json.loads(l).get("instruksi_kondisi") for l in Path(args.pools).open()):
        raise SystemExit("ABORT: --instruction kondisi but the pool has no "
                         "instruksi_kondisi field.")
    if args.doc_metadata_salah and not args.doc_metadata:
        raise SystemExit("ABORT: --doc-metadata-salah needs --doc-metadata, because it changes what the label "
                         "says and does not add a label by itself.")
    if args.slot_format != "plain" and (args.family not in ("zerank2", "zerank2official")
                                        or args.instruction != "profile"):
        raise SystemExit("ABORT: --slot-format tag/labeled is only for "
                         "family zerank2 with --instruction profile "
                         "(the format is zerank-2 vendor prompt convention).")
    if args.doc_metadata_field != "both" and not args.doc_metadata:
        raise SystemExit("ABORT: --doc-metadata-field requires --doc-metadata.")
    if args.doc_label_gaya and not args.doc_metadata:
        raise SystemExit("ABORT: --doc-label-gaya requires --doc-metadata.")
    for flag_name, value in (("--doc-label-en", args.doc_label_en),
                             ("--doc-label-sem", args.doc_label_sem)):
        if value and not args.doc_metadata:
            raise SystemExit(f"ABORT: {flag_name} requires --doc-metadata.")
        if value and args.doc_metadata_field != "both":
            raise SystemExit(f"ABORT: {flag_name} already fixes the label field; "
                             "do not combine it with --doc-metadata-field.")
        if value and args.doc_label_gaya:
            raise SystemExit(f"ABORT: {flag_name} and --doc-label-gaya both set "
                             "the label; pick one.")
    if args.doc_label_en and args.doc_label_sem:
        raise SystemExit("ABORT: --doc-label-en and --doc-label-sem both set the "
                         "label; pick one.")
    if args.doc_label_en and args.instruction != "profile_nama_en":
        raise SystemExit("ABORT: --doc-label-en is only interpretable with --instruction profile_nama_en, because "
                         "the contrast it measures is the label language against an instruction that names the "
                         "program in English.")
    if args.doc_label_gaya and args.doc_metadata_field != "both":
        raise SystemExit("ABORT: --doc-label-gaya already fixes the label to "
                         "program only; do not combine it with "
                         "--doc-metadata-field.")
    if args.doc_metadata_field != "both" and args.doc_metadata_salah:
        raise SystemExit("ABORT: the single-field control and the wrong label "
                         "must not be combined; one manipulation per cell.")
    from revision_labels import (revision_label_rules, program_only_instruction,
                             semester_only_instruction, random_label, uniform_label,
                             wrong_both_source_label, wrong_semester_source_label,
                             fixed_profile, observed_semesters)
    fixed = args.profil_tetap_prodi is not None or args.profil_tetap_semester is not None
    if args.profil_tetap_prodi is not None and args.profil_tetap_semester is not None:
        raise SystemExit("ABORT: hold one field fixed per cell, not both.")
    if fixed and args.instruction != "profile":
        raise SystemExit("ABORT: --profil-tetap-* needs --instruction profile.")
    if fixed and args.vendor_memory:
        raise SystemExit("ABORT: --profil-tetap-* and --vendor-memory both rewrite "
                         "the instruction; one manipulation per cell.")
    if fixed:
        program_pool = {(r.get("profile") or {}).get("department_name")
                      or (r.get("profile") or {}).get("department") for r in pools}
        if not program_pool <= {"CSCI", "STAT"}:
            raise SystemExit("ABORT: --profil-tetap-* is defined for the paired track "
                             f"only; this pool holds programs {sorted(map(str, program_pool))}.")
    if args.instruction in ("profile_program", "profile_semester") and args.vendor_memory:
        raise SystemExit("ABORT: the vendor memory block names both fields, which "
                         "would undo a single-field instruction.")
    label_r1 = [n for n, v in (("--doc-metadata-salah-semester", args.doc_metadata_salah_semester),
                               ("--doc-label-acak", args.doc_label_acak),
                               ("--doc-label-seragam", args.doc_label_seragam)) if v]
    if label_r1 and not args.doc_metadata:
        raise SystemExit(f"ABORT: {label_r1[0]} requires --doc-metadata.")
    if label_r1 and (args.doc_metadata_field != "both" or args.doc_label_gaya
                     or args.doc_label_en or args.doc_label_sem):
        raise SystemExit(f"ABORT: {label_r1[0]} sets the label itself; do not combine it "
                         "with --doc-metadata-field, --doc-label-gaya, --doc-label-en or "
                         "--doc-label-sem.")
    if (args.doc_label_acak or args.doc_label_seragam) and len(label_r1) > 1:
        raise SystemExit("ABORT: the random and uniform labels stand alone; one "
                         "manipulation per cell.")
    if (args.doc_label_acak or args.doc_label_seragam) and args.doc_metadata_salah:
        raise SystemExit("ABORT: the random and uniform labels stand alone; do not "
                         "combine them with --doc-metadata-salah.")
    if args.doc_label_seragam and args.instruction != "profile":
        raise SystemExit("ABORT: --doc-label-seragam is defined with the full "
                         "profile in the slot (--instruction profile).")
    meta_map = ((load_id_label_map() if args.peta_label == "id"
                  else load_metadata_map()) if args.doc_metadata else {})

    def find_meta(c):
        return (candidate_meta(meta_map, c) if args.peta_label == "id"
                else meta_map.get((c["text"] or "")[:180].strip()))
    all_depts = sorted({(m or {}).get("department") for m in meta_map.values()
                         if (m or {}).get("department")})
    if args.doc_metadata_salah and len(all_depts) < 2:
        raise SystemExit(f"ABORT: only {len(all_depts)} departments in the "
                         "metadata map, a wrong label cannot be formed.")
    all_sems = observed_semesters(meta_map) if args.doc_metadata else []
    if (args.doc_metadata_salah_semester or args.doc_label_acak) and len(all_sems) < 2:
        raise SystemExit(f"ABORT: only {len(all_sems)} semester value(s) in the label "
                         "map; a wrong or random semester cannot be formed.")
    unmapped = 0
    n_flipped = 0
    n_r1 = {"semsalah": 0, "duasalah": 0, "acak": 0, "seragam": 0}

    lock = json.loads((SHARED / "config_lock.json").read_text())
    if args.doc_metadata_salah and args.doc_metadata_salah_semester:
        wrong_token = "duasalah"
    elif args.doc_metadata_salah_semester:
        wrong_token = "semsalah"
    elif args.doc_metadata_salah:
        wrong_token = "salah"
    else:
        wrong_token = ""
    out_name = (f"scores_{args.family}_{Path(args.model).name}_{args.tag}_"
                f"{args.instruction}{'_vmem' if args.vendor_memory else ''}"
                f"{'_tetapprodi' + args.profil_tetap_prodi if args.profil_tetap_prodi else ''}"
                f"{'_tetapsem' + str(args.profil_tetap_semester) if args.profil_tetap_semester else ''}"
                f"{'_docmeta' if args.doc_metadata else ''}"
                f"{wrong_token}"
                f"{'acak' if args.doc_label_acak else ''}"
                f"{'seragam' if args.doc_label_seragam else ''}"
                f"{'_fmt' + args.slot_format if args.slot_format != 'plain' else ''}"
                f"{'_field' + args.doc_metadata_field if args.doc_metadata_field != 'both' else ''}")
    raw_out = Path(args.out_dir) if args.out_dir else (Q2 / "raw")
    raw_out.mkdir(parents=True, exist_ok=True)
    out_path = raw_out / f"{out_name}.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        n_present = sum(1 for _ in out_path.open())
        if n_present >= len(pools):
            print(f"SKIP (idempotent): {out_path.name} already has {n_present} rows")
            return
        print(f"resuming from the {n_present} rows that already exist")
        pools = pools[n_present:]
        mode = "a"
    else:
        mode = "w"

    import fcntl
    path_key = Q2 / "raw/.gpu.lock"
    path_key.parent.mkdir(parents=True, exist_ok=True)
    key = open(path_key, "w")
    print("waiting for the GPU lock ...", flush=True)
    fcntl.flock(key, fcntl.LOCK_EX)
    key.write(f"{os.getpid()} {out_name}\n")
    key.flush()
    print("GPU lock acquired", flush=True)

    import torch
    print(f"loading {args.model} ({args.dtype}) family {args.family} ...")
    t_load = time.perf_counter()
    scorer = FAMILY[args.family](args.model, args.dtype)
    torch.cuda.reset_peak_memory_stats()
    print(f"loaded in {time.perf_counter()-t_load:.0f}s; "
          f"process VRAM {torch.cuda.memory_allocated()/2**20:.0f} MiB")

    t0 = time.perf_counter()
    n = 0
    with out_path.open(mode) as f:
        for row in pools:
            query = row["query"]
            if args.instruction == "kondisi":
                inst = row.get("instruksi_kondisi") or TASK_DEFAULT
                if args.family == "qwen3rr":
                    scorer.instruction = inst
                else:
                    query = f"{inst}\n\nQuery: {row['query']}"
            elif args.instruction in ("profile", "profile_id", "profile_nama_en",
                                      "profile_program", "profile_semester"):
                p = row.get("profile") or global_profile.get(row["query_id"], {})
                if fixed:
                    p = fixed_profile(p, program=args.profil_tetap_prodi,
                                     semester=args.profil_tetap_semester)
                if args.instruction == "profile_nama_en":
                    inst = profile_instruction_name_en(p)
                elif args.instruction == "profile_id":
                    inst = profile_instruction_id(p)
                elif args.instruction == "profile_program":
                    inst = program_only_instruction(p)
                elif args.instruction == "profile_semester":
                    inst = semester_only_instruction(p)
                else:
                    inst = profile_instruction(p)
                if args.vendor_memory:
                    inst = vendor_memory(p) + inst
                if args.family == "qwen3rr":
                    scorer.instruction = inst
                elif args.slot_format == "tag":
                    query = (f"<query>{row['query']}</query>\n"
                             f"<instruction>{inst}</instruction>")
                elif args.slot_format == "labeled":
                    query = f"Query: {row['query']}\nInstructions: {inst}"
                else:
                    query = f"{inst}\n\nQuery: {row['query']}"
            elif args.family != "qwen3rr":
                query = f"{TASK_DEFAULT}\n\nQuery: {row['query']}"
            if args.doc_metadata:
                doc = []
                p_rows = row.get("profile") or global_profile.get(row["query_id"], {})
                for c in row["cands"]:
                    m = find_meta(c)
                    if m is None:
                        unmapped += 1
                        pre = ""
                    elif args.doc_label_acak:
                        pre = random_label(c["text"] or "", all_depts, all_sems) + "\n\n"
                        n_r1["acak"] += 1
                    elif args.doc_label_seragam:
                        if not (p_rows.get("department") or p_rows.get("department_name")):
                            raise SystemExit(f"ABORT: row {row['query_id']} has no "
                                             "profile, the uniform label cannot be built.")
                        pre = uniform_label(p_rows) + "\n\n"
                        n_r1["seragam"] += 1
                    elif args.doc_metadata_salah and args.doc_metadata_salah_semester:
                        pre = wrong_both_source_label(m, c["text"] or "", all_depts,
                                                    all_sems) + "\n\n"
                        n_r1["duasalah"] += 1
                    elif args.doc_metadata_salah_semester:
                        pre = wrong_semester_source_label(m, c["text"] or "", all_sems) + "\n\n"
                        n_r1["semsalah"] += 1
                    elif args.doc_metadata_salah:
                        lbl = wrong_source_label(m, c["text"] or "", all_depts)
                        n_flipped += 1
                        pre = lbl + "\n\n"
                    elif args.doc_label_gaya:
                        pre = label_program_style(m, args.doc_label_gaya) + "\n\n"
                    elif args.doc_label_en == "en":
                        pre = label_program_en(m) + "\n\n"
                    elif args.doc_label_en == "id":
                        pre = label_program_style(m, "nama") + "\n\n"
                    elif args.doc_label_en == "kode":
                        pre = source_label(m, "program") + "\n\n"
                    elif args.doc_label_sem:
                        pre = label_semester_style(m, args.doc_label_sem) + "\n\n"
                    else:
                        pre = source_label(m, args.doc_metadata_field) + "\n\n"
                    doc.append(pre + c["text"][:4000])
            else:
                doc = [c["text"][:4000] for c in row["cands"]]
            free_now = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"], text=True).split()[0])
            if free_now < args.min_free_mib:
                raise SystemExit(
                    f"ABORT VRAM WHILE RUNNING: free {free_now} MiB < "
                    f"{args.min_free_mib} at query {n+1}. Lower "
                    f"--batch or --max-len, or use --dtype int8.")
            t1 = time.perf_counter()
            score = scorer.score(query, doc, args.batch, args.max_len)
            dt = time.perf_counter() - t1
            order = sorted(zip((c["id"] for c in row["cands"]), score),
                          key=lambda x: (-x[1], x[0]))
            f.write(json.dumps({
                "query_id": row["query_id"], "gold": row["gold"],
                "counterpart": row.get("counterpart"),
                "ranking": [i for i, _ in order],
                "scores": [round(s, 6) for _, s in order],
                "latency_s": round(dt, 4),
            }) + "\n")
            f.flush()
            torch.cuda.empty_cache()
            n += 1
            if n % 25 == 0:
                print(f"  {n}/{len(pools)} {time.perf_counter()-t0:.0f}s "
                      f"vram_peak={torch.cuda.max_memory_allocated()/2**20:.0f}MiB",
                      flush=True)

    meta = {
        "run_id": out_name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/2_scoring/score_arms.py",
        "perintah_persis": " ".join(["python score_arms.py"] + os.sys.argv[1:]),
        "config_version": lock["config_version"],
        "status_arm": "post hoc, outside the pre-registered family of eight",
        "model": args.model, "keluarga": args.family, "dtype": args.dtype,
        "instruction": args.instruction, "vendor_memory": args.vendor_memory,
        "instruksi_profil_id_kerangka": (profile_instruction_id({"department": "<DEPT>", "current_semester": "<N>"})
                                         if args.instruction == "profile_id" else None),
        "slot_format": args.slot_format,
        "slot_format_sumber": (None if args.slot_format == "plain" else
                               "zeroentropy.dev, Prompting Best Practices For "
                               "Instruction-Following Rerankers, 22 Jan 2026 "
                               "(tag = <query>..</query>\\n<instruction>.."
                               "</instruction>; labeled = 'Query: ..\\n"
                               "Instructions: ..')"),
        "doc_metadata_injected": args.doc_metadata,
        "doc_metadata_label_salah": args.doc_metadata_salah,
        "doc_metadata_field": args.doc_metadata_field,
        "doc_label_gaya": args.doc_label_gaya,
        "doc_label_en": args.doc_label_en,
        "doc_label_sem": args.doc_label_sem,
        "r1_ijies": {
            "profil_tetap": ({"prodi": args.profil_tetap_prodi,
                              "semester": args.profil_tetap_semester} if fixed else None),
            "instruksi_kerangka": (
                program_only_instruction({"department": "<DEPT>"})
                if args.instruction == "profile_program" else
                semester_only_instruction({"current_semester": "<N>"})
                if args.instruction == "profile_semester" else None),
            "doc_metadata_salah_semester": args.doc_metadata_salah_semester,
            "doc_label_acak": args.doc_label_acak,
            "doc_label_seragam": args.doc_label_seragam,
            "aturan_label": (revision_label_rules() if label_r1 else None),
            "kolam_semester": all_sems if label_r1 else None,
            "kolam_prodi": all_depts if label_r1 else None,
            "n_label": n_r1,
        },
        "format_label_en": (label_program_en({"department": "STAT"})
                            if args.doc_label_en == "en" else None),
        "format_label_semester": (
            label_semester_style({"semester": 2}, args.doc_label_sem)
            if args.doc_label_sem else None),
        "format_label_program_gaya": (
            label_program_style({"department": "STAT"}, args.doc_label_gaya)
            if args.doc_label_gaya else None),
        "format_label_sumber": (
            None if not args.doc_metadata else
            "[Source: <DEPT> degree program textbook]"
            if args.doc_metadata_field == "program" else
            "[Source: degree program textbook, semester <N>]"
            if args.doc_metadata_field == "semester" else
            "[Source: <DEPT> degree program textbook, semester <N>]"),
        "kontrol_label_salah": ({
            "tujuan": "separates the content of the label from its presence",
            "manipulasi": "only the department field replaced; the semester stays right, so one factor is "
                          "manipulated",
            "pemilihan": "deterministic: sha256('dept_salah_v1|' + text[:180]) modulo the sorted list of the "
                         "other programs",
            "kolam_prodi": all_depts,
            "n_label_dibalik": n_flipped,
            "cara_baca": "if the scores still jump as in the right-docmeta condition, the jump is a token-length "
                         "artifact and not evidence that the reranker reads the metadata",
        } if args.doc_metadata_salah and not args.doc_metadata_salah_semester else None),
        "n_kandidat_tak_terpetakan": unmapped,
        "peta_label_sumber": (source_summary() if args.peta_label == "id"
                              else {"peta_label": "teks"}),
        "instruksi_default_verbatim": TASK_DEFAULT,
        "kontrak_inferensi": scorer.template_info,
        "n": n, "durasi_s": round(time.perf_counter() - t0, 1),
        "latensi_mean_s": None, "vram_peak_mib": round(
            torch.cuda.max_memory_allocated() / 2**20, 1),
        "sumber": {"pools": sha(Path(args.pools))},
        "output": {"path": manifest_output_path(out_path),
                   "sha256": sha(out_path)},
        "segel_holdout": ({"dibuka_kedua": True, "n_kueri_holdout": n_holdout,
                           "alasan": "scores the post hoc arms on the sealed population after every analysis was frozen; no "
                                     "configuration, threshold, or weight changed"}
                          if n_holdout else None),
        "label_source": "deterministic (model scores; temperature does not apply)",
    }
    lat = [json.loads(l)["latency_s"] for l in out_path.open()]
    meta["latensi_mean_s"] = round(sum(lat) / len(lat), 4)
    lat_s = sorted(lat)
    meta["latensi_p95_s"] = round(lat_s[int(0.95 * len(lat_s)) - 1], 4)
    (raw_out / f"{out_name}.manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps({k: meta[k] for k in
                      ("n", "durasi_s", "latensi_mean_s", "latensi_p95_s",
                       "vram_peak_mib")}))
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
