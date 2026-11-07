import hashlib
import json
import os

from app.benchmark.rads_synthesis.chunk_selector import ChunkSelector
from app.benchmark.rads_synthesis.corpus import load_child_chunks, load_course_digraph, load_course_graph
from app.benchmark.rads_synthesis.difficulty import STYLES, derive_for_query, derive_raw_for_query
from app.benchmark.rads_synthesis.generator import BatchFileBackend, OllamaBackend, validate_output
from app.utils.logging import get_logger
from app.benchmark.rads_synthesis.gold import assign_splits
from app.benchmark.rads_synthesis.models import BenchmarkQuery, DIFFICULTIES
from app.benchmark.rads_synthesis.templates import TEMPLATE_VERSION, build_prompt
from app.benchmark.sanitizer import ChunkSanitizer
from app.domain.models import StudentContext

REJECT_BUCKETS = ("duplicate", "ood", "gold", "class_fit", "meta_language", "error")

GRAPH_VERSION = "dag_v2.2-b30"

CORPUS_VERSION = "corpus_v2-30books-66669"

log = get_logger(__name__)


def batch_id_for(bundle, seed: int) -> str:
    chunk_ids = ",".join(sorted(c.chunk_id for c in bundle.chunks))
    digest = hashlib.sha1(f"{TEMPLATE_VERSION}|{chunk_ids}".encode()).hexdigest()[:10]
    return f"{bundle.class_label[:4]}-s{seed}-{TEMPLATE_VERSION}-{digest}"


def student_profile_for(bundle, gold_chunk_ids=None) -> dict:
    chunks = bundle.chunks
    label = bundle.class_label
    if label == "prerequisite_chain_depth":
        advanced = max(chunks, key=lambda c: c.semester)
        foundation = min(chunks, key=lambda c: c.semester)
        return {"department_id": advanced.department_id,
                "department_name": advanced.department_name,
                "current_semester": advanced.semester or 1,
                "enrolled_courses": [foundation.course_id]}
    anchor = chunks[0]
    if label == "multi_disciplinary_ambiguity" and gold_chunk_ids:
        by_id = {c.chunk_id: c for c in chunks}
        for gid in gold_chunk_ids:
            if gid in by_id:
                anchor = by_id[gid]
                break
    semester = (bundle.class_metadata.get("student_semester")
                if label == "temporal_scope_drift" else anchor.semester) or 1
    return {"department_id": anchor.department_id,
            "department_name": anchor.department_name,
            "current_semester": semester,
            "enrolled_courses": []}


def _sanitized_copy(bundle, sanitizer: ChunkSanitizer):
    from copy import copy
    clean_chunks = []
    for c in bundle.chunks:
        clean = copy(c)
        clean.raw_text = sanitizer.sanitize(c.raw_text).clean_text
        clean_chunks.append(clean)
    clean_bundle = copy(bundle)
    clean_bundle.chunks = clean_chunks
    return clean_bundle


def _reject_bucket(qc: dict) -> str:
    if qc.get("dedup") == "duplicate":
        return "duplicate"
    if qc.get("class_fit") == "term_missing":
        return "class_fit"
    if qc.get("meta") == "meta_language":
        return "meta_language"
    if qc.get("gold") == "not_retrievable":
        return "gold"
    return "ood"


class RadsPipeline:
    def __init__(self, seed: int = 42, out_dir: str = "data/rads",
                 hard_mix: float = 0.0):
        self.seed = seed
        self.hard_mix = hard_mix
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.queries_path = os.path.join(out_dir, "queries.jsonl")
        self.rejects_path = os.path.join(out_dir, "rejects.jsonl")
        self.report_path = os.path.join(out_dir, "rads_report.json")
        self.sanitizer = ChunkSanitizer()

    def _accepted_rows(self) -> list:
        rows = []
        corrupt = 0
        if os.path.exists(self.queries_path):
            with open(self.queries_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        corrupt += 1
        if corrupt:
            log.warning(f"[rads] {corrupt} corrupt rows in {self.queries_path} "
                        "skipped (safe: batch_id will be regenerated)")
        return rows

    def _done_batch_ids(self) -> set:
        done = set()
        for row in self._accepted_rows():
            if row.get("generation", {}).get("template_version") == TEMPLATE_VERSION:
                done.add(row.get("class_metadata", {}).get("batch_id", ""))
        return done

    def build_bundles(self, per_class: int, only_classes: set = None) -> list:
        from app.benchmark.rads_synthesis.qc import _loaded_embedder
        chunks = load_child_chunks()
        distances = load_course_graph()
        directed = load_course_digraph()
        embed_fn = lambda texts: _loaded_embedder().embed_text_dense(texts)
        selector = ChunkSelector(chunks, distances, seed=self.seed,
                                 embed_fn=embed_fn, directed_distances=directed,
                                 hard_mix=self.hard_mix)
        bundles = selector.select_all(per_class, only_classes=only_classes)
        for i, b in enumerate(bundles):
            b.class_metadata["batch_id"] = batch_id_for(b, self.seed)
            b.class_metadata["template_version"] = TEMPLATE_VERSION
            b.class_metadata["difficulty_target"] = DIFFICULTIES[i % len(DIFFICULTIES)]
            b.class_metadata["style_target"] = STYLES[i % len(STYLES)]
        return bundles

    def _make_query(self, bundle, data: dict, meta: dict) -> BenchmarkQuery:
        gold = data["gold_relevant_chunks"]
        profile = student_profile_for(bundle, gold_chunk_ids=gold)
        allowed = sorted(c.chunk_id for c in bundle.chunks)
        return BenchmarkQuery.new(
            query_id=bundle.class_metadata["batch_id"],
            class_label=bundle.class_label,
            difficulty=data["difficulty"],
            query_text=data["query_text"].strip(),
            expected_answer_summary=data["expected_answer_summary"].strip(),
            reasoning_chain=data["reasoning_chain"],
            gold_relevant_chunks=gold,
            student_profile=profile,
            class_metadata=bundle.class_metadata,
            generation={**meta, "template_version": TEMPLATE_VERSION,
                        "seed": self.seed, "source_chunk_ids": allowed,
                        "graph_version": GRAPH_VERSION,
                        "corpus_version": CORPUS_VERSION,
                        "difficulty_self_label": data["difficulty"]},
            linguistic_style=bundle.class_metadata.get("style_target", ""),
        )

    def _qc_and_persist(self, query, qc_runner, out_handle, report) -> bool:
        student = StudentContext(
            user_id="rads-qc",
            department_id=query.student_profile["department_id"],
            current_semester=query.student_profile["current_semester"],
            enrolled_courses=query.student_profile["enrolled_courses"],
        )
        qc = qc_runner.check(query, student)
        query.qc = qc
        query.difficulty = derive_for_query(query)
        query.difficulty_score_raw = derive_raw_for_query(query)
        if not qc["passed"]:
            report["rejected_qc"][_reject_bucket(qc)] += 1
            with open(self.rejects_path, "a") as rej:
                rej.write(json.dumps(query.to_dict(), ensure_ascii=False) + "\n")
            return False
        report["accepted"] += 1
        report["per_class"][query.class_label] = report["per_class"].get(query.class_label, 0) + 1
        out_handle.write(json.dumps(query.to_dict(), ensure_ascii=False) + "\n")
        out_handle.flush()
        return True

    def _new_report(self, requested: int) -> dict:
        return {"requested": requested, "skipped_resume": 0, "generated": 0,
                "rejected_generation": 0, "errors": 0,
                "rejected_qc": {k: 0 for k in REJECT_BUCKETS},
                "accepted": 0, "per_class": {}, "template_version": TEMPLATE_VERSION}

    def _qc_runner(self):
        from app.benchmark.rads_synthesis.qc import QcRunner
        preload = [r["query_text"] for r in self._accepted_rows()]
        return QcRunner(preload_texts=preload)

    def run_ollama(self, per_class: int) -> dict:
        backend = OllamaBackend()
        bundles = self.build_bundles(per_class)
        qc_runner = self._qc_runner()
        done = self._done_batch_ids()
        report = self._new_report(len(bundles))
        try:
            with open(self.queries_path, "a") as out:
                for bundle in bundles:
                    if bundle.class_metadata["batch_id"] in done:
                        report["skipped_resume"] += 1
                        continue
                    try:
                        clean = _sanitized_copy(bundle, self.sanitizer)
                        prompt = build_prompt(clean)
                        allowed = {c.chunk_id for c in bundle.chunks}
                        data, meta = backend.generate(prompt, allowed)
                        if data is None:
                            report["rejected_generation"] += 1
                            continue
                        report["generated"] += 1
                        query = self._make_query(bundle, data, meta)
                        self._qc_and_persist(query, qc_runner, out, report)
                    except Exception as e:
                        report["errors"] += 1
                        report.setdefault("error_samples", []).append(
                            f"{bundle.class_metadata['batch_id']}: {type(e).__name__}: {e}"[:200])
        finally:
            with open(self.report_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        return report

    def _accepted_per_class(self) -> dict:
        counts = {}
        for row in self._accepted_rows():
            label = row.get("class_label", "")
            counts[label] = counts.get(label, 0) + 1
        return counts

    def emit_fable_batch(self, per_class: int, batch_path: str,
                         target_accepted: int = None) -> int:
        bundles = self.build_bundles(per_class)
        done = self._done_batch_ids()
        accepted = self._accepted_per_class() if target_accepted else {}
        items = []
        for bundle in bundles:
            if bundle.class_metadata["batch_id"] in done:
                continue
            if target_accepted and accepted.get(bundle.class_label, 0) >= target_accepted:
                continue
            clean = _sanitized_copy(bundle, self.sanitizer)
            items.append({
                "batch_id": bundle.class_metadata["batch_id"],
                "prompt": build_prompt(clean),
                "allowed_chunk_ids": sorted(c.chunk_id for c in bundle.chunks),
                "bundle_meta": {**bundle.class_metadata, "class_label": bundle.class_label},
                "chunk_meta": [{"chunk_id": c.chunk_id, "course_id": c.course_id,
                                "course_code": c.course_code, "department_id": c.department_id,
                                "department_name": c.department_name, "semester": c.semester}
                               for c in bundle.chunks],
            })
        BatchFileBackend.emit_batch(items, batch_path)
        return len(items)

    def ingest_fable_results(self, results_path: str, per_class: int) -> dict:
        results = BatchFileBackend.ingest_results(results_path)
        bundles = {b.class_metadata["batch_id"]: b
                   for b in self.build_bundles(per_class)}
        qc_runner = self._qc_runner()
        done = self._done_batch_ids()
        report = self._new_report(len(results))
        try:
            with open(self.queries_path, "a") as out:
                for batch_id, (data, meta) in sorted(results.items()):
                    if batch_id in done:
                        report["skipped_resume"] += 1
                        continue
                    bundle = bundles.get(batch_id)
                    if bundle is None:
                        report["errors"] += 1
                        report.setdefault("error_samples", []).append(
                            f"{batch_id}: no bundle found (another seed, per_class, or template?)")
                        continue
                    if data is None:
                        report["rejected_generation"] += 1
                        continue
                    allowed = {c.chunk_id for c in bundle.chunks}
                    problems = validate_output(data, allowed)
                    if problems:
                        report["rejected_generation"] += 1
                        report.setdefault("rejected_generation_samples", []).append(
                            f"{batch_id}: {'; '.join(problems)}"[:200])
                        continue
                    report["generated"] += 1
                    try:
                        query = self._make_query(bundle, data, meta)
                        self._qc_and_persist(query, qc_runner, out, report)
                    except Exception as e:
                        report["errors"] += 1
                        report.setdefault("error_samples", []).append(
                            f"{batch_id}: {type(e).__name__}: {e}"[:200])
        finally:
            with open(self.report_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        return report

    def retrofit_rows(self) -> dict:
        from app.benchmark.rads_synthesis.difficulty import (
            classify_style, derive_for_query, derive_raw_for_query)
        stats = {"rows": 0, "difficulty_changed": 0, "style_tagged": 0,
                 "score_raw_filled": 0}
        for path in (self.queries_path, self.rejects_path):
            if not os.path.exists(path):
                continue
            rows = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            for row in rows:
                stats["rows"] += 1
                query = BenchmarkQuery(**row)
                new_diff = derive_for_query(query)
                row.setdefault("generation", {}).setdefault(
                    "difficulty_self_label", row.get("difficulty", ""))
                if row.get("difficulty") != new_diff:
                    stats["difficulty_changed"] += 1
                row["difficulty"] = new_diff
                if row.get("difficulty_score_raw", -1) < 0:
                    stats["score_raw_filled"] += 1
                row["difficulty_score_raw"] = derive_raw_for_query(query)
                row.setdefault("annotation_status", "synthetic_unvalidated")
                if not row.get("linguistic_style"):
                    row["linguistic_style"] = classify_style(row.get("query_text", ""))
                    stats["style_tagged"] += 1
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
        return stats

    def finalize_splits(self) -> dict:
        queries = [BenchmarkQuery(**row) for row in self._accepted_rows()]
        counts = assign_splits(queries)
        tmp_path = self.queries_path + ".tmp"
        with open(tmp_path, "w") as f:
            for q in queries:
                f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
        os.replace(tmp_path, self.queries_path)
        return counts
