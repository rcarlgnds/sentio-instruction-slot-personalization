import random
import re
from collections import defaultdict

import numpy as np

from app.benchmark.rads_synthesis.models import SelectionBundle

AMBIGUOUS_TERMS = [
    "normalization", "normalisasi",
    "network",
    "regression",
    "distribution",
    "variance",
    "sampling",
    "index",
    "class",
    "object",
    "model",
    "demand",
    "market",
    "inheritance",
    "cluster",
    "protocol",
    "capital",
    "function",
    "table",
    "correlation",
]

DEFINITIONAL_PATTERNS = re.compile(
    r"\b(is defined as|is called|refers to|adalah|didefinisikan sebagai|dikenal sebagai)\b",
    re.IGNORECASE)
CONCEPTUAL_PATTERNS = re.compile(
    r"\b(why|because|trade-?off|advantage|disadvantage|mengapa|keuntungan|kelemahan|dibandingkan)\b",
    re.IGNORECASE)

MIN_CHUNK_CHARS = 300
CLASS2_SAMPLE_PER_COURSE = 50
CLASS2_COHERENCE_FLOOR = 0.55
WORD_RE = re.compile(r"[a-zA-Z]{5,}")

BOILERPLATE_HEAD = re.compile(
    r"(?i)^\s*(table of contents|contents|references|bibliography|glossary|"
    r"index|exercises|answer key|solutions to|list of (figures|tables)|"
    r"chapter \d+\s*$)")
DOT_LEADER = re.compile(r"\.{5,}|(?:\.\s){4,}\.")
GLYPH_JUNK = re.compile(r"GLYPH<[^>]*>")
CITATION_MARKERS = re.compile(
    r"(\bet al\.|\bIEEE\b|\bJournal\b|\bProceedings\b|https?://|\bAccessed\b|"
    r"\bretrieved\b|\bpp\.\s?\d|\bvol\.\s?\d)", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _is_boilerplate(text: str) -> bool:
    head = text[:120]
    if BOILERPLATE_HEAD.search(head):
        return True
    if len(DOT_LEADER.findall(text)) >= 3:
        return True
    if len(GLYPH_JUNK.findall(text)) >= 3:
        return True
    n_markers = len(CITATION_MARKERS.findall(text))
    n_years = len(YEAR_RE.findall(text))
    if (n_markers >= 4 and n_years >= 3
            and (n_markers + n_years) / max(len(text), 1) > 0.008):
        return True
    digits = sum(ch.isdigit() for ch in text)
    if digits / max(len(text), 1) > 0.2:
        return True
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) >= 8:
        short_or_numbered = sum(1 for l in lines if len(l.strip()) < 40 or re.match(r"\s*\d+[.)]", l))
        if short_or_numbered / len(lines) > 0.7:
            return True
    return False


def _lexical_coherence(text_a: str, text_b: str) -> float:
    wa = set(WORD_RE.findall(text_a.lower()))
    wb = set(WORD_RE.findall(text_b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class ChunkSelector:
    def __init__(self, chunks: list, course_distances: dict, seed: int = 42,
                 embed_fn=None, directed_distances: dict = None,
                 hard_mix: float = 0.0):
        self.rng = random.Random(seed)
        self.seed = seed
        self.embed_fn = embed_fn
        self.hard_mix = max(0.0, min(1.0, hard_mix))
        self.chunks = [c for c in chunks
                       if len(c.raw_text) >= MIN_CHUNK_CHARS and not _is_boilerplate(c.raw_text)]
        self.course_distances = course_distances
        self.directed_distances = directed_distances or {}
        self.by_course = defaultdict(list)
        self.by_department = defaultdict(list)
        for c in self.chunks:
            self.by_course[c.course_id].append(c)
            self.by_department[c.department_name].append(c)
        self._pair_rankings = {}

    def _hard_slot(self, i: int) -> bool:
        return (i % 100) < round(self.hard_mix * 100)

    def select_class1(self, n: int) -> list:
        bundles = []
        term_hits = {}
        for term in AMBIGUOUS_TERMS:
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            per_dept = defaultdict(list)
            for c in self.chunks:
                if pattern.search(c.raw_text):
                    per_dept[c.department_name].append(c)
            if len(per_dept) >= 2:
                term_hits[term] = per_dept
        terms = sorted(term_hits)
        if not terms:
            return bundles
        while len(bundles) < n:
            term = terms[len(bundles) % len(terms)]
            per_dept = term_hits[term]
            if self._hard_slot(len(bundles)) and len(per_dept) >= 3:
                k = 3
            else:
                k = min(len(per_dept), self.rng.choice([2, 2, 3]))
            depts = self.rng.sample(sorted(per_dept), k=k)
            picked = [self.rng.choice(per_dept[d]) for d in depts]
            bundles.append(SelectionBundle(
                class_label="multi_disciplinary_ambiguity",
                chunks=picked,
                class_metadata={"ambiguous_term": term,
                                "departments": depts,
                                "seed": self.seed},
            ))
        return bundles

    def _ranked_pairs(self, course_a: str, course_b: str) -> list:
        key = (course_a, course_b)
        if key in self._pair_rankings:
            return self._pair_rankings[key]
        pool_a = self.by_course[course_a]
        pool_b = self.by_course[course_b]
        sample_a = pool_a if len(pool_a) <= CLASS2_SAMPLE_PER_COURSE else self.rng.sample(pool_a, CLASS2_SAMPLE_PER_COURSE)
        sample_b = pool_b if len(pool_b) <= CLASS2_SAMPLE_PER_COURSE else self.rng.sample(pool_b, CLASS2_SAMPLE_PER_COURSE)
        if self.embed_fn is not None:
            vecs_a = np.asarray(self.embed_fn([c.raw_text for c in sample_a]), dtype=np.float32)
            vecs_b = np.asarray(self.embed_fn([c.raw_text for c in sample_b]), dtype=np.float32)
            vecs_a /= np.linalg.norm(vecs_a, axis=1, keepdims=True) + 1e-9
            vecs_b /= np.linalg.norm(vecs_b, axis=1, keepdims=True) + 1e-9
            sims = vecs_a @ vecs_b.T
            scored = [(sample_a[i], sample_b[j], float(sims[i, j]))
                      for i in range(len(sample_a)) for j in range(len(sample_b))]
        else:
            scored = [(ca, cb, _lexical_coherence(ca.raw_text, cb.raw_text))
                      for ca in sample_a for cb in sample_b]
        scored.sort(key=lambda t: (-t[2], t[0].chunk_id, t[1].chunk_id))
        self._pair_rankings[key] = scored
        return scored

    def _class2_course_pairs(self) -> list:
        directed = []
        for a in sorted(self.directed_distances):
            for b, dist in sorted(self.directed_distances[a].items()):
                if 2 <= dist <= 4 and a in self.by_course and b in self.by_course:
                    directed.append((a, b, dist, "directed"))
        if directed:
            return directed
        sibling = []
        for a in sorted(self.course_distances):
            for b, dist in sorted(self.course_distances[a].items()):
                if a < b and 2 <= dist <= 4 and a in self.by_course and b in self.by_course:
                    sibling.append((a, b, dist, "undirected-fallback"))
        return sibling

    def select_class2(self, n: int) -> list:
        pairs = self._class2_course_pairs()
        bundles = []
        if not pairs:
            return bundles
        viable = []
        for a, b, dist, direction in pairs:
            ranked = self._ranked_pairs(a, b)
            ranked = [t for t in ranked if t[2] >= CLASS2_COHERENCE_FLOOR] if self.embed_fn else ranked
            if ranked:
                viable.append((a, b, dist, direction, ranked))
        if not viable:
            return bundles
        take_counter = defaultdict(int)
        deep = [v for v in viable if v[2] >= 3]
        while len(bundles) < n:
            pool = deep if (self._hard_slot(len(bundles)) and deep) else viable
            a, b, dist, direction, ranked = pool[len(bundles) % len(pool)]
            idx = take_counter[(a, b)] % len(ranked)
            take_counter[(a, b)] += 1
            chunk_a, chunk_b, coherence = ranked[idx]
            bundles.append(SelectionBundle(
                class_label="prerequisite_chain_depth",
                chunks=[chunk_a, chunk_b],
                class_metadata={"hop_distance": dist,
                                "direction": direction,
                                "course_pair": [chunk_a.course_code, chunk_b.course_code],
                                "coherence": round(coherence, 4),
                                "coherence_rank": idx + 1,
                                "seed": self.seed},
            ))
        return bundles

    def select_class3(self, n: int) -> list:
        bundles = []
        candidates = [c for c in self.chunks if c.semester > 0]
        if not candidates:
            return bundles
        while len(bundles) < n:
            chunk = self.rng.choice(candidates)
            min_drift = 4 if self._hard_slot(len(bundles)) else 2
            drift_options = [s for s in range(1, 8)
                             if abs(s - chunk.semester) >= min_drift]
            if not drift_options:
                drift_options = [s for s in range(1, 8)
                                 if abs(s - chunk.semester) >= 2]
            student_semester = self.rng.choice(drift_options)
            bundles.append(SelectionBundle(
                class_label="temporal_scope_drift",
                chunks=[chunk],
                class_metadata={"content_semester": chunk.semester,
                                "student_semester": student_semester,
                                "drift": student_semester - chunk.semester,
                                "seed": self.seed},
            ))
        return bundles

    def select_class4(self, n: int) -> list:
        definitional = [c for c in self.chunks if DEFINITIONAL_PATTERNS.search(c.raw_text)]
        conceptual = [c for c in self.chunks if CONCEPTUAL_PATTERNS.search(c.raw_text)]
        bundles = []
        if not definitional or not conceptual:
            return bundles
        while len(bundles) < n:
            flip = len(bundles) % 2 == 0
            chunk = self.rng.choice(definitional if flip else conceptual)
            bundles.append(SelectionBundle(
                class_label="adversarial_routing",
                chunks=[chunk],
                class_metadata={"content_style": "definitional" if flip else "conceptual",
                                "target_boundary": "sounds-socratic-needs-factual" if flip
                                                   else "sounds-factual-needs-reasoning",
                                "query_form": "short" if (self._hard_slot(len(bundles))
                                                          or len(bundles) % 3 == 2) else "long",
                                "seed": self.seed},
            ))
        return bundles

    def select_all(self, per_class: int, only_classes: set = None) -> list:
        selectors = [
            ("multi_disciplinary_ambiguity", self.select_class1),
            ("prerequisite_chain_depth", self.select_class2),
            ("temporal_scope_drift", self.select_class3),
            ("adversarial_routing", self.select_class4),
        ]
        bundles = []
        for label, fn in selectors:
            if only_classes is None or label in only_classes:
                bundles.extend(fn(per_class))
        return bundles
