import json
import urllib.request
from collections import defaultdict, deque

import psycopg
from psycopg.rows import dict_row

from app.benchmark.rads_synthesis.models import SourceChunk
from app.config import settings

PAYLOAD_FIELDS = ["chunk_id", "course_id", "course_name", "department_id",
                  "department_name", "semester", "raw_text", "chunk_level"]


def load_child_chunks() -> list:
    url = f"{settings.qdrant_url}/collections/{settings.qdrant_collection}/points/scroll"
    code_by_course = _course_codes()
    chunks = []
    offset = None
    while True:
        body = {"limit": 1000, "with_payload": PAYLOAD_FIELDS, "with_vector": False,
                "filter": {"must": [{"key": "chunk_level", "match": {"value": "child"}}]}}
        if offset:
            body["offset"] = offset
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        result = json.load(urllib.request.urlopen(req, timeout=90))["result"]
        for p in result["points"]:
            pl = p["payload"]
            chunks.append(SourceChunk(
                chunk_id=pl.get("chunk_id", ""),
                course_id=pl.get("course_id", ""),
                course_code=code_by_course.get(pl.get("course_id", ""), ""),
                course_name=pl.get("course_name", ""),
                department_id=pl.get("department_id", ""),
                department_name=pl.get("department_name", ""),
                semester=int(pl.get("semester") or 0),
                raw_text=pl.get("raw_text", ""),
            ))
        offset = result.get("next_page_offset")
        if not offset:
            break
    chunks.sort(key=lambda c: c.chunk_id)
    return chunks


def _course_codes() -> dict:
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, course_code FROM courses WHERE is_active = true")
            return {str(r["id"]): r["course_code"] for r in cur.fetchall()}


def _fetch_edges() -> list:
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT course_id, prerequisite_id FROM course_prerequisites")
            return [(str(r["course_id"]), str(r["prerequisite_id"])) for r in cur.fetchall()]


def _bfs_all(adjacency: dict) -> dict:
    distances = {}
    for start in sorted(adjacency):
        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in dist:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        distances[start] = dist
    return distances


def load_course_graph() -> dict:
    adjacency = defaultdict(set)
    for course, prereq in _fetch_edges():
        adjacency[course].add(prereq)
        adjacency[prereq].add(course)
    return _bfs_all(adjacency)


def load_course_digraph() -> dict:
    adjacency = defaultdict(set)
    for course, prereq in _fetch_edges():
        adjacency[prereq].add(course)
    return _bfs_all(adjacency)
