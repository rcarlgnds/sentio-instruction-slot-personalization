from collections import deque

from app.utils.logging import get_logger

log = get_logger(__name__)


class PrerequisiteGraph:
    def __init__(self, edges: list[tuple[str, str]]):
        self._adjacency: dict[str, set[str]] = {}
        for prereq_id, course_id in edges:
            self._adjacency.setdefault(prereq_id, set()).add(course_id)
            self._adjacency.setdefault(course_id, set()).add(prereq_id)

        self._distances: dict[str, dict[str, int]] = {
            node: self._bfs(node) for node in self._adjacency
        }
        log.info(
            f"PrerequisiteGraph loaded: {len(self._adjacency)} nodes, "
            f"{len(edges)} edges, distances precomputed"
        )

    @classmethod
    def load(cls, pg_adapter) -> "PrerequisiteGraph":
        try:
            edges = pg_adapter.fetch_course_prerequisite_edges()
        except Exception as e:
            log.warning(f"Failed to load course_prerequisites, graph empty: {e}")
            edges = []
        if not edges:
            log.warning("PrerequisiteGraph empty: Struct_n will be 0 for every chunk")
        return cls(edges)

    def _bfs(self, start: str) -> dict[str, int]:
        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in self._adjacency.get(node, ()):
                if neighbor not in dist:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        return dist

    def shortest_path(self, course_id: str, completed: list[str]) -> int | None:
        if not course_id or not completed:
            return None
        table = self._distances.get(course_id)
        if table is None:
            return None
        best = None
        for done in completed:
            d = table.get(done)
            if d is not None and (best is None or d < best):
                best = d
        return best
