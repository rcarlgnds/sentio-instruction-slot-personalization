import psycopg
from psycopg.rows import dict_row

from app.config import settings
from app.utils.logging import get_logger

log = get_logger(__name__)

class PostgresAdapter:
    def __init__(self):
        self.conn_str = settings.database_url

    def _conn(self):
        return psycopg.connect(self.conn_str, row_factory=dict_row)

    def count_active_documents(self) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) AS n FROM knowledge_documents WHERE is_active")
                return int(cur.fetchone()["n"])

    def count_queries(self) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) AS n FROM query_audit_log")
                return int(cur.fetchone()["n"])

    def recent_audit(self, limit: int = 8) -> list[dict]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    SELECT query_id, question, mode, gate_result, created_at
                    FROM query_audit_log
                    ORDER BY created_at DESC
                    LIMIT %s
                    ''',
                    (limit,),
                )
                rows = cur.fetchall()
        return [
            {
                "query_id": str(r["query_id"]),
                "question": r["question"],
                "mode": r["mode"],
                "gate_result": r["gate_result"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]

    def update_status(
        self,
        document_id: str,
        status: str,
        chunk_count: int | None = None,
        error: str | None = None,
    ):
        with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        '''
                        UPDATE knowledge_documents
                        SET status = %s,
                            chunk_count = COALESCE(%s, chunk_count),
                            error_message = %s,
                            updated_at = NOW()
                        WHERE id = %s
                        ''',
                        (status, chunk_count, error, document_id),
                    )
                conn.commit()

    def fetch_course_metadata(self, course_id: str) -> dict:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    SELECT
                        c.id AS course_id,
                        c.course_code,
                        c.name AS course_name,
                        c.semester,
                        c.credits,
                        d.id AS department_id,
                        d.name AS department_name
                    FROM courses c
                    JOIN departments d ON c.department_id = d.id
                    WHERE c.id = %s::uuid AND c.is_active = true
                    ''',
                    (course_id,)
                )
                row = cur.fetchone()

        if row is None:
            return {}
        return {
            "course_id": str(row["course_id"]),
            "course_code": row["course_code"],
            "course_name": row["course_name"],
            "semester": row["semester"],
            "credits": row["credits"],
            "department_id": str(row["department_id"]),
            "department_name": row["department_name"],
        }

    def fetch_department_code(self, department_id: str) -> str | None:
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT name FROM departments WHERE id = %s",
                        (department_id,),
                    )
                    row = cur.fetchone()
            return row["name"] if row else None
        except Exception as e:
            log.warning(f"[postgres] fetch_department_code failed: {e}")
            return None

    def fetch_document_filename(self, document_id: str) -> str | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT file_name FROM knowledge_documents WHERE id = %s::uuid",
                    (document_id,),
                )
                row = cur.fetchone()
        return row["file_name"] if row else None

    def insert_query_audit(self, rec: dict) -> None:
        import json as _json

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    INSERT INTO query_audit_log (
                        query_id, user_pseudonym, user_role, department_id,
                        current_semester, question, mode, forced_mode,
                        routing_confidence, routing_gap, top_k,
                        bounds_min, bounds_max, bounds_provisional,
                        weights, config_version, gate_result, refusal_reason,
                        chunks, answer, citations, grounding_status
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s, %s::jsonb,
                        %s::jsonb
                    )
                    ''',
                    (
                        rec["query_id"], rec["user_pseudonym"], rec["user_role"],
                        rec["department_id"], rec["current_semester"], rec["question"],
                        rec["mode"], rec["forced_mode"],
                        rec["routing_confidence"], rec["routing_gap"], rec["top_k"],
                        rec["bounds_min"], rec["bounds_max"], rec["bounds_provisional"],
                        _json.dumps(rec["weights"]), rec["config_version"],
                        rec["gate_result"], rec["refusal_reason"],
                        _json.dumps(rec["chunks"]), rec["answer"],
                        _json.dumps(rec["citations"]),
                        _json.dumps(rec["grounding_status"])
                        if rec.get("grounding_status") else None,
                    ),
                )
            conn.commit()

    def fetch_course_prerequisite_edges(self) -> list[tuple[str, str]]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    SELECT course_id, prerequisite_id
                    FROM course_prerequisites
                    '''
                )
                rows = cur.fetchall()

        return [(str(r["prerequisite_id"]), str(r["course_id"])) for r in rows]

    def fetch_department_relations(self) -> dict[tuple[str, str], float]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    SELECT dept_a_id, dept_b_id, relatedness_score
                    FROM department_relations
                    '''
                )
                rows = cur.fetchall()

        out: dict[tuple[str, str], float] = {}
        for row in rows:
            a = str(row["dept_a_id"])
            b = str(row["dept_b_id"])
            pair = tuple(sorted([a, b]))
            out[pair] = float(row["relatedness_score"])
        return out
