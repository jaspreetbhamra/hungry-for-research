from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

from common.config import secrets, yaml_config


class Neo4jClient:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self._driver = GraphDatabase.driver(
            uri or yaml_config.neo4j.uri,
            auth=(user or secrets.neo4j_user, password or secrets.neo4j_password),
        )

    def close(self):
        self._driver.close()

    def run(
        self, cypher: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(cypher, params or {})
            return [r.data() for r in result]

    def upsert_paper(self, paper_id: str, title: str):
        cypher = """
        MERGE (p:Paper {id: $paper_id})
        ON CREATE SET p.title = $title
        RETURN p
        """
        return self.run(cypher, {"paper_id": paper_id, "title": title})

    def upsert_fact(self, paper_id: str, subject: str, predicate: str, obj: str):
        """
        Upsert a Fact node linked to a Paper and Entities.
        Uses fact_id = sha1(paper_id+subject+predicate+object) for idempotency.
        """
        fact_id = hashlib.sha1(
            f"{paper_id}|{subject}|{predicate}|{obj}".encode("utf-8")
        ).hexdigest()

        cypher = """
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $object})
        MERGE (p:Paper {id: $paper_id})
        MERGE (f:Fact {fact_id: $fact_id})
        ON CREATE SET f.pred = $predicate, f.subject = $subject, f.object = $object
        MERGE (s)-[:SUBJECT]->(f)
        MERGE (f)-[:OBJECT]->(o)
        MERGE (p)-[:CONTAINS]->(f)
        RETURN s, f, o, p
        """
        return self.run(
            cypher,
            {
                "paper_id": paper_id,
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "fact_id": fact_id,
            },
        )

    def query_entities(self, names: List[str]) -> List[Dict[str, Any]]:
        """
        Query triples involving the given entity names.
        """
        cypher = """
        MATCH (s:Entity)-[:SUBJECT]->(f:Fact)-[:OBJECT]->(o:Entity)
        WHERE s.name IN $names OR o.name IN $names
        OPTIONAL MATCH (p:Paper)-[:CONTAINS]->(f)
        RETURN s.name AS subject, f.pred AS predicate, o.name AS object, p.id AS paper_id
        """
        return self.run(cypher, {"names": names})
