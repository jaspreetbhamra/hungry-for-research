from pydantic import BaseSettings, Field


class Neo4jSettings(BaseSettings):
    uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    user: str = Field(default="neo4j", env="NEO4J_USER")
    password: str = Field(default="neo4j", env="NEO4J_PASSWORD")

    class Config:
        env_file = ".env"


neo4j_settings = Neo4jSettings()
