"""
Domain config loader — reads YAML files from configs/domains/ and returns
a typed DomainConfig object used by the processor pipeline and frontend.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Resolve the configs directory relative to the repo root.
# Works both in Docker (/app/configs) and locally (repo root / configs).
_APP_DIR = Path("/app") if Path("/app").exists() else Path(__file__).parents[3]
CONFIGS_DIR = _APP_DIR / "configs" / "domains"


@dataclass
class ChunkingConfig:
    chunk_size: int = 2000
    chunk_overlap: int = 200
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ".", " "])


@dataclass
class LLMConfig:
    model: str = "gpt-4o"
    temperature: float = 0.0
    batch_size: int = 20


@dataclass
class DomainConfig:
    domain_id: str
    display_name: str
    allowed_nodes: List[str]
    allowed_relationships: List[str]
    system_prompt: str
    node_properties: List[str] = field(default_factory=list)
    relationship_properties: List[str] = field(default_factory=list)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    visualization: Dict[str, Dict[str, str]] = field(default_factory=dict)


def list_domains() -> List[str]:
    """Return all available domain IDs."""
    if not CONFIGS_DIR.exists():
        return []
    return [p.stem for p in sorted(CONFIGS_DIR.glob("*.yaml"))]


def load_domain_config(domain_id: str) -> DomainConfig:
    """Load a domain config by ID. Raises FileNotFoundError with helpful message if not found."""
    path = CONFIGS_DIR / f"{domain_id}.yaml"
    if not path.exists():
        available = list_domains()
        raise FileNotFoundError(
            f"Domain config '{domain_id}' not found at {path}. "
            f"Available domains: {available or ['(none — add YAML files to configs/domains/)']}"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    chunking_raw = raw.get("chunking", {})
    chunking = ChunkingConfig(
        chunk_size=chunking_raw.get("chunk_size", 2000),
        chunk_overlap=chunking_raw.get("chunk_overlap", 200),
        separators=chunking_raw.get("separators", ["\n\n", "\n", ".", " "]),
    )

    llm_raw = raw.get("llm", {})
    llm = LLMConfig(
        model=llm_raw.get("model", "gpt-4o"),
        temperature=float(llm_raw.get("temperature", 0.0)),
        batch_size=int(llm_raw.get("batch_size", 20)),
    )

    config = DomainConfig(
        domain_id=raw["domain_id"],
        display_name=raw["display_name"],
        allowed_nodes=raw["allowed_nodes"],
        allowed_relationships=raw["allowed_relationships"],
        system_prompt=raw["system_prompt"].strip(),
        node_properties=raw.get("node_properties", []),
        relationship_properties=raw.get("relationship_properties", []),
        chunking=chunking,
        llm=llm,
        visualization=raw.get("visualization", {}),
    )
    logger.info("Loaded domain config: %s (%s)", config.domain_id, config.display_name)
    return config
