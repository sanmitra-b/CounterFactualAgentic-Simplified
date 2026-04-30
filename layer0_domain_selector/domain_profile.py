from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class DomainProfile:
    domain: str
    description: str
    keywords: list[str]
    config_path: Path
    config: Dict[str, Any]

    @property
    def slug(self) -> str:
        return self.domain.strip().lower().replace(" ", "_")
