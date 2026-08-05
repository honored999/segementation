from __future__ import annotations

import re
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = (
    "standalone_nnunet2d/README.md",
    "standalone_nnunet2d/REPRODUCTION_NOTES.md",
)


def _states_smoke_and_online_validation_are_not_official_reproduction(
    text: str,
) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower())
    return bool(
        re.search(
            r"(?:smoke[^.]{0,120}online validation|"
            r"online validation[^.]{0,120}smoke)"
            r"[^.]{0,120}(?:not|never|exclude|excluded|does not|is not|are not)"
            r"[^.]{0,120}official reproduction",
            normalized,
        )
    )


@pytest.mark.parametrize("relative_path", DOCUMENTS)
def test_documentation_contract_declares_pending_server_parity_and_non_official_validation(
    relative_path: str,
) -> None:
    document = REPOSITORY_ROOT / relative_path
    text = document.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", text.lower())

    required_phrases = (
        "official_alignment_pending",
        "parity report",
        "server oracle capture",
    )
    missing_phrases = [
        phrase for phrase in required_phrases if phrase not in normalized
    ]
    assert not missing_phrases, f"{relative_path} is missing: {missing_phrases}"
    assert "smoke" in normalized
    assert "online validation" in normalized
    assert _states_smoke_and_online_validation_are_not_official_reproduction(text)
