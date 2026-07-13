"""Leakage-free deterministic patient-level cross-validation splits."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json, random
from .dataset_2d import SampleRecord

@dataclass(frozen=True)
class PatientFold:
    fold: int; train_patients: list[str]; val_patients: list[str]

def build_patient_folds(records: list[SampleRecord], seed: int = 2026, n_splits: int = 5) -> list[PatientFold]:
    patients = sorted({record.patient for record in records})
    if len(patients) < n_splits: raise ValueError("Need at least n_splits distinct patients")
    random.Random(seed).shuffle(patients)
    return [PatientFold(index, sorted(p for i, p in enumerate(patients) if i % n_splits != index), sorted(p for i, p in enumerate(patients) if i % n_splits == index)) for index in range(n_splits)]

def save_folds(folds: list[PatientFold], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps([asdict(fold) for fold in folds], ensure_ascii=False, indent=2), encoding="utf-8")

