from __future__ import annotations
from pathlib import Path
import torch
def save_checkpoint(path: Path, *, model, optimizer, scheduler, epoch: int, best_metric: float, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"epoch":epoch,"best_metric":best_metric,**metadata}, path)
def load_checkpoint(path: Path, device: torch.device) -> dict: return torch.load(path, map_location=device, weights_only=False)

def validate_resume(checkpoint: dict, config: dict) -> None:
    """Reject resume when the resolved experiment definition differs."""
    if checkpoint.get('config') != config: raise ValueError('Checkpoint configuration differs from current resolved configuration.')
