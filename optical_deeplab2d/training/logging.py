from __future__ import annotations
import csv
from pathlib import Path

FIELDS = ['epoch','train_total_loss','train_bce_loss','train_dice_loss','val_global_dice','val_mean_image_dice','val_mean_patient_dice','val_precision','val_recall','foreground_pixels','background_pixels','raw_pos_weight','pos_weight','encoder_lr','new_layers_lr','epoch_time','gpu_memory_mb']
def append_log(path: Path, row: dict) -> None:
    exists=path.exists()
    with path.open('a',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS); 
        if not exists: writer.writeheader()
        writer.writerow({key:row.get(key,'') for key in FIELDS})
