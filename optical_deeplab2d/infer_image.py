from __future__ import annotations
import argparse
from pathlib import Path
import json, sys
import numpy as np
import torch
from PIL import Image
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optical_deeplab2d.models.hybrid_deeplabv3plus import HybridOpticalDeepLabV3Plus
from optical_deeplab2d.models.electronic_deeplabv3plus import ElectronicDeepLabV3Plus
from optical_deeplab2d.models.electronic_deepseg_decoder import ElectronicDeepSegDecoder

MODEL_TYPES = {
 'hybrid_ideal': HybridOpticalDeepLabV3Plus,
 'electronic_baseline': ElectronicDeepLabV3Plus,
 'electronic_deepseg_decoder': ElectronicDeepSegDecoder,
}

def main() -> None:
 p=argparse.ArgumentParser(description='Run checkpoint-consistent inference for one DWI image.');p.add_argument('--image',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args(); ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 model_type=ckpt.get('model_type')
 if model_type is None: raise ValueError("Checkpoint is missing required model_type metadata")
 if model_type not in MODEL_TYPES: raise ValueError(f"Unknown model type: {model_type}")
 cls=MODEL_TYPES[model_type]
 model=cls(ckpt['encoder_name'],None).eval();model.load_state_dict(ckpt['model_state_dict']);raw=np.asarray(Image.open(a.image));raw=raw[...,0] if raw.ndim==3 else raw; scale=float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype,np.integer) else 1.; image=torch.from_numpy(raw.astype(np.float32)/scale)[None,None]; probability=model(image).sigmoid()[0,0].detach().numpy(); prediction=(probability>=ckpt['threshold']).astype(np.uint8);a.output_dir.mkdir(parents=True,exist_ok=True);np.save(a.output_dir/'prediction.npy',prediction);Image.fromarray((probability*255).astype(np.uint8)).save(a.output_dir/'probability.png');Image.fromarray(prediction*255).save(a.output_dir/'prediction.png');Image.fromarray((raw/scale*255).astype(np.uint8)).convert('RGB').save(a.output_dir/'overlay.png');(a.output_dir/'metadata.json').write_text(json.dumps({k:ckpt.get(k) for k in ('model_type','encoder_name','threshold','normalization','fold','seed')},indent=2),encoding='utf-8')
if __name__=='__main__':main()
