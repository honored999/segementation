"""Standalone H2Former diagnostic report adapter for existing source-space predictions."""
from __future__ import annotations

import json
from pathlib import Path

from generate_nnunet_result_report import (
    _hash_file, _native_channel_figure, _summary_figure, check_geometry,
    collect, protected_output, read_metrics, resolved, select_cases, select_slices,
)

SUPPORTED = {"h2former", "h2former_lite_upernet", "h2former_lite_upernet_w128_ppm1236"}


def _source_statement(confirmed, declaration):
    if bool(declaration and declaration.strip()) != bool(confirmed):
        raise ValueError("user confirmation requires both flag and nonempty declaration")
    if confirmed:
        return "USER CONFIRMED (user statement only): " + declaration.strip()
    return "UNKNOWN; specified checkpoint used only for diagnostic features"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))


def _identity(metadata, config, manifest):
    from standalone_nnunet2d.models.factory import resolve_checkpoint_model_identity
    identity = resolve_checkpoint_model_identity(metadata)
    if identity[0] not in SUPPORTED or identity[1] != "single_output":
        raise ValueError(f"unsupported standalone checkpoint identity: {identity}")
    def serialized_json_identity(candidate):
        """Restore only JSON's tuple-to-list conversion for the known PPM scales."""
        import copy
        value = copy.deepcopy(candidate)
        nodes = [value, value.get("architecture")]
        for key in ("resolved_config", "config"):
            config_node = value.get(key)
            if isinstance(config_node, dict):
                nodes.append(config_node.get("model"))
        for node in nodes:
            if isinstance(node, dict) and "ppm_scales" in node:
                scales = node["ppm_scales"]
                if not isinstance(scales, list) or any(type(x) is not int for x in scales):
                    raise ValueError("serialized ppm_scales must be an integer JSON array")
                node["ppm_scales"] = tuple(scales)
        return value

    for name, candidate in (("resolved config", config),
                            ("prediction manifest", manifest.get("checkpoint"))):
        if name == "resolved config":
            if not isinstance(candidate, dict) or not isinstance(candidate.get("model"), dict):
                raise ValueError("resolved config requires model identity")
            candidate = {"resolved_config": candidate}
        if not isinstance(candidate, dict) or not candidate:
            raise ValueError(f"{name} requires a checkpoint model identity object")
        # Reuse the strict factory; its legacy PlainConv fallback mismatches here.
        if resolve_checkpoint_model_identity(serialized_json_identity(candidate)) != identity:
            raise ValueError(f"{name} model identity conflicts with diagnostic checkpoint")
    return identity[0]


def _manifest_cases(manifest, images, predictions, *, allow_pending):
    from standalone_nnunet2d.training.official_config import DEFAULT_RUN_STATE
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("policy"), dict):
        raise ValueError("prediction manifest schema/policy invalid")
    policy = manifest["policy"]
    from standalone_nnunet2d.alignment_evidence import OFFICIAL_ALIGNED
    if (policy.get("output_space") != "source" or
            policy.get("run_state") not in {OFFICIAL_ALIGNED, DEFAULT_RUN_STATE} or
            policy.get("run_state") != policy.get("alignment_status")):
        raise ValueError("prediction manifest source-space/alignment policy invalid")
    from standalone_nnunet2d.alignment_evidence import validate_checkpoint_alignment_metadata
    validate_checkpoint_alignment_metadata({
        "run_type": policy["run_state"], "run_state": policy["alignment_status"],
        "alignment_evidence": policy.get("alignment_evidence"),
    })
    if policy.get("run_state") == DEFAULT_RUN_STATE and not allow_pending:
        raise ValueError("pending prediction manifest requires explicit --allow-pending")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != len(predictions):
        raise ValueError("prediction manifest case coverage mismatch")
    seen = set()
    for item in cases:
        if not isinstance(item, dict) or item.get("case_id") not in predictions:
            raise ValueError("prediction manifest has unknown case")
        cid = item["case_id"]
        if cid in seen:
            raise ValueError("prediction manifest has duplicate case")
        seen.add(cid)
        if resolved(item.get("source_path", "")) != images[cid] or resolved(item.get("prediction_path", "")) != predictions[cid]:
            raise ValueError(f"{cid}: prediction manifest path mismatch")
    if seen != set(predictions):
        raise ValueError("prediction manifest case coverage mismatch")
    return policy


def inspect(args):
    if args.fold != 0:
        raise ValueError("standalone report currently supports fold 0 only")
    if args.prediction_dir is None or args.metrics_dir is None or args.manifest is None or args.config is None:
        raise ValueError("standalone requires --prediction-dir, --metrics-dir, --manifest and --config")
    paths = {name: resolved(getattr(args, name)) for name in
             ("model_dir", "images_dir", "labels_dir", "prediction_dir", "metrics_dir", "manifest", "config", "checkpoint")}
    output = protected_output(args.output_dir, paths.values())
    if not all(paths[name].is_file() for name in ("manifest", "config", "checkpoint")):
        raise ValueError("standalone manifest/config/checkpoint file missing")
    images = collect(paths["images_dir"], input_channel=True)
    labels = collect(paths["labels_dir"])
    predictions = collect(paths["prediction_dir"])
    if not set(predictions) <= set(images) or not set(predictions) <= set(labels):
        raise ValueError("standalone predictions include cases missing from image/GT sources")
    images = {cid: images[cid] for cid in predictions}
    labels = {cid: labels[cid] for cid in predictions}
    rows, summary = read_metrics(paths["metrics_dir"], predictions)
    manifest = _json(paths["manifest"])
    config = _json(paths["config"])
    policy = _manifest_cases(manifest, images, predictions, allow_pending=args.allow_pending)
    return dict(paths=paths, output=output, images=images, labels=labels,
                predictions=predictions, rows=rows, summary=summary,
                manifest=manifest, config=config, policy=policy)


def _capture(model, tile):
    """Capture decoder inputs, which are the four post-Swin fused BCHW tensors."""
    import torch
    import torch.nn.functional as F
    from generate_nnunet_result_report import _channel_ids
    slot = {}
    module = model.decode4 if hasattr(model, "decode4") else model.decoder

    def hook(_module, inputs):
        if len(slot):
            raise ValueError("decoder input hook ran more than once for one tile")
        features = (inputs[0], inputs[1]) if hasattr(model, "decode4") else inputs[0]
        if len(features) != (2 if hasattr(model, "decode4") else 4):
            raise ValueError("decoder input feature count mismatch")
        deep, middle = (features[0], features[1]) if hasattr(model, "decode4") else (features[3], features[2])
        side = model.image_size
        for name, value, channels, factor in (("deep", deep, 512, 16), ("middle", middle, 256, 8)):
            expected = (1, channels, side // factor, side // factor)
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} fused feature BCHW shape {tuple(value.shape)} != {expected}")
            ids = _channel_ids(channels)
            slot[name] = dict(channels=value[0, ids].detach().float().cpu().numpy().copy(),
                              channel_ids=ids, feature_shape=tuple(value.shape),
                              magnitude=F.interpolate(value.detach().float().abs().mean(1, keepdim=True),
                                                      size=tile.shape[-2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy().copy())

    handle = module.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            model(tile)
        if set(slot) != {"deep", "middle"}:
            raise ValueError("decoder fused features were not captured")
        return slot
    finally:
        handle.remove()


def _slice_features(model, normalized, selected, device):
    """Source-space diagnostic magnitude for selected slices only; no GT input."""
    import numpy as np
    import torch
    from standalone_nnunet2d.engine.predictor import _tile_starts, compute_gaussian
    height, width = normalized.shape[1:]
    patch = model.image_size
    pad_y, pad_x = max(0, patch-height), max(0, patch-width)
    top, left = pad_y//2, pad_x//2
    native = None
    intermediate = None
    maps = {}
    was_training = model.training
    model.eval()
    try:
        for z in selected:
            padded = np.pad(normalized[z], ((top, pad_y-top), (left, pad_x-left)))
            ph, pw = padded.shape
            ys, xs = _tile_starts(ph, patch, .5), _tile_starts(pw, patch, .5)
            weight = compute_gaussian((patch, patch), sigma_scale=1/8,
                                      value_scaling_factor=10.0, dtype=torch.float32,
                                      device=torch.device("cpu")).numpy()
            total = np.zeros((ph, pw), np.float32)
            weights = np.zeros((ph, pw), np.float32)
            for yi, y in enumerate(ys):
                for xi, x in enumerate(xs):
                    tile = torch.from_numpy(padded[y:y+patch, x:x+patch].copy()).to(device).view(1, 1, patch, patch)
                    captured = _capture(model, tile)
                    total[y:y+patch, x:x+patch] += captured["deep"]["magnitude"] * weight
                    weights[y:y+patch, x:x+patch] += weight
                    if z == selected[0] and native is None:
                        common = dict(original_slice=z, preprocessed_slice=z,
                                      window_index=yi*len(xs)+xi, window_yx_padded=(y, x),
                                      window_count=len(ys)*len(xs),
                                      padding_offsets_yx=((top, pad_y-top), (left, pad_x-left)),
                                      source_shape_yx=(height, width), padded_shape_yx=(ph, pw),
                                      window_shape_yx=(patch, patch))
                        native = {**common, **{k:v for k,v in captured["deep"].items() if k != "magnitude"},
                                  "stage":3, "module":"decode4" if hasattr(model, "decode4") else "decoder",
                                  "input_index":0 if hasattr(model, "decode4") else 3,
                                  "feature_source":"forward_features[3] post-Swin fused"}
                        intermediate = {**common, **{k:v for k,v in captured["middle"].items() if k != "magnitude"},
                                        "stage":2, "module":"decode4" if hasattr(model, "decode4") else "decoder",
                                        "input_index":1 if hasattr(model, "decode4") else 2,
                                        "feature_source":"forward_features[2] post-Swin fused"}
            if np.any(weights == 0):
                raise ValueError("diagnostic window coverage incomplete")
            maps[z] = (total/weights)[top:top+height, left:left+width]
    finally:
        model.train(was_training)
    return maps, native, intermediate


def _native_mapping(native):
    """Captured provenance, excluding feature arrays."""
    return None if native is None else {k:v for k,v in native.items() if k not in {"channels", "magnitude"}}


def _architecture(model, name, output, *, detail=False):
    """Numbered panels following the production encoder and decoder forwards."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from standalone_nnunet2d.models.factory import get_model_contract
    size = model.image_size
    lite = name != "h2former"
    channels = tuple(getattr(model.decoder, "in_channels", (64,128,256,512))) if lite else (64,128,256,512)
    width = model.decoder.classifier.in_channels if lite else None
    scales = tuple(model.decoder.pool_scales) if lite else ()
    classes = model.decoder.classifier.out_channels if lite else get_model_contract(name).num_classes
    fig, axes = plt.subplots(4 if detail else 3, 1, figsize=(18,19 if detail else 15))
    def panel(ax, title):
        ax.set(xlim=(0,18), ylim=(-1,7)); ax.axis("off")
        ax.set_title(title,loc="left",fontsize=15,weight="bold")
        return {}
    def box(ax,nodes,key,x,y,label,w=2.6,h=.8):
        nodes[key]=(x,y,w,h)
        p=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=.06",facecolor="#edf4fa",edgecolor="#476b84")
        p.set_gid(key); ax.add_patch(p)
        ax.text(x,y,label,ha="center",va="center",fontsize=10)
    def edge(ax,nodes,a,b,side=False,label=None):
        x,y,w,h=nodes[a]; xx,yy,ww,hh=nodes[b]
        if side:
            sign=1 if xx>x else -1; start,end=(x+sign*w/2,y),(xx-sign*ww/2,yy)
        else:
            sign=1 if yy>y else -1; start,end=(x,y+sign*h/2),(xx,yy-sign*hh/2)
        p=FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=12,color="#476b84",linewidth=1.4)
        p.set_gid(a+"->"+b); ax.add_patch(p)
        if label:
            ax.text((start[0]+end[0])/2,(start[1]+end[1])/2,label,fontsize=8,
                    bbox=dict(facecolor="white",edgecolor="none",pad=1))
    ax=axes[0]; n=panel(ax,"1 | Four hybrid stages: CNN + multi-scale branch -> add -> Swin -> fused BCHW")
    for i,c in enumerate(channels):
        x=2.2+4.45*i; src="DWI" if i==0 else f"S{i} input"
        box(ax,n,src,x,6,f"DWI {size} x {size}" if i==0 else f"previous fused S{i}")
        box(ax,n,f"CNN{i+1}",x-.95,4.5,"stem + layer1" if i==0 else f"CNN layer{i+1}",w=1.8)
        box(ax,n,f"MS{i+1}",x+.95,4.5,"PatchEmbed\n(2,4,8,16) conv + ECA" if i==0 else f"MS{i+1}: norm/BCHW\nconv branches + ECA",w=1.9)
        box(ax,n,f"add{i+1}",x,3,"add CNN + MS")
        box(ax,n,f"swin{i+1}",x,1.7,f"Swin {i+1} tokens\nNHWC -> BCHW")
        box(ax,n,f"S{i+1}",x,.3,f"S{i+1}: {c} x {size//2**(i+1)} x {size//2**(i+1)}")
        for a,b in ((src,f"CNN{i+1}"),(src,f"MS{i+1}"),(f"CNN{i+1}",f"add{i+1}"),(f"MS{i+1}",f"add{i+1}"),(f"add{i+1}",f"swin{i+1}"),(f"swin{i+1}",f"S{i+1}")): edge(ax,n,a,b)
        if i:
            # Route previous output to the actual next-stage input through the
            # gap between stage columns, avoiding all branch boxes.
            from matplotlib.path import Path as PlotPath
            px,py,pw,ph=n[f"S{i}"]; tx,ty,tw,th=n[src]
            gap=(px+pw/2+tx-tw/2)/2
            route=PlotPath([(px+pw/2,py),(gap,py),(gap,ty),(tx-tw/2,ty)],
                           [PlotPath.MOVETO,PlotPath.LINETO,PlotPath.LINETO,PlotPath.LINETO])
            arrow=FancyArrowPatch(path=route,arrowstyle="-|>",mutation_scale=12,color="#476b84")
            arrow.set_gid(f"S{i}->"+src); ax.add_patch(arrow)
    idx=1
    if lite and detail:
        ax=axes[idx]; idx+=1; n=panel(ax,"2 | PPM: S4 bypass + pooled branches, upsample, concat, bottleneck")
        box(ax,n,"S4",9,6,f"S4 {channels[-1]} x {size//16} x {size//16}")
        for i,scale in enumerate(scales):
            x=2.2+i*4.3
            box(ax,n,f"pool{scale}",x,4.5,f"AvgPool {scale} x {scale}\n1x1 conv {width} / ReLU")
            box(ax,n,f"up{scale}",x,2.8,f"bilinear to {size//16} x {size//16}")
            edge(ax,n,"S4",f"pool{scale}"); edge(ax,n,f"pool{scale}",f"up{scale}")
        box(ax,n,"concat",9,1.2,f"concat S4 + {len(scales)} branches\n{channels[-1]+len(scales)*width} channels",w=4)
        for scale in scales: edge(ax,n,f"up{scale}","concat")
        from matplotlib.path import Path as PlotPath
        sx,sy,sw,sh=n["S4"]; cx,cy,cw,ch=n["concat"]
        route=PlotPath([(sx+sw/2,sy),(17.5,sy),(17.5,cy),(cx+cw/2,cy)],
                       [PlotPath.MOVETO,PlotPath.LINETO,PlotPath.LINETO,PlotPath.LINETO])
        bypass=FancyArrowPatch(path=route,arrowstyle="-|>",mutation_scale=12,color="#476b84")
        bypass.set_gid("S4->concat"); ax.add_patch(bypass)
        box(ax,n,"bottleneck",9,-.2,f"3x3 bottleneck -> P4 ({width})",w=4); edge(ax,n,"concat","bottleneck")
    ax=axes[idx]; idx+=1
    n=panel(ax,f"{idx} | "+("Lite FPN: lateral + top-down, refinement, four-level fusion" if lite else "Original decoder: S4/S3 -> decode4; S2 -> decode3; S1 -> decode2"))
    for i,c in enumerate(channels): box(ax,n,f"S{i+1}",2.2+4.45*i,6,f"S{i+1}: {c} x {size//2**(i+1)} x {size//2**(i+1)}")
    if lite:
        for i in range(3):
            x=2.2+4.45*i
            box(ax,n,f"L{i+1}",x,4.4,f"1x1 lateral -> {width}")
            box(ax,n,f"P{i+1}",x,2.5,f"add lateral + up(P{i+2})\n3x3 refinement -> P{i+1}")
            edge(ax,n,f"S{i+1}",f"L{i+1}"); edge(ax,n,f"L{i+1}",f"P{i+1}")
        box(ax,n,"PPM",15.55,4.4,f"PPM {scales}\nbottleneck -> {width}")
        box(ax,n,"P4",15.55,2.5,"P4: PPM output")
        edge(ax,n,"S4","PPM"); edge(ax,n,"PPM","P4")
        for i in (4,3,2): edge(ax,n,f"P{i}",f"P{i-1}",True,"bilinear up")
        box(ax,n,"fusion",9,.5,f"upsample P1..P4 to {size//2} x {size//2}; concat {4*width} -> 3x3 fusion {width}",w=11)
        for i in range(1,5): edge(ax,n,f"P{i}","fusion")
        box(ax,n,"window",9,-.55,f"1x1 classifier {classes}; bilinear -> window logits {size} x {size}",w=10,h=.55); edge(ax,n,"fusion","window")
    else:
        for key,x,y,c in (("decode4",13.3,4.4,channels[2]),("decode3",8.85,2.8,channels[1]),("decode2",4.4,1.2,channels[0])):
            box(ax,n,key,x,y,f"{key}: upconv + skip concat\n3x3 conv / BN / ReLU -> {c}",w=3.6)
        for a,b in (("S4","decode4"),("S3","decode4"),("decode4","decode3"),("S2","decode3"),("decode3","decode2"),("S1","decode2")): edge(ax,n,a,b)
        box(ax,n,"decode0",11,-.2,f"decode0: bilinear x2 + 1x1 conv\nwindow logits {classes} x {size} x {size}",w=5); edge(ax,n,"decode2","decode0",True)
        if detail:
            ax=axes[idx]; idx+=1; n=panel(ax,f"{idx} | Decoder block: deeper input and next shallower skip")
            for key,x,y,label in (("deep",3,5,"higher-depth input"),("skip",13,5,"next shallower fused skip"),("up",3,3,"2x2 transposed conv / stride 2"),("cat",9,1.5,"channel concat (2 x output channels)"),("conv",9,0,"3x3 conv -> BN -> ReLU")): box(ax,n,key,x,y,label,w=5)
            for a,b in (("deep","up"),("up","cat"),("skip","cat"),("cat","conv")): edge(ax,n,a,b)
    ax=axes[idx]; n=panel(ax,f"{idx+1} | Window output and source-space processing: separate products")
    for key,x,y,label in (("logits",3,5,f"window logits {classes} x {size} x {size}"),("aggregate",9,5,"production: unflip / mirror mean\nGaussian tile logit mean"),("source",15,5,"remove symmetric padding\nsource-space class argmax"),("diagnostic",3,2,"diagnostic magnitude\nmean(abs(S4 channels))"),("gaussian",9,2,"upsample per window\nGaussian overlap mean"),("display",15,2,"remove padding -> source slice\ndisplay only; not saved logits replay")): box(ax,n,key,x,y,label,w=5,h=1.2)
    for a,b in (("logits","aggregate"),("aggregate","source"),("diagnostic","gaussian"),("gaussian","display")): edge(ax,n,a,b,True)
    ax.text(1,0,"Saved masks are read separately; diagnostic capture has no mirror TTA. Dimensions use the current model / explicit contract.",fontsize=11)
    fig.tight_layout(h_pad=2); fig.savefig(output,dpi=110,bbox_inches="tight"); plt.close(fig)


def run(args):
    import os
    import shutil
    import tempfile
    import SimpleITK as sitk
    import torch
    from standalone_nnunet2d.alignment_evidence import validate_checkpoint_alignment_metadata
    from standalone_nnunet2d.data.preprocessing import z_score_normalize
    from standalone_nnunet2d.predict import _read_checkpoint, _load_model
    from standalone_nnunet2d.training.official_config import DEFAULT_RUN_STATE
    info = inspect(args)
    if args.check:
        print(json.dumps({"status":"METADATA ONLY; checkpoint content and geometry PENDING",
                          "cases":len(info["rows"]), "selected":select_cases(info["rows"]),
                          "output":str(info["output"])},indent=2))
        return 0
    _, metadata = _read_checkpoint(info["paths"]["checkpoint"])
    name = _identity(metadata, info["config"], info["manifest"])
    run_state, _ = validate_checkpoint_alignment_metadata(metadata)
    if run_state == DEFAULT_RUN_STATE and not args.allow_pending:
        raise ValueError("pending checkpoint requires explicit --allow-pending")
    if run_state != info["policy"]["run_state"]:
        raise ValueError("checkpoint and prediction manifest run_state mismatch")
    for cid in info["rows"]:
        headers=[]
        for key in ("images","labels","predictions"):
            reader=sitk.ImageFileReader(); reader.SetFileName(str(info[key][cid])); reader.ReadImageInformation(); headers.append(reader)
        check_geometry(headers[0],headers[1],cid)
        check_geometry(headers[0],headers[2],cid)
    device=torch.device(args.device)
    model, loaded=_load_model(info["paths"]["checkpoint"],device)
    if _identity(loaded,info["config"],info["manifest"]) != name:
        raise ValueError("loaded checkpoint identity changed")
    high,low=select_cases(info["rows"])
    items=[]
    for group, ids in (("High Dice",high),("Low Dice",low)):
        for cid in ids:
            image=sitk.ReadImage(str(info["images"][cid]))
            gt=sitk.ReadImage(str(info["labels"][cid]))
            pred=sitk.ReadImage(str(info["predictions"][cid]))
            check_geometry(image,gt,cid); check_geometry(image,pred,cid)
            raw=sitk.GetArrayFromImage(image)
            gt_array=sitk.GetArrayFromImage(gt)>0
            pred_array=sitk.GetArrayFromImage(pred)>0
            slices=select_slices(gt_array)
            maps,native,middle=_slice_features(model,z_score_normalize(raw),slices,device) if slices else ({},None,None)
            panels={z:(raw[z].copy(),gt_array[z].copy(),pred_array[z].copy(),maps[z]) for z in slices}
            items.append(dict(group=group,cid=cid,panels=panels,slices=slices,
                              native=native,intermediate_native=middle,
                              provenance={"source_slice_indices":slices,
                                          "window_count":native["window_count"] if native else 0,
                                          "padding":"actual offsets in native mappings; full-source z-score",
                                          "tile_step_size":.5,"mirroring":False},
                              geometry={k:getattr(image,"Get"+k)() for k in ("Size","Spacing","Origin","Direction")}))
    output=info["output"]
    if not output.parent.is_dir():
        raise ValueError(f"output parent does not exist: {output.parent}")
    staging=Path(tempfile.mkdtemp(prefix=f".{output.name}-",dir=output.parent))
    try:
        source_status = _source_statement(args.confirm_prediction_checkpoint, args.prediction_checkpoint_declaration)
        note = "Diagnostic checkpoint features; saved prediction source " + source_status
        _summary_figure(items,info["rows"],model,3,staging/"summary.png",
                        trainer_name="nnUNetTrainerTopK10",source_note=note)
        _native_channel_figure(items,staging/"feature_channels.png")
        _native_channel_figure(items,staging/"feature_channels_64x64.png",layer="intermediate_native")
        _architecture(model,name,staging/"architecture_overview.png")
        _architecture(model,name,staging/"architecture_detail.png",detail=True)
        txt=[f"Dataset501 standalone {name}, fold 0 diagnostic report; not five-fold OOF or clinical evidence.",
             f"Checkpoint: {info['paths']['checkpoint']}",f"Checkpoint SHA256: {_hash_file(info['paths']['checkpoint'])}",
             f"Config: {info['paths']['config']}",f"Prediction manifest: {info['paths']['manifest']}",
             f"Prediction provenance: {source_status}",
             "Resolved configuration (as supplied):", json.dumps(info["config"],ensure_ascii=False,indent=2),
             "Training data backend: " + str(info["config"].get("data_source", {}).get("type", "unknown") if isinstance(info["config"].get("data_source"),dict) else "unknown"),
             "Diagnostic prediction entry: source NIfTI / full-volume z-score / axial windows; separate from training backend.",
             f"Checkpoint run_state: {run_state}; manifest run_state: {info['policy']['run_state']}",
             "Features: post-Swin fused forward_features[3] and [2] via decoder inputs; actual BCHW shapes below.",
             "Source mapping: full-volume z-score, source axial slices, symmetric minimum-window padding, 0.5-step windows, Gaussian overlap for magnitude; no mirroring or GT inference input. Diagnostic feature is not a replay of saved prediction.",
             "Native: one shared window per case, fixed 8 channels/layer copied to CPU; per-channel min-max color normalization. Summary: per-slice percentile normalization; not absolute cross-case scale.",
             "Selection: full-case Dice high 3 and low 3 distinct cases; max 3 GT-positive slices per case; GT affects display selection only.",
             f"Metrics: {info['paths']['metrics_dir']}; F2 definition: {info['summary'].get('f2_mode','unknown')}; AVD percent, HD95 mm; missing remains missing.",
             "Metric definitions from evaluate_segmentation_metrics.py: Dice=2TP/(2TP+FP+FN); IoU=TP/(TP+FP+FN); recall=TP/(TP+FN). F2 paper mode=5TP/(5TP+4FP+FN); standard mode=5TP/(5TP+FP+4FN), selected by source summary f2_mode. AVD=abs(predicted volume-GT volume)/GT volume x 100%, with source empty-GT handling. LCD=absolute lesion-count difference. HD95=95th percentile symmetric surface distance in mm. Existing values are read only; none are recomputed here.",
             "Evidence: local diagnostic generation; real checkpoint, case geometry, provenance and visual approval require server validation.",
             "Summary metrics:",json.dumps(info["summary"],ensure_ascii=False,indent=2),"Model repr:",repr(model)]
        for item in items:
            txt.extend([f"{item['group']}: {item['cid']} slices={item['slices']}",
                        "Metrics: "+json.dumps(info["rows"][item["cid"]]),
                        "Geometry: "+json.dumps(item["geometry"]),
                        "Feature mapping: "+json.dumps(item["provenance"]),
                        "Native deep mapping: "+json.dumps(_native_mapping(item["native"])),
                        "Native middle mapping: "+json.dumps(_native_mapping(item["intermediate_native"]))])
        (staging/"report.txt").write_text("\n".join(txt)+"\n",encoding="utf-8")
        protected_output(output,(*info["paths"].values(),))
        os.rename(staging,output)
    finally:
        if staging.exists(): shutil.rmtree(staging)
    return 0
