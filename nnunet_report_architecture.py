"""Draw original PlainConvUNet report architecture from a restored network.

No inference or checkpoint loading occurs here. The caller owns output staging.
"""

from pathlib import Path


NAVY = "#17364c"
INK = "#355166"
LINE = "#536e7d"
BLUE = "#e5f0fa"
GREEN = "#e1f3e9"
ORANGE = "#ffe4b8"
PURPLE = "#ebe5f8"
BG = "#f7f9fc"


def _metadata(network, patch_size, final_stage, intermediate_stage):
    encoder, decoder = network.encoder, network.decoder
    count = len(encoder.stages)
    if (count < 2 or final_stage != count - 1
            or not 0 <= intermediate_stage < final_stage
            or len(encoder.output_channels) != count
            or len(encoder.strides) != count
            or len(decoder.stages) != count - 1
            or len(decoder.transpconvs) != count - 1
            or len(decoder.seg_layers) != count - 1
            or len(patch_size) != 2):
        raise ValueError("inconsistent PlainConvUNet diagram metadata")
    size = [int(v) for v in patch_size]
    if any(v <= 0 for v in size):
        raise ValueError("invalid PlainConvUNet patch size")
    sizes = []
    for stride in encoder.strides:
        pair = (stride, stride) if isinstance(stride, int) else stride
        if len(pair) != 2 or any(int(v) < 1 for v in pair):
            raise ValueError("invalid PlainConvUNet encoder stride")
        size = [(v + int(st) - 1) // int(st) for v, st in zip(size, pair)]
        sizes.append(tuple(size))
    if sizes[intermediate_stage] != (64, 64):
        raise ValueError("intermediate PlainConvUNet diagram stage is not native 64x64")
    return tuple(int(v) for v in encoder.output_channels), sizes


def _box(ax, x, y, w, h, label, color, fs=9, bold=False):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=.08", edgecolor=INK,
        facecolor=color, linewidth=1.1))
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fs,
            color=NAVY, weight="bold" if bold else "normal", linespacing=1.08)


def _arrow(ax, points, color=LINE, lw=1.2):
    for first, second in zip(points[:-2], points[1:-1]):
        ax.plot((first[0], second[0]), (first[1], second[1]),
                color=color, lw=lw, solid_capstyle="round")
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=11))


def _shape(size):
    return f"{size[0]}×{size[1]}"


def draw_overview(ax, network, patch_size, final_stage, intermediate_stage):
    """Compact U-shaped module flow for an independent overview PNG."""
    channels, sizes = _metadata(network, patch_size, final_stage, intermediate_stage)
    count = len(channels)
    top = 2.27 + 1.15 * (count - 1)
    ys = [top - i*1.15 for i in range(count)]
    height = top + 2.35
    ax.set_xlim(0, 18.5)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.set_facecolor(BG)
    ax.text(.25, height-.48, "Original nnUNetTrainerTopK10 | PlainConvUNet",
            fontsize=19, weight="bold", color=NAVY)
    ax.text(.25, height-.88,
            f"RESTORED NETWORK METADATA  |  patch {_shape(patch_size)}  |  {count} encoder stages / {count-1} decoder steps",
            fontsize=9.5, color=INK)
    ax.text(2.2, top+1.00, "ENCODER", fontsize=11.3, weight="bold", color=NAVY)
    ax.text(9.9, top+1.00, "DECODER | transpose conv + skip concat + conv block",
            fontsize=11.3, weight="bold", color=NAVY)
    _box(ax, .18, ys[0], 1.18, .68, "DWI input", "#eef1f4", 8.6)
    _arrow(ax, [(1.36, ys[0]+.34), (1.65, ys[0]+.34)])
    for index, y in enumerate(ys):
        capture = ("HOOK A · native 64×64" if index == intermediate_stage else
                   "HOOK B · final pre-decoder" if index == final_stage else "")
        label = f"S{index} | {_shape(sizes[index])} | {channels[index]} ch"
        if capture:
            label += "\n" + capture
        _box(ax, 1.65, y, 3.15, .68, label, ORANGE if capture else BLUE,
             8.3 if capture else 9, bool(capture))
        if index < count-1:
            _arrow(ax, [(3.225, y), (3.225, ys[index+1]+.68)])
    for index, y in enumerate(ys[:-1]):
        _box(ax, 10.0, y, 3.10, .68,
             f"D{index} | {_shape(sizes[index])} | {channels[index]} ch",
             GREEN, 9, True)
        _arrow(ax, [(4.80, y+.34), (10.0, y+.34)])
        ax.text(7.0, y+.43, f"skip S{index}", fontsize=8.2, color=INK,
                bbox=dict(facecolor=BG, edgecolor="none", pad=.4))
    _arrow(ax, [(4.80, ys[-1]+.34), (11.55, ys[-1]+.34),
                (11.55, ys[-2])], color="#946f34", lw=1.5)
    for index in range(count-2, 0, -1):
        _arrow(ax, [(11.55, ys[index]+.68), (11.55, ys[index-1])],
               color="#368168", lw=1.5)
    n_classes = network.decoder.seg_layers[-1].out_channels
    _box(ax, 13.52, ys[0], 1.95, .68,
         f"seg_layers[-1]\n1×1 main head · {n_classes} logits", PURPLE, 8.1, True)
    _box(ax, 15.83, ys[0], 2.37, .68, "Single inference\nlogit tensor", PURPLE, 8.8, True)
    _arrow(ax, [(13.10, ys[0]+.34), (13.52, ys[0]+.34)])
    _arrow(ax, [(15.47, ys[0]+.34), (15.83, ys[0]+.34)])
    ax.text(.42, 1.39,
            f"S{final_stage} → D{final_stage-1} → … → D0 → main head; S{final_stage-1}..S0 enter matching decoder steps.",
            fontsize=10, color=NAVY, weight="bold")
    ax.text(.42, .89,
            "Detailed view: ① encoder captures  ② local decoder operations  ③ training and inference heads.",
            fontsize=9.4, color=INK)
    ax.text(.42, .43, "Original U-Net skip decoder. Diagram values come from the loaded network, without a synthetic forward pass.",
            fontsize=9, color="#8c5b22")


def _panel(ax, height, title, note):
    from matplotlib.patches import FancyBboxPatch
    ax.set_xlim(0, 20)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((.05, .05), 19.9, height-.1,
        boxstyle="round,pad=0,rounding_size=.18", facecolor="white",
        edgecolor="#d2e0e8", lw=1.2))
    ax.text(.40, height-.42, title, fontsize=14.4, weight="bold", color=NAVY)
    ax.text(.40, height-.78, note, fontsize=8.8, color=INK)


def draw_detail(network, patch_size, final_stage, intermediate_stage, output):
    """Three numbered, independent panels; no wires cross panel boundaries."""
    import matplotlib.pyplot as plt
    channels, sizes = _metadata(network, patch_size, final_stage, intermediate_stage)
    count = len(channels)
    enc_h, dec_h, head_h = 4.55, 2.25 + 1.34*(count-1), 4.35
    fig = plt.figure(figsize=(21, 8.6 + 1.65*(count-1)), facecolor=BG)
    fig.suptitle("Original nnUNetTrainerTopK10 | detailed PlainConvUNet view",
                 x=.035, y=.987, ha="left", fontsize=19.5,
                 weight="bold", color=NAVY)
    fig.text(.035, .963,
             f"RESTORED NETWORK METADATA | patch {_shape(patch_size)} | {count} encoder stages | {count-1} decoder steps",
             fontsize=10.0, color=INK)
    grid = fig.add_gridspec(3, 1, left=.03, right=.97, bottom=.02,
                            top=.943, height_ratios=[enc_h, dec_h, head_h], hspace=.11)
    enc = fig.add_subplot(grid[0])
    _panel(enc, enc_h, "① Encoder | stages and two feature captures",
           "The native 64×64 stage and final pre-decoder stage are independently marked.")
    gap = .25
    width = (19.2 - (count-1)*gap) / count
    starts = [.4 + index*(width+gap) for index in range(count)]
    for index, x in enumerate(starts):
        capture = "\nHOOK A" if index == intermediate_stage else "\nHOOK B" if index == final_stage else ""
        _box(enc, x, 1.86, width, .80,
             f"S{index} · {_shape(sizes[index])}\n{channels[index]} ch{capture}",
             ORANGE if capture else BLUE, 7.8 if count > 8 else 8.2, bool(capture))
        stride = network.encoder.strides[index]
        pair = (stride, stride) if isinstance(stride, int) else stride
        enc.text(x+width/2, 1.43, f"stride {tuple(pair)}",
                 ha="center", fontsize=7.8, color=INK)
        if index:
            _arrow(enc, [(starts[index-1]+width, 2.26), (x, 2.26)], lw=1.0)
    enc.text(.43, .51,
             f"DWI {_shape(patch_size)} → S0 → … → S{final_stage}; S0..S{final_stage-1} provide decoder skips.",
             fontsize=8.8, color=INK)

    dec = fig.add_subplot(grid[1])
    _panel(dec, dec_h, "② Decoder | one local operation at each skip scale",
           "Each row: ConvTranspose2d → matching encoder skip concat → StackedConvBlocks. Named tensors link rows.")
    top = dec_h - 2.20
    for step in range(count-1):
        skip = count-2-step
        y = top-step*1.34
        source = f"S{final_stage} bottleneck" if step == 0 else f"D{skip+1}"
        boxes = [(.42, 2.20, source, ORANGE if step == 0 else GREEN),
                 (2.99, 3.02, f"ConvTranspose2d → {_shape(sizes[skip])}", GREEN),
                 (6.40, 3.13, f"concat [up, S{skip}]", GREEN),
                 (9.96, 3.13, "StackedConvBlocks", GREEN),
                 (13.54, 2.22, f"D{skip} · {channels[skip]} ch", GREEN)]
        for x, w, label, color in boxes:
            _box(dec, x, y, w, .57, label, color,
                 8.0 if x in (2.99, 9.96) else 8.7, x == 13.54)
        for first, second in zip(boxes[:-1], boxes[1:]):
            _arrow(dec, [(first[0]+first[1], y+.285), (second[0], y+.285)], lw=1.05)
        _box(dec, 6.82, y+.68, 2.28, .34, f"encoder S{skip} skip", BLUE, 7.5)
        _arrow(dec, [(7.96, y+.68), (7.96, y+.57)], lw=1.0)
        dec.text(16.15, y+.28, f"{_shape(sizes[skip])} · step {step}",
                 va="center", fontsize=8.3, color=INK)
    dec.text(.42, .51,
             f"D{final_stage-1}..D1 feed the next row. D0 feeds the inference main head in ③.",
             fontsize=8.8, color=INK)

    heads = fig.add_subplot(grid[2])
    _panel(heads, head_h, "③ Output heads | training deep supervision and inference",
           "Each decoder stage has a seg_layers head. Inference with deep supervision disabled returns only the D0 main output.")
    rows = [(2.30, [(f"TRAIN: D{final_stage-1} … D0", ORANGE),
                    (f"seg_layers[0..{final_stage-1}]", PURPLE),
                    ("deep-supervision logits list\nhighest resolution first", PURPLE),
                    ("TopK10 loss\ntraining only", ORANGE)]),
            (1.05, [("INFER: D0", GREEN), ("seg_layers[-1]", PURPLE),
                    (f"main logits · {network.decoder.seg_layers[-1].out_channels} classes", PURPLE),
                    ("single tensor returned", GREEN)])]
    positions = [.42, 4.80, 9.18, 13.56]
    for y, blocks in rows:
        for x, (label, color) in zip(positions, blocks):
            _box(heads, x, y, 3.87, .66, label, color, 8.6, x == .42)
        for index in range(3):
            _arrow(heads, [(positions[index]+3.87, y+.33),
                           (positions[index+1], y+.33)], lw=1.05)
    heads.text(.42, .42,
               "Historical saved-prediction checkpoint and TTA provenance are separate from fresh diagnostic features.",
               fontsize=8.7, color=INK)
    try:
        fig.savefig(Path(output), dpi=150, facecolor=BG)
    finally:
        plt.close(fig)
