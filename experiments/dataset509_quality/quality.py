"""Offline, training-only native-plane contrast sidecar; never changes the pool."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation

SCORING = dict(schema='native-ring-contrast/v1', inner_iterations=1,
               outer_iterations=5, min_ring_pixels=16, min_lesion_pixels=3,
               quantiles=[0.05, 0.5, 0.95], weight_floor=0.1,
               filter_threshold=0.5)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def contrast(image, label, cfg=SCORING):
    image, label = np.asarray(image), np.asarray(label)
    need(image.ndim == 2 and image.shape == label.shape, 'PAIR_SHAPE')
    need(np.isfinite(image).all() and np.isin(label, [0, 1]).all(), 'PAIR_NONFINITE_OR_LABEL')
    mask = label == 1
    size = int(mask.sum())
    # Clipped at image boundaries; zeros in the image are excluded as exterior.
    ring = (binary_dilation(mask, iterations=cfg['outer_iterations']) &
            ~binary_dilation(mask, iterations=cfg['inner_iterations']) & (image != 0))
    if size < cfg['min_lesion_pixels'] or int(ring.sum()) < cfg['min_ring_pixels']:
        return None, size
    local = image[ring].astype(np.float64)
    scale = 1.4826 * np.median(np.abs(local - np.median(local)))
    nonzero = image[image != 0].astype(np.float64)
    spread = np.percentile(nonzero, 95) - np.percentile(nonzero, 5)
    scale = max(float(scale), float(spread) * 0.01, 1e-8)
    value = float((np.median(image[mask]) - np.median(local)) / scale)
    need(np.isfinite(value), 'NONFINITE_CONTRAST')
    return value, size


def quality_weight(value, reference, floor=0.1):
    if value is None:
        return 0.0
    lo, mid, hi = reference
    width = (mid - lo) if value < mid else (hi - mid)
    # Central 90% training interval maps to [floor, 1], with smooth tails.
    distance = min(abs(value - mid) / max(width, 1e-8), 40.)
    return float(max(floor, np.exp(-0.5 * distance * distance)))


def summary(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), quantiles=np.quantile(a, [0, .05, .5, .95, 1]).tolist()) if len(a) else dict(count=0)


def build(package, pool, split_path, output):
    package, pool, output = Path(package), Path(pool), Path(output)
    need(not output.exists(), 'SIDECAR_EXISTS')
    manifest = read(package / 'train_manifest.json')
    rows = manifest['samples']
    provenance = read(package / 'provenance.private.json')
    origins = {r['sample']: r for r in provenance['samples']}
    split = read(split_path)[0]
    train, val = set(split['train']), set(split['val'])
    need(len(train) == 100 and len(val) == 28 and not train & val, 'SPLIT_CONTRACT')
    need({r['case_id'] for r in provenance['cases']} == train, 'REFERENCE_NOT_TRAIN100')
    pm = read(pool / 'manifest.json')
    need(pm['split_sha256'] == sha(split_path), 'POOL_SPLIT_CHANGED')
    publication = read(package.parent / 'publication.json')
    need(publication['train_manifest_sha256'] == sha(package / 'train_manifest.json') and
         publication['dataset_sha256'] == sha(package / 'dataset.json'), 'PUBLICATION_CHANGED')
    need(pm['publication_sha256'] == sha(package.parent / 'publication.json'), 'POOL_PACKAGE_MISMATCH')
    need(len(rows) == 2071 and len({r['path'] for r in rows}) == len(rows), 'MANIFEST_CONTRACT')
    references, real_hashes, synthetic = [], [], {cid: [] for cid in train}
    real_cases = set()
    real_sizes = []
    for row in rows:
        origin = origins[row['sample']]
        need(origin['case_id'] in train and origin['split'] == 'train', 'REFERENCE_LEAKAGE')
        path = (package / row['path']).resolve()
        need(package.resolve() in path.parents and sha(path) == row['sha256'], 'PAIR_CHANGED')
        with np.load(path, allow_pickle=False) as pair:
            value, size = contrast(pair['image'], pair['label'])
        need(row['kind'] in ('REAL', 'SYNTHETIC'), 'UNKNOWN_SOURCE')
        if row['kind'] == 'REAL':
            real_cases.add(origin['case_id'])
            real_sizes.append(size)
            real_hashes.append(row['sha256'])
            if value is not None:
                references.append(value)
        else:
            synthetic[origin['case_id']].append(dict(contrast=value, lesion_pixels=size,
                                                    native_pair_sha256=row['sha256']))
    need(real_cases == train and len(real_hashes) == 1593, 'REAL_REFERENCE_COVERAGE')
    need(len(references) >= 100, 'INSUFFICIENT_VALID_REFERENCE_SIGNAL')
    reference = np.quantile(references, SCORING['quantiles']).tolist()
    need(reference[0] < reference[1] < reference[2], 'DEGENERATE_REFERENCE_SIGNAL')
    cases, weights, sizes, valid_scores = {}, [], [], []
    for cid, entries in synthetic.items():
        if not entries:
            continue
        record = pm['cases'][cid]
        path = (pool / record['file']).resolve()
        need(pool.resolve() in path.parents and sha(path) == record['sha256'], 'POOL_CONTENT_CHANGED')
        need(len(entries) == record['count'], 'PLANE_ORDER_COUNT_MISMATCH')
        for entry in entries:
            entry['weight'] = quality_weight(entry['contrast'], reference)
            weights.append(entry['weight'])
            sizes.append(entry['lesion_pixels'])
            if entry['contrast'] is not None:
                valid_scores.append(entry['contrast'])
        cases[cid] = entries
    need(set(cases) == set(pm['cases']) and len(weights) == 478, 'POOL_COVERAGE')
    audit = dict(real_contrast=summary(references), synthetic_contrast=summary(valid_scores),
                 weights=summary(weights), real_reference_valid_fraction=len(references)/1593,
                 real_lesion_pixels=summary(real_sizes),
                 synthetic_lesion_pixels=summary(sizes), valid_fraction=len(valid_scores)/478,
                 filter_retained_fraction=float(np.mean(np.array(weights) >= SCORING['filter_threshold'])),
                 retained_lesion_pixels=summary([e['lesion_pixels'] for es in cases.values() for e in es
                                               if e['weight'] >= SCORING['filter_threshold']]))
    sidecar = dict(schema='dataset509-quality/v1', scoring=SCORING,
                   scoring_sha256=digest(SCORING), reference=reference,
                   training_reference_sha256=digest(dict(split=sha(split_path), pairs=real_hashes)),
                   package_manifest_sha256=sha(package / 'train_manifest.json'),
                   provenance_sha256=sha(package / 'provenance.private.json'),
                   pool_manifest_sha256=sha(pool / 'manifest.json'), split_sha256=sha(split_path),
                   cases=cases, audit=audit,
                   evidence='CANDIDATE_SIGNAL_ONLY_NOT_VALIDATED_IMAGE_QUALITY',
                   order_contract='segmentation512.prepare@c498118: manifest order within each source case')
    write_new(output, sidecar)
    return audit


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('package', 'pool', 'split', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.package, a.pool, a.split, a.output), allow_nan=False))
