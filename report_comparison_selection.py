"""Frozen post-hoc qualitative cohort. No inference, training or split mutation."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re

DATASET = 'Dataset501_StrokeLesion'
BASELINE = 'nnUNetTrainer__nnUNetPlans__2d'
ORDER = [(g, r) for g in ('large', 'medium', 'small') for r in ('good', 'bad')]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_pairs, parse_constant=invalid)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def selection_id(document):
    # Every semantic field included. Only ID and explicit locator-only metadata excluded.
    return hashlib.sha256(canonical({k: v for k, v in document.items() if k not in ('selection_id', 'locations')})).hexdigest()


def _number(value, name, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or positive and value <= 0:
        raise ValueError('invalid ' + name)
    return value


def geometry(image):
    import numpy as np
    if image.GetDimension() != 3:
        raise ValueError('only original 3D geometry supported')
    result = {key: list(getattr(image, 'Get' + key)()) for key in ('Size', 'Spacing', 'Origin', 'Direction')}
    if any(type(v) is not int or v <= 0 for v in result['Size']):
        raise ValueError('invalid Size')
    for v in result['Spacing']:
        _number(v, 'spacing', True)
    for v in result['Origin'] + result['Direction']:
        _number(v, 'geometry')
    d = np.asarray(result['Direction']).reshape(3, 3)
    if not np.allclose(d.T @ d, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(abs(np.linalg.det(d)), 1, atol=1e-6):
        raise ValueError('direction must be orthonormal')
    return result


def _volume(path):
    """NIfTI mm units are mandatory; SITK alone does not expose unit codes."""
    import gzip
    import struct
    path = Path(path)
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rb') as handle:
        header = handle.read(540)
    if len(header) < 348:
        raise ValueError('NIfTI header missing')
    endian = '<' if struct.unpack('<i', header[:4])[0] in (348, 540) else '>'
    size = struct.unpack(endian + 'i', header[:4])[0]
    if size == 348:
        units = header[123] & 7
    elif size == 540 and len(header) >= 540:
        units = struct.unpack(endian + 'i', header[500:504])[0] & 7
    else:
        raise ValueError('unsupported NIfTI header')
    if units != 2:
        raise ValueError('physical spatial units must explicitly be millimetres')
    import SimpleITK as sitk
    image = sitk.ReadImage(str(path))
    return image, sitk.GetArrayFromImage(image)


def identity(path, root, image=None):
    path, root = Path(path).resolve(), Path(root).resolve()
    relative = path.relative_to(root).as_posix()
    if image is None:
        image, _ = _volume(path)
    return dict(locator=relative, sha256=file_hash(path), geometry=geometry(image))


def _safe_locator(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value or value.startswith('/') or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('unsafe relative locator')


def _validate(document):
    canonical(document)
    if document.get('schema_version') != 1 or type(document.get('schema_version')) is not int or document.get('dataset') != DATASET or type(document.get('fold')) is not int or document['fold'] != 0:
        raise ValueError('selection schema/dataset/fold invalid')
    if document.get('selection_id') != selection_id(document):
        raise ValueError('selection_id mismatch')
    cohort = document.get('population')
    if not isinstance(cohort, list) or len(cohort) != len(set(cohort)) or cohort != sorted(cohort) or any(not isinstance(cid, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', cid) for cid in cohort):
        raise ValueError('invalid population IDs')
    baseline = document.get('baseline', {})
    if baseline.get('trainer_plans_configuration') != BASELINE:
        raise ValueError('invalid baseline identity')
    def hash_field(value):
        if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError('invalid source fingerprint')
    hash_field(document['split_sha256'])
    for field in ('plans_sha256', 'dataset_sha256', 'metrics_sha256'):
        hash_field(baseline[field])
    fingerprints = baseline['prediction_fingerprints']
    if not isinstance(fingerprints, dict) or set(fingerprints) != set(cohort):
        raise ValueError('baseline prediction fingerprint population mismatch')
    for value in fingerprints.values(): hash_field(value)
    if baseline.get('checkpoint_status') != 'UNKNOWN; saved prediction checkpoint never loaded or independently verified':
        raise ValueError('baseline checkpoint provenance must remain UNKNOWN')
    if set(document['generator_code_sha256']) != {'report_comparison_selection.py', 'generate_report_comparison_selection.py'}:
        raise ValueError('generation code fingerprints missing')
    for value in document['generator_code_sha256'].values(): hash_field(value)
    if document.get('axis') != {'array_order': 'z/y/x (SimpleITK); not NIfTI x/y/z or anatomical plane', 'axis': 0, 'index_base': 0}:
        raise ValueError('invalid original axis contract')
    entries = document.get('entries')
    if not isinstance(entries, list) or len(entries) != 6:
        raise ValueError('exactly six selected cases required')
    if [(e.get('size_group'), e.get('baseline_role')) for e in entries] != ORDER:
        raise ValueError('selection group/order invalid')
    ids = [e.get('case_id') for e in entries]
    if len(set(ids)) != 6 or not set(ids) <= set(cohort):
        raise ValueError('duplicate or unknown selected case')
    q = document.get('grouping', {}).get('thresholds_ml', [])
    if len(q) != 2 or not 0 < _number(q[0], 'threshold') < _number(q[1], 'threshold'):
        raise ValueError('thresholds must be positive and increasing')
    for e in entries:
        if not isinstance(e.get('dice_source'), str) or not e['dice_source'].strip():
            raise ValueError('baseline Dice source missing')
        dice = _number(e.get('baseline_dice'), 'Dice')
        if not 0 <= dice <= 1:
            raise ValueError('Dice outside [0,1]')
        volume = _number(e.get('gt_volume_ml'), 'GT volume', True)
        group = 'small' if volume <= q[0] else 'medium' if volume <= q[1] else 'large'
        if group != e['size_group']:
            raise ValueError('volume/group conflict')
        for key in ('image', 'gt'):
            item = e[key]
            _safe_locator(item['locator'])
            if not isinstance(item['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
                raise ValueError('invalid content fingerprint')
            g = item['geometry']
            class Header:
                def GetDimension(self): return 3
                def __getattr__(self, name): return lambda: tuple(g[name[3:]])
            if any(len(g[k]) != n for k, n in (('Size', 3), ('Spacing', 3), ('Origin', 3), ('Direction', 9))):
                raise ValueError('invalid geometry shape')
            geometry(Header())
        import numpy as np
        image_geometry, gt_geometry = e['image']['geometry'], e['gt']['geometry']
        # Accept NIfTI metadata roundoff; voxel dimensions must remain exact.
        if image_geometry['Size'] != gt_geometry['Size'] or any(
                not np.allclose(image_geometry[key], gt_geometry[key], rtol=0.0, atol=1e-6)
                for key in ('Spacing', 'Origin', 'Direction')):
            raise ValueError('image/GT geometry conflict')
        indices = e.get('display_slices')
        if not isinstance(indices, list) or not 1 <= len(indices) <= 3 or any(type(i) is not int or not 0 <= i < e['gt']['geometry']['Size'][2] for i in indices) or indices != sorted(set(indices)):
            raise ValueError('invalid slice indices')
        rep = e.get('representative_slice')
        if type(rep) is not int or rep not in indices:
            raise ValueError('invalid representative slice')
    return document


def validate(document):
    try:
        return _validate(document)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError('invalid selection schema: ' + str(error)) from error


def load_selection(path, info):
    document = validate(read_json(path))
    if set(document['population']) != set(info['rows']) or set(info['predictions']) != set(document['population']):
        raise ValueError('frozen split population/current metrics/predictions mismatch')
    for e in document['entries']:
        for key, map_key in (('image', 'images'), ('gt', 'labels')):
            cid = e['case_id']
            if cid not in info[map_key] or not info[map_key][cid].is_file():
                raise ValueError(cid + ': selected source missing')
            if e[key]['locator'] != info[map_key][cid].name:
                raise ValueError(cid + ': relative source locator mismatch')
    info['selection'] = document
    info['selection_path'] = Path(path).resolve()
    return document


def verify_selection(info):
    import numpy as np
    document = info.get('selection')
    if not document:
        return
    info['selection_verified'] = False
    validate(document)
    for e in document['entries']:
        cid = e['case_id']
        for key, map_key in (('image', 'images'), ('gt', 'labels')):
            path = info[map_key][cid]
            if file_hash(path) != e[key]['sha256']:
                raise ValueError(cid + ': ' + key + ' content fingerprint mismatch')
            image, values = _volume(path)
            if geometry(image) != e[key]['geometry']:
                raise ValueError(cid + ': ' + key + ' geometry mismatch')
            if key == 'gt':
                if not np.isin(values, [0, 1]).all() or any(not np.any(values[z]) for z in e['display_slices']):
                    raise ValueError(cid + ': missing positive slice/binary GT')
                expected = slice_selection(values)
                if expected != (e['display_slices'], e['representative_slice']):
                    raise ValueError(cid + ': frozen slice contract mismatch')
                ml = float(np.count_nonzero(values) * math.prod(image.GetSpacing()) / 1000)
                if not math.isclose(ml, e['gt_volume_ml'], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(cid + ': frozen volume mismatch')
    info['selection_verified'] = True


def slice_selection(mask):
    import numpy as np
    area = np.count_nonzero(mask, axis=(1, 2))
    ranked = sorted(map(int, np.flatnonzero(area)), key=lambda i: (-int(area[i]), i))[:3]
    return sorted(ranked), ranked[0] if ranked else None


def choose(cases, thresholds=None):
    import numpy as np
    positive = []
    for c in cases:
        v = _number(c['gt_volume_ml'], 'volume')
        if v < 0 or not 0 <= _number(c['baseline_dice'], 'Dice') <= 1:
            raise ValueError('invalid volume/Dice')
        if v > 0: positive.append(c)
    if not positive:
        raise ValueError('no positive GT')
    q = list(map(float, thresholds if thresholds is not None else np.quantile([c['gt_volume_ml'] for c in positive], [1/3, 2/3], method='linear')))
    if len(q) != 2 or not 0 < _number(q[0], 'threshold') < _number(q[1], 'threshold'):
        raise ValueError('repeated/invalid thresholds')
    groups = {g: [] for g in ('large', 'medium', 'small')}
    for c in positive:
        v = c['gt_volume_ml']
        groups['small' if v <= q[0] else 'medium' if v <= q[1] else 'large'].append(c)
    entries = []
    for g in groups:
        members = groups[g]
        if len(members) < 2:
            raise ValueError(f'{g}: group needs two distinct cases; found {len(members)}; thresholds={q}')
        good = sorted(members, key=lambda c: (-c['baseline_dice'], c['case_id']))[0]
        bad = sorted([c for c in members if c['case_id'] != good['case_id']], key=lambda c: (c['baseline_dice'], c['case_id']))[0]
        entries.extend(dict(c, size_group=g, baseline_role=r) for c, r in ((good, 'good'), (bad, 'bad')))
    return entries, q


def report_groups(info, selector):
    if info.get('selection'):
        return [(f"{e['size_group']} | baseline-{e['baseline_role']} | baseline Dice {e['baseline_dice']:.3f}", [e['case_id']]) for e in info['selection']['entries']]
    high, low = selector(info['rows'])
    return [('High Dice', high), ('Low Dice', low)]


def selected_slices(info, cid, mask, selector):
    if info.get('selection'):
        e = next(e for e in info['selection']['entries'] if e['case_id'] == cid)
        return e['display_slices'].copy(), e['representative_slice']
    slices = selector(mask)
    return slices, slices[0] if slices else None


def save_selection(info, output):
    if not info.get('selection'):
        return ''
    if not info.get('selection_verified'):
        raise ValueError('selection full identity PENDING')
    document = validate(info['selection'])
    (Path(output) / 'comparison_selection.json').write_bytes(canonical(document) + b'\n')
    return ('\nselection_id: ' + document['selection_id'] + '\nFrozen selection: ' + json.dumps(document['entries'], ensure_ascii=False) +
            '\nPost-hoc qualitative display based on GT volume and standard baseline Dice; not random sampling or formal improvement evidence. GT never enters inference.\n')


def check_selection_output(info, output):
    if not info.get('selection'):
        return
    copy = validate(read_json(Path(output) / 'comparison_selection.json'))
    if copy != info['selection'] or ('selection_id: ' + copy['selection_id']) not in (Path(output) / 'report.txt').read_text(encoding='utf-8'):
        raise ValueError('report output incomplete: frozen selection copy/ID')


def _baseline_metadata(root):
    """Validate static supported identity only; never inspect a checkpoint."""
    from generate_nnunet_result_report import require_simpleitk_reader
    from report_metrics_table import binary_labels
    documents = {}
    for name in ('dataset', 'plans'):
        path = root / (name + '.json')
        try:
            document = read_json(path)
            canonical(document)  # Also reject numeric overflow (e.g. 1e999).
        except (ValueError, OSError) as error:
            raise ValueError(f'{path.name}: cannot read strict JSON: {error}') from error
        if not isinstance(document, dict):
            raise ValueError(f'{path.name}: JSON object required')
        documents[name] = document
    dataset, plans = documents['dataset'], documents['plans']
    if dataset.get('channel_names') != {'0': 'DWI'}:
        raise ValueError('dataset.json: single DWI channel_names {"0": "DWI"} for _0000 required')
    # Standard metadata need not declare these fields. When present, declarations
    # must agree with the exact Dataset501_StrokeLesion directory checked above.
    for name, document in documents.items():
        for field in ('name', 'dataset_name', 'dataset_id'):
            if field not in document:
                continue
            value = document[field]
            if field == 'dataset_id':
                valid = (type(value) is int and value == 501 or
                         type(value) is str and value in ('501', 'Dataset501', DATASET))
            else:
                valid = type(value) is str and value in ('StrokeLesion', 'Dataset501', DATASET)
            if not valid:
                raise ValueError(f'{name}.json: conflicting or invalid {field} for {DATASET}')
    if 'labels' not in dataset:
        raise ValueError('dataset.json: explicit binary labels required')
    try:
        foreground = binary_labels(dataset, plans)
    except ValueError as error:
        raise ValueError(f'dataset/plans binary labels: {error}') from error
    if foreground != 1:
        raise ValueError('dataset/plans labels: foreground class index must be 1')
    try:
        require_simpleitk_reader(plans)
    except ValueError as error:
        raise ValueError(f'plans.json: {error}') from error
    configurations = plans.get('configurations')
    if not isinstance(configurations, dict) or not isinstance(configurations.get('2d'), dict):
        raise ValueError('plans.json: configurations must contain a 2d object')
    architecture = configurations['2d'].get('architecture')
    if not isinstance(architecture, dict) or architecture.get('network_class_name') != 'dynamic_network_architectures.architectures.unet.PlainConvUNet':
        raise ValueError('plans.json: 2d architecture must declare exact PlainConvUNet network_class_name')


def generate(args):
    from generate_nnunet_result_report import collect, check_geometry, protected_output
    import numpy as np
    root = Path(args.baseline_model_dir).resolve()
    if root.name != BASELINE or root.parent.name != DATASET or args.fold != 0:
        raise ValueError('exact standard Dataset501 nnUNetTrainer fold 0 baseline required')
    _baseline_metadata(root)
    splits_path = Path(args.splits_file).resolve()
    splits = read_json(splits_path)
    if not isinstance(splits, list) or len(splits) != 5:
        raise ValueError('existing patient-level five-fold split required')
    val = splits[0]['val']
    train = splits[0]['train']
    if not isinstance(val, list) or len(val) != len(set(val)) or len(train) != len(set(train)) or set(val) & set(train):
        raise ValueError('invalid fixed fold-0 train/val split')
    population = sorted(val)
    images_root, labels_root = Path(args.images_dir).resolve(), Path(args.labels_dir).resolve()
    pred_root, metrics_path = Path(args.prediction_dir).resolve(), Path(args.baseline_metrics_file).resolve()
    if pred_root != root / 'fold_0' / 'validation' or metrics_path != root / 'fold_0' / 'multi_metric_evaluation' / 'case_metrics.csv':
        raise ValueError('standard baseline saved prediction/metrics paths required')
    output = protected_output(args.output_json, [root, images_root, labels_root, pred_root, metrics_path, splits_path])
    images, labels, preds = collect(images_root, input_channel=True), collect(labels_root), collect(pred_root)
    if set(preds) != set(population) or not set(population) <= set(images) or not set(population) <= set(labels):
        raise ValueError('split validation population/source/prediction mismatch')
    with metrics_path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or 'case_id' not in reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('invalid baseline metrics columns')
        rows = {}
        for row in reader:
            cid = row['case_id']
            if not cid or cid in rows or None in row or any(v is None for v in row.values()):
                raise ValueError('duplicate/malformed baseline metrics')
            rows[cid] = row
    if set(rows) != set(population):
        raise ValueError('baseline metrics/split population mismatch')
    cases, empty = [], []
    for cid in population:
        image, _ = _volume(images[cid]); gt, mask = _volume(labels[cid]); pred, prediction = _volume(preds[cid])
        check_geometry(image, gt, cid); check_geometry(image, pred, cid)
        geometry(image); geometry(gt); geometry(pred)
        if not np.isin(mask, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
            raise ValueError(cid + ': only binary foreground {0,1} supported')
        if 'dice' in rows[cid] and rows[cid]['dice'].strip() not in ('', 'NA', 'N/A'):
            dice = float(rows[cid]['dice'])
            _number(dice, cid + ' Dice')
            if not 0 <= dice <= 1: raise ValueError(cid + ': Dice outside [0,1]')
            source = 'baseline case_metrics.csv:dice'
        else:
            a, b = prediction == 1, mask == 1
            denom = int(np.count_nonzero(a)) + int(np.count_nonzero(b))
            dice = 2 * int(np.count_nonzero(a & b)) / denom if denom else 1.0
            source = 'recomputed saved prediction/GT 2TP/(2TP+FP+FN); both empty=1'
        volume = float(np.count_nonzero(mask) * math.prod(gt.GetSpacing()) / 1000)
        if not volume:
            empty.append(dict(case_id=cid, reason='empty GT; retained in full metrics'))
            continue
        slices, rep = slice_selection(mask)
        cases.append(dict(case_id=cid, baseline_dice=dice, dice_source=source, gt_volume_ml=volume,
                          image=identity(images[cid], images_root, image), gt=identity(labels[cid], labels_root, gt),
                          display_slices=slices, representative_slice=rep, fewer_than_three=len(slices) < 3))
    entries, thresholds = choose(cases, args.thresholds_ml)
    document = dict(schema_version=1, dataset=DATASET, fold=0, population=population,
                    split_sha256=file_hash(splits_path),
                    baseline=dict(trainer_plans_configuration=BASELINE, plans_sha256=file_hash(root/'plans.json'),
                                  dataset_sha256=file_hash(root/'dataset.json'), metrics_sha256=file_hash(metrics_path),
                                  prediction_fingerprints={cid: file_hash(preds[cid]) for cid in population},
                                  checkpoint_status='UNKNOWN; saved prediction checkpoint never loaded or independently verified'),
                    volume_definition='original full GT foreground count * product(mm spacing) / 1000 mL; binary {0,1}; NIfTI units mm',
                    fingerprint_definition='SHA256 exact file bytes; relocation supported; recompression/header changes intentionally require a new manifest',
                    id_definition='SHA256 strict UTF-8 canonical JSON; all fields except selection_id and locator-only locations',
                    axis={'array_order':'z/y/x (SimpleITK); not NIfTI x/y/z or anatomical plane','axis':0,'index_base':0},
                    grouping=dict(method='explicit mL thresholds' if args.thresholds_ml else 'positive GT quantiles; numpy method=linear',
                                  quantiles=[1/3,2/3], thresholds_ml=thresholds, boundaries='small V<=q1; medium q1<V<=q2; large V>q2'),
                    sorting='large/medium/small; good (-Dice,case_id); bad remaining (Dice,case_id); display indices ascending; representative max GT area then min index',
                    entries=entries, excluded=empty, generator_code_sha256={name:file_hash(Path(__file__).with_name(name)) for name in ('report_comparison_selection.py','generate_report_comparison_selection.py')},
                    locations=dict(baseline_model_dir=str(root), images_dir=str(images_root), labels_dir=str(labels_root), prediction_dir=str(pred_root), metrics_file=str(metrics_path), splits_file=str(splits_path)))
    document['selection_id'] = selection_id(document)
    validate(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: never overwrite even if another process creates the path during selection.
    with output.open('xb') as handle:
        handle.write(canonical(document) + b'\n')
    return document
