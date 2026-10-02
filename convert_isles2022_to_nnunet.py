"""Read-only preflight and byte-preserving ISLES2022 DWI conversion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import nibabel as nib
import numpy as np
import SimpleITK as sitk


class ConversionError(ValueError):
    """Input or output safety contract violation."""


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def within(path, root):
    return path == root or root in path.parents


def boundaries(source, output):
    source = Path(source).resolve(strict=True)
    output_arg = Path(output).absolute()
    output = output_arg.resolve()
    if not source.is_dir():
        raise ConversionError('Source root must be a directory')
    if within(source, output) or within(output, source):
        raise ConversionError('Resolved source/output overlap')
    if output.name != 'Dataset508_ISLES2022DWI':
        raise ConversionError('Resolved output basename must be exactly Dataset508_ISLES2022DWI')
    if os.path.lexists(output_arg):
        raise ConversionError('Output already exists; never overwrite or resume')
    if output.parent.exists():
        conflicts = [p.name for p in output.parent.iterdir()
                     if p.name.casefold().startswith('dataset508_')]
        if conflicts:
            raise ConversionError(f'Dataset508 sibling conflict: {conflicts}')
    return source, output


def discover(source, expected):
    if expected < 1:
        raise ConversionError('expected-cases must be positive')
    found = {'dwi': {}, 'msk': {}}
    # Walk every source directory, including junctions, without following escapes
    # or cycles. This also finds orphan masks outside the expected layout.
    seen = set()
    file_identities = {}
    stack = [source]
    pattern = re.compile(r'sub-strokecase(\d{4})_ses-0001_(dwi|msk)\.nii\.gz')
    while stack:
        directory = stack.pop()
        resolved = directory.resolve(strict=True)
        if not within(resolved, source):
            raise ConversionError(f'Source link escapes root: {directory}')
        if resolved in seen:
            raise ConversionError(f'Ambiguous directory alias/cycle: {directory}')
        seen.add(resolved)
        for path in directory.iterdir():
            target = path.resolve(strict=True)
            if not within(target, source):
                raise ConversionError(f'Source link escapes root: {path}')
            if path.is_dir():
                stack.append(path)
            elif path.name.endswith(('_dwi.nii.gz', '_msk.nii.gz')):
                match = pattern.fullmatch(path.name)
                if not match:
                    raise ConversionError(f'Unexpected DWI/mask filename: {path}')
                case, kind = match.groups()
                subject = f'sub-strokecase{case}'
                relative = Path(subject) / 'ses-0001'
                relative = (relative / 'dwi' if kind == 'dwi'
                            else Path('derivatives') / relative)
                relative /= path.name
                if path.relative_to(source) != relative:
                    raise ConversionError(f'Unexpected/ambiguous {kind} layout: {path}')
                if case in found[kind]:
                    raise ConversionError(f'Duplicate {kind}: {case}')
                # stat follows path aliases; Windows reports volume/file identity
                # as st_dev/st_ino, including for hard links (not content hashes).
                info = path.stat()
                if not info.st_ino:
                    raise ConversionError(f'File identity unavailable: {path}')
                identity = (info.st_dev, info.st_ino)
                if identity in file_identities:
                    raise ConversionError(
                        f'Duplicate source file identity: {file_identities[identity]}, {path}')
                file_identities[identity] = path
                found[kind][case] = path
    if set(found['dwi']) != set(found['msk']):
        raise ConversionError('Missing DWI or orphan/missing label: '
                              f'DWI={sorted(found["dwi"])} masks={sorted(found["msk"])}')
    if len(found['dwi']) != expected:
        raise ConversionError(f'Expected {expected} cases, found {len(found["dwi"])}')
    return [(case, found['dwi'][case], found['msk'][case])
            for case in sorted(found['dwi'])]


def validate_pair(dwi, mask):
    images = []
    for path, is_mask in ((dwi, False), (mask, True)):
        img = sitk.ReadImage(str(path))
        if img.GetDimension() != 3 or img.GetNumberOfComponentsPerPixel() != 1:
            raise ConversionError(f'Expected 3D scalar image: {path}')
        if not all(np.isfinite(v).all() for v in
                   (img.GetSpacing(), img.GetOrigin(), img.GetDirection())):
            raise ConversionError(f'Non-finite geometry: {path}')
        raw = nib.load(str(path))
        if len(raw.shape) != 3 or raw.header.get_data_dtype().fields is not None:
            raise ConversionError(f'Expected 3D scalar NIfTI: {path}')
        # ITK may sanitize NaN/Inf on reading NIfTI; inspect the stored data too.
        pixels = np.asanyarray(raw.dataobj)
        if not np.isfinite(pixels).all():
            raise ConversionError(f'Non-finite pixel values: {path}')
        if is_mask and not (np.isclose(pixels, 0, rtol=0, atol=1e-6) |
                            np.isclose(pixels, 1, rtol=0, atol=1e-6)).all():
            raise ConversionError(
                f'Label values must be 0/1 within scaled-pixel tolerance '
                f'(atol=1e-6, rtol=0): {path}')
        images.append(img)
    for getter in ('GetSize', 'GetSpacing', 'GetOrigin', 'GetDirection'):
        if getattr(images[0], getter)() != getattr(images[1], getter)():
            raise ConversionError(f'Geometry mismatch ({getter}): {dwi}, {mask}')


def copy_verified(source, destination, expected_hash):
    # Exclusive destination creation; no overwrite, even after an interruption.
    digest = hashlib.sha256()
    with source.open('rb') as src, destination.open('xb') as dst:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            dst.write(block)
            digest.update(block)
    actual = sha256(destination)
    if digest.hexdigest() != expected_hash or actual != expected_hash:
        raise ConversionError(f'Copy integrity/source change detected: {source}')
    if sha256(source) != expected_hash:
        raise ConversionError(f'Source changed during conversion: {source}')
    return actual


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def convert(source_root, output_dir, expected_cases=250, check=False):
    source, output = boundaries(source_root, output_dir)
    pairs = discover(source, expected_cases)
    records = []
    for case, dwi, mask in pairs:
        before = (sha256(dwi), sha256(mask))
        validate_pair(dwi, mask)
        if before != (sha256(dwi), sha256(mask)):
            raise ConversionError(f'Source changed during validation: {case}')
        records.append({'case_id': f'ISLES_{case}',
                        'source_dwi': str(dwi.relative_to(source)),
                        'source_label': str(mask.relative_to(source)),
                        'output_dwi': f'imagesTr/ISLES_{case}_0000.nii.gz',
                        'output_label': f'labelsTr/ISLES_{case}.nii.gz',
                        'source_dwi_sha256': before[0],
                        'source_label_sha256': before[1]})
    if check:
        return {'status': 'checked', 'num_cases': len(records)}
    # Repeat path and pairing checks immediately before any output mutation.
    boundaries(source, output)
    if pairs != discover(source, expected_cases):
        raise ConversionError('Source pairing changed during preflight')
    for record, (_, dwi, mask) in zip(records, pairs):
        if (sha256(dwi), sha256(mask)) != (record['source_dwi_sha256'],
                                          record['source_label_sha256']):
            raise ConversionError('Source changed after preflight')
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / 'imagesTr').mkdir()
        (output / 'labelsTr').mkdir()
        for record, (_, dwi, mask) in zip(records, pairs):
            for kind, path in (('dwi', dwi), ('label', mask)):
                if not within(path.resolve(strict=True), source):
                    raise ConversionError(f'Source link now escapes root: {path}')
                record[f'output_{kind}_sha256'] = copy_verified(
                    path, output / record[f'output_{kind}'],
                    record[f'source_{kind}_sha256'])
        write_json(output / 'conversion_manifest.json', {
            'status': 'complete', 'source_root': str(source),
            'output_dir': str(output), 'num_cases': len(records), 'cases': records})
        write_json(output / 'dataset.json', {
            'channel_names': {'0': 'DWI'},
            'labels': {'background': 0, 'lesion': 1},
            'numTraining': len(records), 'file_ending': '.nii.gz',
            'overwrite_image_reader_writer': 'SimpleITKIO'})
    except BaseException:
        print(f'INCOMPLETE output retained at {output}. Do not train or resume. '
              'Inspect and manually move the incomplete directory outside the '
              'nnUNet_raw parent, then rerun to a fresh absent output. '
              'This script never deletes or overwrites.', file=sys.stderr)
        raise
    return {'status': 'complete', 'num_cases': len(records)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected-cases', type=int, default=250)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = convert(args.source_root, args.output_dir, args.expected_cases, args.check)
    except (OSError, RuntimeError, ConversionError) as exc:
        print(f'FAILED: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
