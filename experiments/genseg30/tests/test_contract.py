from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.genseg30.contract import allocate_counts, check_output, validate_arrays, validate_splits


def test_allocation_exact_ceiling_and_largest_remainder():
    assert allocate_counts({'a': 1, 'b': 1, 'c': 1}) == {'a': 1, 'b': 0, 'c': 0}
    assert allocate_counts({'a': 7, 'b': 2, 'c': 1}) == {'a': 2, 'b': 1, 'c': 0}
    for count in range(1, 40):
        result = allocate_counts({'b': count, 'a': count + 1})
        assert sum(result.values()) == (3 * (2 * count + 1) + 9) // 10
    with pytest.raises(ValueError):
        allocate_counts({'a': 0})


def test_split_rejects_overlap_missing_and_mismatch_before_load():
    folds = [{'train': ['a'], 'val': ['v']}] * 5
    assert validate_splits(folds, folds, expected_sizes=(1, 1)) == (['a'], ['v'])
    with pytest.raises(ValueError):
        validate_splits([], folds, expected_sizes=(1, 1))
    with pytest.raises(ValueError):
        validate_splits([{'train': ['a'], 'val': ['a']}] * 5, folds, expected_sizes=(1, 1))
    with pytest.raises(ValueError):
        validate_splits([{'train': ['other'], 'val': ['v']}] * 5, folds, expected_sizes=(1, 1))


def test_output_rejects_ancestors_descendants_and_existing(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    with pytest.raises(ValueError):
        check_output(source / 'derived', [source])
    with pytest.raises(ValueError):
        check_output(tmp_path, [source])
    with pytest.raises(ValueError):
        check_output(source, [source])
    assert check_output(tmp_path / 'fresh', [source]) == tmp_path / 'fresh'
    link = tmp_path / 'alias'
    try:
        link.symlink_to(source, target_is_directory=True)
    except OSError:
        return
    with pytest.raises(ValueError):
        check_output(link / 'derived', [source])


def test_native_float_and_binary_ignore_contract():
    image = np.zeros((1, 1, 3, 5), dtype=np.float32)
    seg = np.zeros(image.shape, dtype=np.int16)
    seg[0, 0, 0, 0] = -1
    validate_arrays(image, seg)
    with pytest.raises(ValueError):
        validate_arrays(image.astype(np.uint8), seg)
    bad = image.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_arrays(bad, seg)
    seg[0, 0, 0, 0] = 2
    with pytest.raises(ValueError):
        validate_arrays(image, seg)

def test_input_alias_cannot_redirect_a_training_filename_to_validation(tmp_path, monkeypatch):
    from experiments.genseg30.contract import checked_file
    heldout = tmp_path / 'heldout.npz'
    heldout.write_bytes(b'not training data')
    alias = tmp_path / 'train.npz'
    try:
        alias.symlink_to(heldout)
    except OSError:
        # Replace only the OS path-resolution boundary when Windows lacks symlink privilege.
        alias.write_bytes(b'fixture alias')
        original_resolve = Path.resolve
        def resolve(path, *args, **kwargs):
            return heldout if path == alias else original_resolve(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'resolve', resolve)
    with pytest.raises(ValueError):
        checked_file(tmp_path, 'train.npz')

def test_same_size_mutation_with_restored_mtime_changes_identity(tmp_path):
    import os
    from experiments.genseg30.contract import file_identity
    path = tmp_path / 'immutable.bin'
    path.write_bytes(b'original')
    before = file_identity(path)
    stats = path.stat()
    path.write_bytes(b'mutation')
    os.utime(path, ns=(stats.st_atime_ns, stats.st_mtime_ns))
    after = file_identity(path)
    assert path.stat().st_size == stats.st_size and path.stat().st_mtime_ns == stats.st_mtime_ns
    assert before != after
