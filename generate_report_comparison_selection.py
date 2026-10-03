"""Generate a new frozen standard nnU-Net fold-0 qualitative display selection."""
import argparse
from pathlib import Path
from report_comparison_selection import generate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-model-dir','splits-file','images-dir','labels-dir','prediction-dir','baseline-metrics-file','output-json'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--thresholds-ml', type=float, nargs=2, help='explicit increasing positive mL thresholds; otherwise linear positive-GT terciles')
    args = parser.parse_args(argv)
    try:
        result = generate(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print('selection_id: ' + result['selection_id'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
