"""Server CLI: no automatic quality approval or nnU-Net protocol changes."""
import argparse
from pathlib import Path

from .contract import Source, require, sha256
from .pipeline import ACKNOWLEDGEMENT, approve_pool, audit_pool, export_pool, train_generator


def parser():
    root = argparse.ArgumentParser(description='Dataset501 fold0 offline conditional-GAN extra 2D slices')
    commands = root.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'train', 'export', 'audit', 'approve'):
        sub = commands.add_parser(name)
        sub.add_argument('--source', type=Path, required=True, help='Original nnUNetPlans_2d folder, read-only')
        sub.add_argument('--raw-root', type=Path, required=True, help='Protected nnUNet_raw root, never read here')
        if name in ('train', 'export'):
            sub.add_argument('--output', type=Path, required=True, help='Fresh derived directory outside source/repo/raw')
            sub.add_argument('--seed', type=int, default=20261014 if name == 'train' else 20261015)
            sub.add_argument('--device', default='cpu', help='Explicit cpu or cuda:0 on training server')
        if name == 'train':
            sub.add_argument('--epochs', type=int, default=200, help='Generator-only maximum complete passes')
            sub.add_argument('--max-steps', type=int, default=5000, help='Generator-only update budget; whichever limit first')
            sub.add_argument('--width', type=int, default=64)
            sub.add_argument('--downs', type=int, default=8, choices=(5, 6, 7, 8))
            sub.add_argument('--discriminator-width', type=int, default=64)
            sub.add_argument('--lr', type=float, default=2e-5)
            sub.add_argument('--l1-weight', type=float, default=100.)
        elif name == 'export':
            sub.add_argument('--run', type=Path, required=True, help='Completed generator run directory')
        elif name != 'preflight':
            sub.add_argument('--pool', type=Path, required=True)
            if name == 'audit':
                sub.add_argument('--require-quality', action='store_true')
                sub.add_argument('--manifest-sha256', help='Optional independently recorded immutable manifest hash')
            else:
                sub.add_argument('--reviewer', required=True)
                sub.add_argument('--acknowledgement', required=True, choices=[ACKNOWLEDGEMENT],
                                 help='Explicit assertion AFTER reviewing training-derived image-mask grids')
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        source = Source(args.source, args.raw_root)
        print('DECLARED DATASET501 GENERATOR/POOL WORKFLOW: real train-source mode; '
              'unit-test fixtures are synthetic engineering evidence only.', flush=True)
        if args.command in ('preflight', 'train'):
            source.require_no_unpack_writes()
        if args.command == 'preflight':
            print('COMPATIBILITY PASS: Blosc2 unpack is a no-op; fixed fold0 76/19; metadata only. '
                  'Array contents and resources must still be checked before training.')
        elif args.command == 'train':
            train_generator(source, args.output, epochs=args.epochs, max_steps=args.max_steps,
                            width=args.width, downs=args.downs, discriminator_width=args.discriminator_width,
                            seed=args.seed, device=args.device, lr=args.lr, l1_weight=args.l1_weight)
        elif args.command == 'export':
            export_pool(source, args.run, args.output, seed=args.seed, device=args.device)
        elif args.command == 'audit':
            manifest = audit_pool(source, args.pool, require_quality=args.require_quality,
                                  expected_sha256=args.manifest_sha256)
            print(f"AUDIT PASS: real_train_slices={manifest['original_slices']}, "
                  f"extra_2d_slices={manifest['synthetic_slices']}, real_train_patients={len(source.train)}")
            print('manifest_sha256=' + sha256(args.pool / 'manifest.json'))
        else:
            digest = approve_pool(source, args.pool, args.reviewer, args.acknowledgement)
            print('EXPLICIT HUMAN QUALITY APPROVAL RECORDED: manifest_sha256=' + digest)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        root.exit(2, 'GenSeg30 refused: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
