import pytest
from experiments.genseg30.cli import parser


def test_cli_requires_explicit_acknowledgement_and_source():
    with pytest.raises(SystemExit):
        parser().parse_args(['approve', '--source', '/source', '--raw-root', '/raw',
                             '--pool', '/pool', '--reviewer', 'name'])
    args = parser().parse_args(['train', '--source', '/source', '--raw-root', '/raw', '--output', '/fresh',
                               '--epochs', '200', '--max-steps', '5000', '--device', 'cpu'])
    assert args.epochs == 200 and args.max_steps == 5000
    with pytest.raises(SystemExit):
        parser().parse_args(['train', '--source', '/source', '--output', '/fresh'])

def test_cli_complete_tiny_train_export_audit_and_manual_gate(source_factory, tmp_path, capsys):
    from experiments.genseg30.cli import main
    from experiments.genseg30.pipeline import ACKNOWLEDGEMENT
    source = source_factory('b2nd')
    common = ['--source', str(source.folder), '--raw-root', str(source.raw_root)]
    run, pool = tmp_path / 'cli_run', tmp_path / 'cli_pool'
    main(['train', *common, '--output', str(run), '--epochs', '1', '--max-steps', '1',
          '--width', '1', '--downs', '5', '--discriminator-width', '1', '--device', 'cpu'])
    main(['export', *common, '--run', str(run), '--output', str(pool), '--device', 'cpu'])
    main(['audit', *common, '--pool', str(pool)])
    with pytest.raises(SystemExit) as error:
        main(['audit', *common, '--pool', str(pool), '--require-quality'])
    assert error.value.code == 2
    main(['approve', *common, '--pool', str(pool), '--reviewer', 'fixture reviewer',
          '--acknowledgement', ACKNOWLEDGEMENT])
    main(['audit', *common, '--pool', str(pool), '--require-quality'])
    output = capsys.readouterr().out
    assert 'actual_updates=1' in output and 'extra_2d_slices=24' in output
    assert 'synthetic engineering evidence only' in output

def test_cli_train_refuses_npz_before_generator_budget(source_factory, tmp_path, monkeypatch):
    from experiments.genseg30 import cli
    source = source_factory()
    def forbidden(*args, **kwargs):
        raise AssertionError('Spent generator budget before backend gate')
    monkeypatch.setattr(cli, 'train_generator', forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(['train', '--source', str(source.folder), '--raw-root', str(source.raw_root),
                  '--output', str(tmp_path / 'fresh')])
    assert error.value.code == 2 and not (tmp_path / 'fresh').exists()
