import sys
import torch
import numpy as np

from experiments.genseg30.models import Generator, pad_plane, train_step


def test_native_pad_and_unpad():
    plane = np.ones((9, 11), dtype=np.float32)
    padded = pad_plane(plane, 32, -1)
    assert padded.shape == (32, 32)
    assert np.array_equal(padded[:9, :11], plane)
    assert padded[-1, -1] == -1


def test_upstream_checkpoint_roundtrip_arch_and_import_isolation(tmp_path):
    torch.set_num_threads(1)
    before_path = sys.path.copy()
    before_modules = {k: v for k, v in sys.modules.items() if k.startswith('architecture_pix2pix')}
    generator = Generator(width=1, downs=5, seed=17, device='cpu')
    tensor = torch.zeros((1, 1, 32, 32))
    generator.net.eval()
    with torch.no_grad():
        output = generator.net(tensor).clone()
    checkpoint = tmp_path / 'generator.pt'
    generator.save(checkpoint, {'fixture': True})
    restored, metadata = Generator.load(checkpoint, 'cpu')
    restored.net.eval()
    with torch.no_grad():
        assert torch.equal(output, restored.net(tensor))
        assert torch.equal(output, generator.net(tensor))  # second module cannot reset first globals
    assert metadata == {'fixture': True}
    for name in ('conv_arch', 'upconv_arch'):
        value = getattr(restored.networks, name)
        assert torch.equal(value, getattr(generator.networks, name))
        assert not value.requires_grad
    assert restored.networks.arch_parameters()[0] is restored.networks.upconv_arch
    assert restored.networks.arch_parameters()[1] is restored.networks.conv_arch
    assert sys.path == before_path
    assert {k: v for k, v in sys.modules.items() if k.startswith('architecture_pix2pix')} == before_modules


def test_actual_upstream_gan_one_cpu_update():
    torch.set_num_threads(1)
    generator = Generator(width=1, downs=5, seed=3, device='cpu')
    discriminator = generator.discriminator(width=1)
    optimizer_g = torch.optim.Adam(generator.net.parameters(), lr=2e-5, betas=(.5, .999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-5, betas=(.5, .999))
    before = next(generator.net.parameters()).detach().clone()
    losses = train_step(generator, discriminator, optimizer_g, optimizer_d,
                        torch.ones(1, 1, 32, 32), torch.zeros(1, 1, 32, 32), 100.)
    assert all(np.isfinite(x) for x in losses.values())
    assert not torch.equal(before, next(generator.net.parameters()))
