"""Contract of the FiLM noise conditioner (Phase 5)."""

from __future__ import annotations

import pytest
import torch

from configs.sparc_config import build_sparc_config
from models.film import FiLMConditioner, apply_film
from models.sparc_net import build_model

SITES = (96, 144, 192, 144, 96)
BATCH = 3


def _sigmas(batch: int = BATCH, gauss: float = 0.024, speckle: float = 0.165):
    return torch.full((batch, 1), gauss), torch.full((batch, 1), speckle)


def test_shapes_and_site_count() -> None:
    film = FiLMConditioner(SITES)
    coefficients = film(*_sigmas())
    assert len(coefficients) == len(SITES)
    for (gamma, beta), channels in zip(coefficients, SITES):
        assert gamma.shape == (BATCH, channels, 1, 1)
        assert beta.shape == (BATCH, channels, 1, 1)


def test_parameter_count_matches_the_analytic_formula() -> None:
    film = FiLMConditioner(SITES)
    assert sum(p.numel() for p in film.parameters()) == FiLMConditioner.parameter_count(SITES)
    assert FiLMConditioner.parameter_count(SITES) == 46_656


def test_identity_at_initialisation() -> None:
    """gamma = beta = 0 so the modulation is exactly the identity on step one."""
    film = FiLMConditioner(SITES)
    for gamma, beta in film(*_sigmas()):
        assert float(gamma.detach().abs().max()) == 0.0
        assert float(beta.detach().abs().max()) == 0.0
    x = torch.randn(BATCH, SITES[0], 8, 8)
    torch.testing.assert_close(apply_film(x, film(*_sigmas())[0]), x)


def test_apply_film_broadcasts_over_space() -> None:
    x = torch.randn(2, 6, 5, 7)
    gamma = torch.randn(2, 6, 1, 1)
    beta = torch.randn(2, 6, 1, 1)
    expected = x * (1.0 + gamma) + beta
    torch.testing.assert_close(apply_film(x, (gamma, beta)), expected)


def test_apply_film_passes_through_when_unconditioned() -> None:
    x = torch.randn(2, 6, 5, 7)
    torch.testing.assert_close(apply_film(x, None), x)


def test_apply_film_rejects_a_channel_mismatch() -> None:
    x = torch.randn(2, 6, 5, 7)
    with pytest.raises(ValueError, match="channels but the feature map"):
        apply_film(x, (torch.randn(2, 8, 1, 1), torch.randn(2, 8, 1, 1)))


def test_gradients_reach_every_conditioner_parameter() -> None:
    film = FiLMConditioner(SITES)
    # Zero-initialised heads produce zero gradients into the trunk on the first step,
    # so perturb them: the property under test is connectivity, not initialisation.
    for head in film.heads:
        torch.nn.init.normal_(head.weight, std=0.02)
    total = sum((g.square().mean() + b.square().mean()) for g, b in film(*_sigmas()))
    total.backward()
    for name, parameter in film.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_conditioning_actually_depends_on_sigma() -> None:
    """Different noise levels must produce different coefficients once trained."""
    torch.manual_seed(0)
    film = FiLMConditioner(SITES)
    for head in film.heads:
        torch.nn.init.normal_(head.weight, std=0.05)
    low = film(*_sigmas(batch=1, gauss=0.001, speckle=0.05))[0][0]
    high = film(*_sigmas(batch=1, gauss=0.060, speckle=0.19))[0][0]
    assert float((low - high).abs().max()) > 1e-4


def test_sparc_xl_output_responds_to_the_noise_estimate() -> None:
    """End-to-end: the whole model must be sensitive to (sigma_g, sigma_s).

    Drives the NoiseHead's own prediction by changing the input's noise content, which
    is the only path the conditioner has.
    """
    torch.manual_seed(0)
    model = build_model("sparc-xl").eval()
    for head in model.film.heads:
        torch.nn.init.normal_(head.weight, std=0.05)
    base = torch.rand(1, 1, 128, 128)
    with torch.no_grad():
        quiet = model(base)
        noisy = model(base + torch.randn_like(base) * 0.3)
    assert float((quiet - noisy).abs().max()) > 1e-4


def test_film_sites_follow_the_trunk_widths() -> None:
    config = build_sparc_config("sparc-xl")
    assert config.film_sites == (96, 144, 192, 144, 96)
    assert build_sparc_config("sparc-xl-moderate").film_sites == (80, 136, 176, 136, 80)
    assert build_sparc_config("sparc-base").film_sites == (48, 96, 160, 96, 48)


def test_film_requires_the_noise_head() -> None:
    # use_vst is switched off so the FiLM guard is the one under test; sparc-xl enables
    # both and the stabiliser's own guard would otherwise fire first.
    with pytest.raises(ValueError, match="use_film requires use_noise_head"):
        build_sparc_config("sparc-xl", use_noise_head=False, use_vst=False)


def test_vst_requires_the_noise_head() -> None:
    with pytest.raises(ValueError, match="use_vst requires use_noise_head"):
        build_sparc_config("sparc-base", use_vst=True, use_noise_head=False)


def test_rejects_an_odd_hidden_width() -> None:
    with pytest.raises(ValueError, match="film_hidden must be"):
        build_sparc_config("sparc-xl", film_hidden=63)
