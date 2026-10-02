"""
Spatially correlated Gaussian random field for the strength of the unconfined plane-strain specimen.

ONE FUNCTION, EVALUATED ANYWHERE.  The field is a random-Fourier-feature (spectral) representation

    g(x) = sqrt(2 / N) * sum_i cos(k_i . x + phi_i),   k_i ~ N(0, (2 / l_c^2) I),  phi_i ~ U(0, 2 pi),

of a zero-mean, unit-variance stationary Gaussian field with the SQUARED-EXPONENTIAL correlation

    rho(r) = E[g(x) g(x + r)] = exp(-|r|^2 / l_c^2).

Because g is a closed-form function of the coordinates (and not a vector on one particular grid),
the finite-element driver (element centroids) and the meshfree driver (particle centres) see the
SAME realisation for the same seed; on the 22 x 45 discretisation the two point sets coincide anyway.
With N = 4000 modes the marginal is Gaussian to the precision that matters here (CLT).

The strength factor is  s(x) = 1 + cov * g(x)  (mean 1, coefficient of variation cov).

This file exists twice, IDENTICALLY (check with sha256sum):
  EdelweissMeshfree/examples/152_ortho_finite_strain_triaxial_shear_band_test/ps_random_field.py
  Marmot/modules/materials/GradientEnhancedOrthoCDPFiniteStrain/testCases/edelweissFE/ps_random_field.py
"""

import numpy as np

DEFAULT_SEED = 20261002
DEFAULT_LC = 2.5  # correlation length [mm] (= l_d of the paper card)
DEFAULT_COV = 0.05
N_MODES = 4000


def _modes(lc, seed, nModes=N_MODES):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((nModes, 2)) * (np.sqrt(2.0) / lc)
    phi = rng.uniform(0.0, 2.0 * np.pi, nModes)
    return k, phi


def gaussian_field(xy, lc=DEFAULT_LC, seed=DEFAULT_SEED):
    """standard-normal correlated field g at the points xy (n, 2) [mm]"""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))[:, :2]
    k, phi = _modes(lc, seed)
    return np.sqrt(2.0 / len(phi)) * np.cos(xy @ k.T + phi).sum(axis=1)


def strength_factor(xy, cov=DEFAULT_COV, lc=DEFAULT_LC, seed=DEFAULT_SEED):
    """s(x) = 1 + cov g(x): the factor applied to the strengths at the points xy"""
    return 1.0 + cov * gaussian_field(xy, lc, seed)


if __name__ == "__main__":
    # the 22 x 45 cell centres of the 37 x 75 mm specimen (h = 1.6667 mm)
    W, H, nX, nY = 37.0, 75.0, 22, 45
    X, Y = np.meshgrid((np.arange(nX) + 0.5) * W / nX, (np.arange(nY) + 0.5) * H / nY, indexing="ij")
    s = strength_factor(np.column_stack([X.ravel(), Y.ravel()]))
    print(f"cell centres: mean {s.mean():.4f}, CoV {s.std() / s.mean():.4f}, min {s.min():.4f}, max {s.max():.4f}")
    # empirical correlation from many realisations at a few lags, against exp(-r^2/lc^2)
    p = np.array([[0.0, 0.0], [1.25, 0.0], [2.5, 0.0], [0.0, 5.0]])
    G = np.array([gaussian_field(p, seed=sd) for sd in range(400)])
    for j, r in zip((1, 2, 3), (1.25, 2.5, 5.0)):
        print(f"lag {r:4.2f} mm: empirical rho {np.mean(G[:, 0] * G[:, j]):+.3f}, target {np.exp(-r * r / 6.25):.3f}")
