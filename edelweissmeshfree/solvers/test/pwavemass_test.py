# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _
# | ____|__| | ___| |_      _____(_)___ ___
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \
# |_____\__,_|\___|_| \_/\_/_\___|_|___/___/
# |  \/  | ___  ___| |__  / _|_ __ ___  ___
# | |\/| |/ _ \/ __| '_ \| |_| '__/ _ \/ _ \
# | |  | |  __/\__ \ | | |  _| | |  __/  __/
# |_|  |_|\___||___/_| |_|_| |_|  \___|\___|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#
#  Research Group for Computational Mechanics of Materials
#  Institute of Structural Engineering, BOKU University, Vienna
#
#  2023 - today
#
#  Matthias Neuner |  matthias.neuner@boku.ac.at
#  Thomas Mader    |  thomas.mader@bokut.ac.at
#
#  This file is part of EdelweissMeshfree.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMeshfree.
#  ---------------------------------------------------------------------
"""The P-wave fictitious mass (eq. 35 of Sutcliffe et al., IJNME 2026): the material point term and its geometric and
Hencky enhancements (eqs. 36 and 38)."""

from types import SimpleNamespace

import numpy as np
import pytest

from edelweissmeshfree.solvers.pwavemass import PWaveFictitiousMass


class FakeMP:
    def __init__(self, dF, Fn=np.eye(3), volume=0.5):
        self.results = {
            "delta deformation gradient": np.asarray(dF, float).ravel(),
            "deformation gradient": np.asarray(Fn, float).ravel(),
        }
        self.initialVolume = volume

    def getResultArray(self, name):
        return self.results[name]


def provider(**kw):
    return PWaveFictitiousMass(SimpleNamespace(domainSize=2), 3.0, **kw)


def test_undeformed():
    assert provider().materialPointTerms([FakeMP(np.eye(3))])[0] == pytest.approx(3.0 * 0.5)


def test_geometric_factor_and_hencky_enhancement():
    # uniaxial compression to 0.5 in x: dF^-1 has the column sum 2 -> U = 2; eps_III = ln 0.5, J = 0.5
    dF = np.diag([0.5, 1.0, 1.0])
    term = provider(henckyEnhancement=False).materialPointTerms([FakeMP(dF)])[0]
    assert term == pytest.approx(4.0 * 3.0 * 0.5 * 0.5)  # U^2 M V0 J
    enhanced = provider().materialPointTerms([FakeMP(dF)])[0]
    assert enhanced == pytest.approx(term * np.exp(-np.log(0.5)) / 0.5)


def test_a_rotation_keeps_the_geometric_factor_positive():
    c, s = np.cos(2.0), np.sin(2.0)
    dF = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    U2 = provider(henckyEnhancement=False).materialPointTerms([FakeMP(dF)])[0] / (3.0 * 0.5)
    assert U2 == pytest.approx(max(abs(c + s), abs(c - s)) ** 2)


def test_a_modulus_per_material_point():
    mps = [FakeMP(np.eye(3)), FakeMP(np.eye(3))]
    p = PWaveFictitiousMass(SimpleNamespace(domainSize=2), lambda theMPs: np.array([1.0, 2.0]))
    np.testing.assert_allclose(p.materialPointTerms(mps), [0.5, 1.0])
