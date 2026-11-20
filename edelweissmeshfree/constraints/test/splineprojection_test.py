# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         __  __ ____  __  __
# | ____|__| | ___| |_      _____(_)___ ___|  \/  |  _ \|  \/  |
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |\/| | |_) | |\/| |
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \ |  | |  __/| |  | |
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|  |_|_|   |_|  |_|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2023 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
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
"""Tests of the projection onto the B-spline space of degree p - 1 (B-bar/F-bar projection of Elguedj et al.) for the
DisplacementFBar B-spline cells: the space, exact global tangent (finite differences; NeoHooke and J2; global, local
and lumped projection; with and without aggregation of the projection space), patch test, volume conservation,
reproduction of Q_{p-1}, symmetric positive semidefinite tangent at a zero increment."""

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.systemmatrixassembly import VIJSystemMatrixAssembly
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.config.mplibrary import getMaterialPointClass
from edelweissmeshfree.constraints.aggregatedmpm import AggregatedMaterialPointMethod
from edelweissmeshfree.constraints.splineprojection import SplineProjection, TensorBSplineSpace
from edelweissmeshfree.generators import rectangularbsplinegridgenerator
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from edelweissmeshfree.sets.materialpointset import MaterialPointSet

pytest.importorskip("edelweissmeshfree.cells.marmotcell.bsplinemarmotcell")

NEOHOOKE = {"material": "COMPRESSIBLENEOHOOKE", "properties": np.array([833.0, 385.0, 1.0])}
J2 = {"material": "FINITESTRAINJ2PLASTICITY", "properties": np.array([833.0, 385.0, 6.0, 6.0, 0.0, 0.0, 1, 1.0])}


def _inside(x, y):
    """A domain with an inclined top edge (cut cells of all sizes) on [0, 6] x [0, 5]; the grid keeps an empty layer."""
    return x < 4.6 and y < 3.3 + 0.25 * x


def _model(material, setup, mode, p, n=None):
    """A 6 x 5 B-spline grid of degree p, (p+1)^2 material points per cell (sub-cell centres, perturbed) in the domain;
    setup: "standalone" (no aggregation), "agmpm" (displacements and projection space aggregated), "agmpmNoExt"
    (displacements aggregated, projection space not)."""

    journal = Journal()
    model = MPMModel(2)
    rectangularbsplinegridgenerator.generateModelData(
        model, journal, x0=0.0, l=6.0, y0=0.0, h=5.0, nX=6, nY=5, order=p, cellProvider="BSplineMarmotCell",
        cellType="DisplacementFBar/BSpline/{:}".format(p),
    )  # fmt: skip
    factory = getMaterialPointClass("marmot")
    rng = np.random.default_rng(5)
    n = n or p + 1
    d = 1.0 / n
    number = 1
    for i in range(6 * n):
        for j in range(5 * n):
            x, y = (i + 0.5) * d, (j + 0.5) * d
            if not _inside(x, y):
                continue
            xy = np.array([[x + 0.2 * d * rng.uniform(-1, 1), y + 0.2 * d * rng.uniform(-1, 1)]])
            model.materialPoints[number] = factory("Displacement/PlaneStrain", number, xy, d * d, material)
            number += 1
    model.materialPointSets["all"] = MaterialPointSet("all", list(model.materialPoints.values()))
    model.prepareYourself(journal)
    model.mpmManagers.append(
        SmartMaterialPointManager(model.cellSets["all"], model.materialPointSets["all"], 2, options={"KDTreeLevels": 2})
    )
    if setup == "standalone":
        model.cellProjections["thetaBar"] = SplineProjection("thetaBar", model, journal, mode=mode)
    else:
        model.multiPointConstraints["agmpm"] = AggregatedMaterialPointMethod(
            "agmpm", model, journal, projectionMode="spline", splineProjection=mode,
            splineExtension=setup == "agmpm",
        )  # fmt: skip
    model.updateConnectivity()
    return model, model.buildDiscreteSystem()


def _greville(x, p, lengths=(6.0, 5.0), cells=(6, 5)):
    """The Greville abscissae of a control point of the open uniform grid (the generator places the control points
    uniformly, at i * l / (n + p - 1))."""
    out = []
    for d in range(2):
        n = cells[d] + p
        i = int(round(x[d] / (lengths[d] / (n - 1))))
        knots = np.r_[[0.0] * p, np.linspace(0, lengths[d], cells[d] + 1), [lengths[d]] * p]
        out.append(knots[i + 1 : i + p + 1].mean())
    return np.array(out)


def _couplings(system):
    return [c for c in system.constraints if getattr(c, "projectsOverCells", False)]


def _assemble(system, dU):
    locations = system.locations
    timeStep = TimeStep(1, 1.0, 1.0, 1.0, 1.0, 1.0)
    U = locations.constructDofVector()
    U[:] = dU
    system.updateMaterialPoints(dU, timeStep, lambda entities, work: work(entities))
    P = locations.constructDofVector()
    P[:] = 0.0
    K = VIJSystemMatrixAssembly(locations, 1)
    K.beginAssembly()
    for cell in system.activeCells:
        Pe = np.zeros(cell.nDof)
        cell.computeKernels(K[cell], Pe, U[cell], dU[cell], 1.0, 1.0)
        P[cell] += Pe
    for coupling in _couplings(system):
        coupling.computeKernels(U[coupling], dU[coupling], np.zeros(coupling.nDof), K[coupling], timeStep)
    K.flush()
    return np.array(P), K.assembledMatrix().toarray()


def _increment(system, scale):
    dU = system.locations.constructDofVector()
    dU[:] = scale * np.sin(1.7 * np.arange(1, len(dU) + 1))
    return dU


def _volumeChanges(model):
    return np.array(
        [
            np.linalg.det(np.asarray(mp.getResultArray("delta deformation gradient")).reshape(3, 3))
            for mp in model.materialPoints.values()
        ]
    )


@pytest.mark.parametrize("q", [0, 1, 2])
def test_space_partition_of_unity_and_transfer(q):
    space = TensorBSplineSpace([np.array([0.0, 1.0, 2.5, 3.0, 4.0]), np.array([0.0, 1.0, 2.0, 3.0])], q)
    rng = np.random.default_rng(1)
    for span in [(0, 0), (1, 2), (3, 1)]:
        x = rng.uniform([0, 0], [4, 3], size=(7, 2))
        inside = space.lines[0][span[0]] + rng.uniform(0, 1, 7) * np.diff(space.lines[0])[span[0]]
        y = np.c_[inside, np.full(7, space.lines[1][span[1]] + 0.5)]
        assert np.allclose(space.evaluate(span, y).sum(axis=1), 1.0)
        assert np.allclose(space.evaluate(span, x).sum(axis=1), 1.0)  # the extended pieces too
        # transfer: the pieces of span (1, 1) expressed on this span
        T = space.transfer(span, (1, 1))
        assert np.allclose(space.evaluate(span, x) @ T, space.evaluate((1, 1), x))


@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("setup", ["standalone", "agmpm", "agmpmNoExt"])
@pytest.mark.parametrize("mode", ["global", "local", "lumped"])
@pytest.mark.parametrize("material, scale", [(NEOHOOKE, 0.02), (J2, 0.008)], ids=["neohooke", "j2"])
def test_exact_tangent_across_the_cells(material, scale, mode, setup, p):
    model, system = _model(material, setup, mode, p)
    dU = _increment(system, scale)
    P, K = _assemble(system, dU)
    if material is J2:
        assert any(mp.getResultArray("alphaP")[0] > 0 for mp in model.materialPoints.values())
    assert np.ptp(_volumeChanges(model)) > 1e-3

    h = 1e-7
    KFD = np.zeros_like(K)
    for j in range(len(dU)):
        d = dU.copy()
        d[j] += h
        Pp, _ = _assemble(system, d)
        d[j] -= 2 * h
        Pm, _ = _assemble(system, d)
        KFD[:, j] = (Pp - Pm) / (2 * h)
    assert np.linalg.norm(K - KFD) <= 1e-6 * np.linalg.norm(KFD)


@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("setup", ["standalone", "agmpm"])
@pytest.mark.parametrize("mode", ["global", "local", "lumped"])
def test_patch_test_volume_and_reproduction(mode, setup, p):
    model, system = _model(NEOHOOKE, setup, mode, p)
    projector = system.cellPatchProjections[0]
    assert isinstance(projector, SplineProjection)
    H = np.array([[0.02, -0.03], [0.01, 0.04]])
    dU = system.locations.constructDofVector()
    dU[:] = 0.0
    for cell in system.activeCells:  # homogeneous increment: the linear field at the Greville abscissae
        idx = system.locations[cell]
        X = np.array([_greville(n.coordinates, p) for n in cell.nodes])
        perm = cell.dofIndicesPermutation
        nodeWise = (X @ H.T).ravel()
        dU[idx] = nodeWise[perm] if perm is not None else nodeWise
    _assemble(system, dU)
    assert np.allclose(_volumeChanges(model), np.linalg.det(np.eye(2) + H), rtol=0, atol=1e-12)

    dU = _increment(system, 0.02)
    _assemble(system, dU)
    smoothed = _volumeChanges(model)
    own = []
    for cell in system.activeCells:
        n = len(cell.assignedMaterialPoints)
        v, w = np.zeros(n), np.zeros(n)
        cell.computeMaterialPointProjectionContributions(np.ascontiguousarray(dU[cell]), v, w, np.zeros(n * cell.nDof))
        own += list(zip([mp.number for mp in cell.assignedMaterialPoints], v, w))
    order = {mp.number: i for i, mp in enumerate(model.materialPoints.values())}
    vOwn = sum(w * v for _, v, w in own)
    vBar = sum(w * smoothed[order[num]] for num, _, w in own)
    if mode in ("global", "lumped"):  # the constants are in the space: total volume kept
        assert vBar == pytest.approx(vOwn, rel=1e-12)
    assert np.ptp(smoothed) > 1e-3

    # reproduction of Q_{p-1} (global, local) by the projection operator itself: feed values of a polynomial
    if mode == "global" or (mode == "local" and not projector._rankDeficient):
        space = projector.space
        cells = projector._cells
        X = [np.array([np.asarray(mp.getCenterCoordinates()).ravel()[:2] for mp in c.assignedMaterialPoints])
             for c in cells]  # fmt: skip

        def f(x):
            return 1.0 + 0.3 * x[:, 0] ** (p - 1) - 0.2 * x[:, 1] ** (p - 1) + 0.05 * (x[:, 0] * x[:, 1]) ** (p - 1)

        theta = np.zeros(projector._nTheta)
        values = [f(x) for x in X]
        if mode == "global":
            b = np.zeros(projector._nTheta)
            for c in range(len(cells)):
                np.add.at(b, projector._cellIdx[c], projector._N[c].T @ (projector._w[c] * values[c]))
            theta = projector._Mlu.solve(b)
        else:
            for e in range(len(cells)):
                vg = np.concatenate([values[c] for c in projector._groups[projector._baseOf[e]]])
                np.add.at(theta, projector._cellIdx[e], projector._R[e] @ vg)
        for c in range(len(cells)):
            assert np.allclose(projector._N[c] @ theta[projector._cellIdx[c]], values[c], rtol=0, atol=1e-9)
        assert space.degree == p - 1


@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("setup", ["standalone", "agmpm"])
@pytest.mark.parametrize("mode", ["global", "lumped"])
def test_tangent_is_symmetric_positive_semidefinite_at_a_zero_increment(mode, setup, p):
    model, system = _model(NEOHOOKE, setup, mode, p)
    _, K = _assemble(system, _increment(system, 0.0))
    assert np.linalg.norm(K - K.T) <= 1e-11 * np.linalg.norm(K)
    assert np.linalg.eigvalsh(0.5 * (K + K.T)).min() >= -1e-10 * np.abs(K).max()
