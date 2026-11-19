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
"""Tests of the nodal projection (nodal F-bar) of the DisplacementFBar cells: exact global tangent (finite differences,
NeoHooke and J2, with and without the aggregated MPM), patch test, volume conservation, symmetric positive
semidefinite tangent at a zero increment."""

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.systemmatrixassembly import VIJSystemMatrixAssembly
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.config.mplibrary import getMaterialPointClass
from edelweissmeshfree.constraints.aggregatedmpm import AggregatedMaterialPointMethod
from edelweissmeshfree.constraints.nodalprojection import NodalProjection
from edelweissmeshfree.generators import rectangulargridgenerator
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from edelweissmeshfree.sets.materialpointset import MaterialPointSet

pytest.importorskip("edelweissmeshfree.cells.marmotcell.lagrangianmarmotcell")

NEOHOOKE = {"material": "COMPRESSIBLENEOHOOKE", "properties": np.array([833.0, 385.0, 1.0])}
J2 = {"material": "FINITESTRAINJ2PLASTICITY", "properties": np.array([833.0, 385.0, 6.0, 6.0, 0.0, 0.0, 1, 1.0])}


def _model(material, aggregated):
    """A 5 x 4 grid of unit FBar cells; material points, off the cell centres, fill a staircase domain with a cell of
    two material points and a cell with a single one."""

    journal = Journal()
    model = MPMModel(2)
    rectangulargridgenerator.generateModelData(
        model, journal, x0=0.0, l=5.0, y0=0.0, h=4.0, nX=5, nY=4, cellProvider="LagrangianMarmotCell",
        cellType="DisplacementFBar/Quad4",
    )  # fmt: skip
    factory = getMaterialPointClass("marmot")
    rng = np.random.default_rng(3)
    number = 1
    for i in range(4):
        for j in range(4 if i < 2 else 3):
            for k, (a, b) in enumerate(((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))):
                if i == 3 and b > 0.5 and j == 2:
                    continue  # a cell with two material points
                if i == 3 and j == 1 and k > 0:
                    continue  # a cell with a single material point
                xy = np.array([[i + a + 0.15 * rng.uniform(-1, 1), j + b + 0.15 * rng.uniform(-1, 1)]])
                model.materialPoints[number] = factory("Displacement/PlaneStrain", number, xy, 0.25, material)
                number += 1
    model.materialPointSets["all"] = MaterialPointSet("all", list(model.materialPoints.values()))
    model.prepareYourself(journal)
    model.mpmManagers.append(
        SmartMaterialPointManager(model.cellSets["all"], model.materialPointSets["all"], 2, options={"KDTreeLevels": 2})
    )
    if aggregated:
        model.multiPointConstraints["agmpm"] = AggregatedMaterialPointMethod(
            "agmpm", model, journal, projectionMode="nodal"
        )
    else:
        model.cellProjections["nodalJ"] = NodalProjection("nodalJ", model, journal)
    model.updateConnectivity()
    return model, model.buildDiscreteSystem()


def _couplings(system):
    return [c for c in system.constraints if getattr(c, "projectsOverCells", False)]


def _assemble(system, dU):
    """Residual (internal forces) and tangent as NIST assembles them: material points, cells, then the couplings."""

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


@pytest.mark.parametrize("aggregated", [False, True], ids=["standalone", "agmpm"])
def test_one_projector_and_one_coupling_per_node(aggregated):
    model, system = _model(NEOHOOKE, aggregated)
    assert len(system.cellPatchProjections) == 1
    projector = system.cellPatchProjections[0]
    assert isinstance(projector, NodalProjection)
    assert len(projector.nodes) == len({n for c in system.activeCells for n in c.nodes})
    # one coupling per independent node: every node without aggregation, the interior ones with it
    assert (projector._nSlaves > 0) == aggregated
    assert len(_couplings(system)) == len(projector.nodes) - projector._nSlaves


@pytest.mark.parametrize("aggregated", [False, True], ids=["standalone", "agmpm"])
@pytest.mark.parametrize("material, scale", [(NEOHOOKE, 0.03), (J2, 0.01)], ids=["neohooke", "j2"])
def test_exact_tangent_across_the_cells(material, scale, aggregated):
    model, system = _model(material, aggregated)
    dU = _increment(system, scale)
    P, K = _assemble(system, dU)

    if material is J2:
        assert any(mp.getResultArray("alphaP")[0] > 0 for mp in model.materialPoints.values())
    # the material points receive different (smoothed) volume changes
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
    # without the coupling through the nodes, the tangent would be wrong
    coupling = np.zeros_like(K)
    for c in _couplings(system):
        Kc = np.zeros((c.nDof, c.nDof))
        c.computeKernels(None, None, None, Kc, None)
        idx = system.locations[c]
        coupling[np.ix_(idx, idx)] += Kc
    assert np.linalg.norm(coupling) > 1e-2 * np.linalg.norm(KFD)


@pytest.mark.parametrize("aggregated", [False, True], ids=["standalone", "agmpm"])
def test_patch_test_and_volume_conservation(aggregated):
    model, system = _model(NEOHOOKE, aggregated)
    H = np.array([[0.02, -0.03], [0.01, 0.04]])
    dU = system.locations.constructDofVector()
    dU[:] = 0.0
    # homogeneous increment: dU_a = H X_a on every node of the active cells
    for cell in system.activeCells:
        idx = system.locations[cell]
        X = np.array([n.coordinates for n in cell.nodes])
        perm = cell.dofIndicesPermutation
        nodeWise = (X @ H.T).ravel()
        dU[idx] = nodeWise[perm] if perm is not None else nodeWise
    _assemble(system, dU)
    assert np.allclose(_volumeChanges(model), np.linalg.det(np.eye(2) + H), rtol=0, atol=1e-14)

    # a non-homogeneous increment: the lumped projection keeps the total volume
    dU = _increment(system, 0.03)
    projector = system.cellPatchProjections[0]
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
    assert vBar == pytest.approx(vOwn, rel=1e-13)
    assert np.ptp(smoothed) > 1e-3 and np.abs(projector.nodalValues - 1).max() > 1e-3


@pytest.mark.parametrize("aggregated", [False, True], ids=["standalone", "agmpm"])
def test_tangent_is_symmetric_positive_semidefinite_at_a_zero_increment(aggregated):
    model, system = _model(NEOHOOKE, aggregated)
    dU = _increment(system, 0.0)
    _, K = _assemble(system, dU)
    assert np.linalg.norm(K - K.T) <= 1e-12 * np.linalg.norm(K)
    assert np.linalg.eigvalsh(0.5 * (K + K.T)).min() >= -1e-10 * np.abs(K).max()
