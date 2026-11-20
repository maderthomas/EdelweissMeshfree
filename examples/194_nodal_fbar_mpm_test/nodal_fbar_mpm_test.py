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
"""Nodal F-bar for implicit MPM: the volume change of every material point is smoothed through the nodes of the grid
(``NodalProjection``: lumped L2 projection of det dF_p with the material point volumes as weights onto the nodes,
interpolated back with the cell basis), with the exact tangent across the cells sharing a node, without and with
aggregation (AgMPM, ``projectionMode="nodal"``: the projection acts on the full grid, the aggregation reduces it).

1. The plane strain cantilever of example 191 (nu = 0.4999, h = 0.25, 2 x 2 material points per cell, small dead tip
   load): the nodal F-bar is free of volumetric locking, with and without aggregation (0.938 and 0.881 x the
   Timoshenko deflection, asserted within 15 %; the standard cell gives 0.025 x, see example 191), and the material
   point displacements match the gold files. With aggregation, the nodal values of the exterior nodes are extended from
   the interior ones like the displacements (the projection basis is the extended basis N E); projecting them as
   independent values (``nodalExtension=False``) imposes one volumetric constraint per node of the full grid on the
   reduced displacement space and locks (0.025 x).
2. Patch test: a homogeneous compression of a block (prescribed on all grid nodes in one increment) gives every
   material point exactly the volume change of the homogeneous deformation, on a grid with partially filled cells.
"""

import argparse

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.config.mplibrary import getMaterialPointClass
from edelweissmeshfree.constraints.aggregatedmpm import AggregatedMaterialPointMethod
from edelweissmeshfree.constraints.nodalprojection import NodalProjection
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators import rectangulargridgenerator
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from edelweissmeshfree.sets.materialpointset import MaterialPointSet
from edelweissmeshfree.solvers.meshfreeconvergencecriterion import (
    jobInfoWithMeshfreeTolerances,
    meshfreeCriterionOptions,
)
from edelweissmeshfree.stepactions.distributedload import (
    MaterialPointPointWiseDistributedLoad,
)

E, NU, LENGTH, DEPTH, TIPLOAD = 12.0e3, 0.4999, 10.0, 1.0, 0.03
NEOHOOKE = {
    "material": "COMPRESSIBLENEOHOOKE",
    "properties": np.array([E / (3 * (1 - 2 * NU)), E / (2 * (1 + NU)), 1.0]),
}
VARIANTS = [False, True]
IDS = ["nodal", "nodal-aggregated"]


def timoshenkoTipDeflection():
    """Plane strain Timoshenko cantilever under a tip load (per unit thickness), Cowper's shear coefficient."""
    planeStrainModulus, shearModulus = E / (1 - NU**2), E / (2 * (1 + NU))
    shearCoefficient = 10 * (1 + NU) / (12 + 11 * NU)
    bending = TIPLOAD * LENGTH**3 / (3 * planeStrainModulus * DEPTH**3 / 12)
    return bending + TIPLOAD * LENGTH / (shearCoefficient * shearModulus * DEPTH)


class DeadPointLoad(MaterialPointPointWiseDistributedLoad):
    """A dead point force carried by material points (as in example 191)."""

    def __init__(self, name, model, journal, materialPoints, force):
        super().__init__(name, model, journal, materialPoints, "Pressure", np.asarray(force, dtype=float))

    def assembleLoad(self, P, K, U_np, dU, timeStep, locations):
        for mp in self.mpSet:
            _, force = self.getCurrentMaterialPointLoad(mp, timeStep)
            x = np.asarray(mp.getCenterCoordinates(), dtype=float).ravel()
            for cell in mp.assignedCells:
                N = np.asarray(cell.getInterpolationVector(x)).ravel()
                P[cell] += -np.outer(N, force).ravel()


def addNodalFBar(mpmModel, journal, aggregate, nodalExtension=True):
    if aggregate:
        mpmModel.multiPointConstraints["agmpm"] = AggregatedMaterialPointMethod(
            "agmpm", mpmModel, journal, projectionMode="nodal", nodalExtension=nodalExtension
        )
    else:
        mpmModel.cellProjections["nodalJ"] = NodalProjection("nodalJ", mpmModel, journal)


def solve(mpmModel, journal, stepActions, fieldOutputController):
    jobInfo = jobInfoWithMeshfreeTolerances()
    step = AdaptiveStep(
        0,
        mpmModel,
        fieldOutputController,
        journal,
        jobInfo,
        NIST(jobInfo, journal, linsolver="pardiso", **meshfreeCriterionOptions),
        [],
        stepActions,
        startInc=1.0,
        maxInc=1.0,
        minInc=1e-2,
        maxNumInc=10,
        maxIter=10,
        criticalIter=5,
        maxGrowIter=3,
    )
    try:
        step.solve()
    finally:
        fieldOutputController.finalizeJob()


def run_sim(aggregate: bool, nodalExtension: bool = True):
    dimension = 2
    journal = Journal()
    mpmModel = MPMModel(dimension)

    # grid [0, 11] x [-1, 2] of 0.25 cells; the beam [0, 10] x [0, 1] fills 40 x 4 cells
    rectangulargridgenerator.generateModelData(
        mpmModel, journal, x0=0.0, l=11.0, y0=-1.0, h=3.0, nX=44, nY=12, cellProvider="LagrangianMarmotCell",
        cellType="DisplacementFBar/Quad4",
    )  # fmt: skip
    materialPointFactory = getMaterialPointClass("marmot")
    nX, nY = 80, 8
    dx, dy = LENGTH / nX, DEPTH / nY
    for i in range(nX):
        for j in range(nY):
            number = i * nY + j + 1
            xy = np.array([[(i + 0.5) * dx, (j + 0.5) * dy]])
            mpmModel.materialPoints[number] = materialPointFactory(
                "Displacement/PlaneStrain", number, xy, dx * dy, NEOHOOKE
            )
    mpmModel.materialPointSets["all"] = MaterialPointSet("all", list(mpmModel.materialPoints.values()))
    mpmModel.prepareYourself(journal)

    allCells = mpmModel.cellSets["all"]
    allMPs = mpmModel.materialPointSets["all"]
    mpmModel.mpmManagers.append(SmartMaterialPointManager(allCells, allMPs, dimension, options={"KDTreeLevels": 3}))
    addNodalFBar(mpmModel, journal, aggregate, nodalExtension)

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)
    fieldOutputController.addPerMaterialPointFieldOutput("displacement", allMPs, "displacement")
    fieldOutputController.initializeJob()

    tip = sorted(
        [mp for mp in mpmModel.materialPoints.values() if mp.getCenterCoordinates()[0] > LENGTH - 0.1],
        key=lambda mp: abs(mp.getCenterCoordinates()[1] - DEPTH / 2),
    )[:2]
    clamp = Dirichlet(
        "clamp", mpmModel.nodeSets["rectangular_grid_left"], "displacement", {0: 0.0, 1: 0.0}, mpmModel, journal
    )
    tipLoad = DeadPointLoad("tipload", mpmModel, journal, tip, [0.0, -TIPLOAD / 2])
    stepActions = StepActionCollection()
    stepActions["dirichlet"][clamp.name] = clamp
    stepActions["distributedload"][tipLoad.name] = tipLoad

    solve(mpmModel, journal, stepActions, fieldOutputController)
    return mpmModel, tip


def run_patch_test(aggregate: bool, stretch=(-0.02, 0.01)):
    """A block [0, 1.3] x [0, 0.9] of material points (3 x 3 per unit cell, partially filled boundary cells) on a
    grid [0, 2] x [0, 2] of 0.5 cells; all grid nodes prescribed to the homogeneous displacement diag(stretch) X."""

    dimension = 2
    journal = Journal()
    mpmModel = MPMModel(dimension)
    rectangulargridgenerator.generateModelData(
        mpmModel, journal, x0=0.0, l=2.0, y0=0.0, h=2.0, nX=4, nY=4, cellProvider="LagrangianMarmotCell",
        cellType="DisplacementFBar/Quad4",
    )  # fmt: skip
    factory = getMaterialPointClass("marmot")
    nX, nY = 13, 9
    for i in range(nX):
        for j in range(nY):
            number = i * nY + j + 1
            xy = np.array([[(i + 0.5) * 0.1, (j + 0.5) * 0.1]])
            mpmModel.materialPoints[number] = factory("Displacement/PlaneStrain", number, xy, 0.01, NEOHOOKE)
    mpmModel.materialPointSets["all"] = MaterialPointSet("all", list(mpmModel.materialPoints.values()))
    mpmModel.prepareYourself(journal)
    mpmModel.mpmManagers.append(
        SmartMaterialPointManager(
            mpmModel.cellSets["all"], mpmModel.materialPointSets["all"], dimension, options={"KDTreeLevels": 2}
        )
    )
    addNodalFBar(mpmModel, journal, aggregate)
    fieldOutputController = MPMFieldOutputController(mpmModel, journal)
    fieldOutputController.initializeJob()

    stepActions = StepActionCollection()
    for node in mpmModel.nodes.values():
        nodeSet = NodeSet("n{:}".format(node.label), [node])
        X = np.asarray(node.coordinates, dtype=float)
        bc = Dirichlet(
            "n{:}".format(node.label),
            nodeSet,
            "displacement",
            {0: stretch[0] * X[0], 1: stretch[1] * X[1]},
            mpmModel,
            journal,
        )
        stepActions["dirichlet"][bc.name] = bc
    solve(mpmModel, journal, stepActions, fieldOutputController)
    J = np.array(
        [
            np.linalg.det(np.asarray(mp.getResultArray("deformation gradient")).reshape(3, 3))
            for mp in mpmModel.materialPoints.values()
        ]
    )
    return J, (1 + stretch[0]) * (1 + stretch[1])


def displacements(mpmModel):
    return np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])


def tipDeflection(tip):
    return -np.mean([mp.getResultArray("displacement")[1] for mp in tip])


def goldFile(aggregate: bool):
    return "gold_nodal{:}.csv".format("_aggregated" if aggregate else "")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


@pytest.mark.parametrize("aggregate", VARIANTS, ids=IDS)
def test_sim(assert_gold, aggregate):
    mpmModel, tip = run_sim(aggregate)
    assert tipDeflection(tip) == pytest.approx(timoshenkoTipDeflection(), rel=0.15)
    assert_gold(displacements(mpmModel), np.loadtxt(goldFile(aggregate)), atol=1e-10)


def test_aggregated_nodal_fbar_without_extension_locks():
    _, tip = run_sim(True, nodalExtension=False)
    assert tipDeflection(tip) < 0.1 * timoshenkoTipDeflection()


@pytest.mark.parametrize("aggregate", VARIANTS, ids=IDS)
def test_patch_test(aggregate):
    J, exact = run_patch_test(aggregate)
    assert np.abs(J - exact).max() < 1e-12


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold files.")
    args = parser.parse_args()

    reference = timoshenkoTipDeflection()
    for aggregate, name in zip(VARIANTS, IDS):
        mpmModel, tip = run_sim(aggregate)
        print(
            "{:20}: tip deflection {:.6e} = {:.3f} x Timoshenko".format(
                name, tipDeflection(tip), tipDeflection(tip) / reference
            )
        )
        if args.create_gold:
            np.savetxt(goldFile(aggregate), displacements(mpmModel))
        J, exact = run_patch_test(aggregate)
        print("{:20}: patch test max |J - J_exact| = {:.2e}".format(name, np.abs(J - exact).max()))
