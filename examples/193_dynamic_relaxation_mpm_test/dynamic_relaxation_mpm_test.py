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
"""Implicit quasi-static MPM solved by dynamic relaxation (EdelweissFE ``DynamicRelaxation``; Sutcliffe, Coombs et al.,
"A Matrix-Free, Scalable, and Robust Implicit Material Point Method", IJNME 127 (2026) e70384), without and with the
aggregation (AgMPM) and the ghost stabilisation.

The cantilever of examples 190/192: a plane strain cantilever under its own weight, clamped on the left edge of the
grid, on unit Displacement/Quad4 cells, with small cuts in its top and bottom rows and at its right end; gravity in two
increments. Dynamic relaxation needs only the residual: no tangent is assembled and no linear system is solved.

Checks: every variant converges to ||R|| / ||f_ext|| < 1e-9 and gives the solution of NIST for the same variant (the
same discrete equations); the fictitious mass from the P-wave modulus (eq. 35,
:class:`~edelweissmeshfree.solvers.pwavemass.PWaveFictitiousMass`) gives the same solution as the Gerschgorin mass
(eq. 34); the material point displacements match the gold file of each variant.
"""

import argparse

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.dynamicrelaxation import DynamicRelaxation
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.utils.exceptions import StepFailed

from edelweissmeshfree.constraints.aggregatedmpm import AggregatedMaterialPointMethod
from edelweissmeshfree.constraints.ghoststabilisation import GhostStabilisation
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators import (
    rectangulargridgenerator,
    rectangularmpgenerator,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from edelweissmeshfree.solvers.meshfreeconvergencecriterion import (
    jobInfoWithMeshfreeTolerances,
    meshfreeCriterionOptions,
)
from edelweissmeshfree.solvers.pwavemass import PWaveFictitiousMass
from edelweissmeshfree.stepactions.bodyload import BodyLoad

E = 1.0e4
NU = 0.3
#: E, nu, density
LINEAR_ELASTIC = {"material": "LINEARELASTIC/HUGHES-WINGET", "properties": np.array([E, NU, 1.0])}
GRAVITY = np.array([0.0, -0.1])
#: the ghost penalty of example 192
GHOST = {"displacement": 0.01 * E}
#: the P-wave modulus K + 4/3 G (eq. 50)
PWAVE_MODULUS = E * (1 - NU) / ((1 + NU) * (1 - 2 * NU))

VARIANTS = {
    "standard": (False, False),
    "aggregated": (True, False),
    "ghost": (False, True),
    "aggregated_ghost": (True, True),
}


def run_sim(solver: str, aggregate: bool, ghost: bool, mass: str = "gerschgorin"):
    """Run the cantilever with NIST ("nist") or dynamic relaxation ("dr"); returns the model and the solver."""

    dimension = 2
    journal = Journal()

    mpmModel = MPMModel(dimension)

    rectangulargridgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.0,
        l=10.0,
        y0=-1.0,
        h=5.0,
        nX=10,
        nY=5,
        cellProvider="LagrangianMarmotCell",
        cellType="Displacement/Quad4",
    )
    rectangularmpgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.125,
        l=7.88,
        y0=0.995,
        h=1.01,
        nX=33,
        nY=12,
        mpProvider="marmot",
        mpType="Displacement/PlaneStrain",
        material=LINEAR_ELASTIC,
    )

    mpmModel.prepareYourself(journal)

    allCells = mpmModel.cellSets["all"]
    allMPs = mpmModel.materialPointSets["all"]

    mpmModel.mpmManagers.append(SmartMaterialPointManager(allCells, allMPs, dimension, options={"KDTreeLevels": 3}))

    if aggregate:
        mpmModel.multiPointConstraints["agmpm"] = AggregatedMaterialPointMethod("agmpm", mpmModel, journal)
    if ghost:
        mpmModel.constraints["ghost"] = GhostStabilisation("ghost", mpmModel, GHOST, journal)

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)
    fieldOutputController.addPerMaterialPointFieldOutput("displacement", allMPs, "displacement")
    fieldOutputController.initializeJob()

    clamp = Dirichlet(
        "clamp", mpmModel.nodeSets["rectangular_grid_left"], "displacement", {0: 0.0, 1: 0.0}, mpmModel, journal
    )
    gravity = BodyLoad("gravity", mpmModel, journal, mpmModel.cells.values(), "BodyForce", GRAVITY)

    stepActions = StepActionCollection()
    stepActions["dirichlet"][clamp.name] = clamp
    stepActions["bodyforce"][gravity.name] = gravity

    jobInfo = jobInfoWithMeshfreeTolerances()
    if solver == "nist":
        theSolver = NIST(jobInfo, journal, linsolver="pardiso", **meshfreeCriterionOptions)
    elif mass == "gerschgorin":
        theSolver = DynamicRelaxation(jobInfo, journal, drTolerance=1e-9)
    else:
        # the rectangularmpgenerator gives every material point the volume l h / (nX nY)
        provider = PWaveFictitiousMass(mpmModel, PWAVE_MODULUS, initialVolume=lambda mp: 7.88 * 1.01 / (33 * 12))
        theSolver = DynamicRelaxation(
            jobInfo, journal, drTolerance=1e-9, fictitiousMass="provider", fictitiousMassProvider=provider
        )

    step = AdaptiveStep(
        0,
        mpmModel,
        fieldOutputController,
        journal,
        jobInfo,
        theSolver,
        [],
        stepActions,
        startInc=0.5,
        maxInc=0.5,
        minInc=1e-2,
        maxNumInc=20,
        maxIter=10,
        criticalIter=10**9,
        maxGrowIter=3,
    )

    try:
        step.solve()
    except StepFailed as e:
        journal.errorMessage(str(e), "StepFailed")
        raise
    finally:
        fieldOutputController.finalizeJob()

    return mpmModel, theSolver


def displacements(mpmModel):
    return np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])


def goldFile(variant: str):
    return "gold_{:}.csv".format(variant)


def iterations(solver):
    return [r["iterations"] for r in solver.history if r["converged"] and r["stepProgress"] > 0.0]


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_sim(assert_gold, variant):
    aggregate, ghost = VARIANTS[variant]
    mpmModel, solver = run_sim("dr", aggregate, ghost)

    # two increments, each converged to the paper's tolerance by an iteration (not a direct solve)
    its = iterations(solver)
    assert len(its) == 2 and min(its) > 50
    assert all(r["residuals"][-1] < 1e-9 for r in solver.history if r["converged"])

    # the solution of NIST for the same discrete equations
    reference, _ = run_sim("nist", aggregate, ghost)
    u, uReference = displacements(mpmModel), displacements(reference)
    np.testing.assert_allclose(u, uReference, rtol=0.0, atol=1e-5 * np.abs(uReference).max())

    assert_gold(u, np.loadtxt(goldFile(variant)), rtol=1e-6, atol=1e-10)


def test_the_pwave_mass_gives_the_same_solution_and_feels_the_small_cuts():
    gerschgorin, _ = run_sim("dr", True, False)
    pwave, aggregated = run_sim("dr", True, False, mass="pwave")
    u, uReference = displacements(pwave), displacements(gerschgorin)
    np.testing.assert_allclose(u, uReference, rtol=0.0, atol=1e-6 * np.abs(uReference).max())
    # the P-wave estimate does not see the small cuts (the Gerschgorin one does): without stabilisation they slow the
    # iteration down (the paper's Fig. 9), the aggregation removes it (here 4543 vs 1946 iterations in increment 1)
    _, standard = run_sim("dr", False, False, mass="pwave")
    assert iterations(standard)[0] > 1.5 * iterations(aggregated)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold files.")
    args = parser.parse_args()

    for variant, (aggregate, ghost) in VARIANTS.items():
        for mass in ("gerschgorin", "pwave"):
            mpmModel, solver = run_sim("dr", aggregate, ghost, mass)
            reference, _ = run_sim("nist", aggregate, ghost)
            d = np.abs(displacements(mpmModel) - displacements(reference)).max()
            print(
                "{:16s} {:11s}: DR iterations {:}, tip {:.6e}, max |u_DR - u_NIST| = {:.2e}".format(
                    variant, mass, iterations(solver), displacements(mpmModel)[:, 1].min(), d
                )
            )
            if args.create_gold and mass == "gerschgorin":
                np.savetxt(goldFile(variant), displacements(mpmModel))
