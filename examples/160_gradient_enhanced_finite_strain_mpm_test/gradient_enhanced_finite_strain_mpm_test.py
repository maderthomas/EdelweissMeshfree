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
"""MPM with the gradient-enhanced finite-strain cell, material point and damage material.

A plane strain block (20 x 4) of material points on a background grid of GradientEnhancedFiniteStrain/Quad4
cells is compressed by 2 % from the right, twenty times the damage threshold of Marmot's
GradientEnhancedCompressibleNeoHookeDamage. The grid is taller than the block, so that the lateral expansion
keeps every material point inside it. Checks: the nonlocal field exceeds the damage threshold, the imposed
shortening is reached, and the material point displacements match the gold file.
"""
import argparse

import numpy as np
import pytest
from edelweissfe.config.linsolve import getLinSolverByName
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import StepFailed

from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators import (
    rectangulargridgenerator,
    rectangularmpgenerator,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver
from edelweissmeshfree.stepactions.dirichlet import Dirichlet

KAPPA0 = 1e-3
SHORTENING = 0.4


def run_sim():
    dimension = 2

    journal = Journal()

    mpmModel = MPMModel(dimension)

    rectangulargridgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.0,
        l=20.0,
        y0=-1.0,
        h=6.0,
        nX=10,
        nY=3,
        cellProvider="LagrangianMarmotCell",
        cellType="GradientEnhancedFiniteStrain/Quad4",
    )

    material = {
        "material": "GradientEnhancedCompressibleNeoHookeDamage",
        # K, G, kappa0, kappaF, l, rho
        "properties": np.array([3500.0, 1500.0, KAPPA0, 1e-2, 2.0, 1.0]),
    }

    rectangularmpgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.1,
        l=19.8,
        y0=0.1,
        h=3.8,
        nX=20,
        nY=8,
        mpProvider="marmot",
        mpType="GradientEnhancedFiniteStrain/PlaneStrain",
        material=material,
    )

    mpmModel.prepareYourself(journal)
    mpmModel.nodeFields["displacement"].createFieldValueEntry("dU")
    mpmModel.nodeFields["nonlocal damage"].createFieldValueEntry("dU")

    allCells = mpmModel.cellSets["all"]
    allMPs = mpmModel.materialPointSets["all"]

    mpmManager = SmartMaterialPointManager(allCells, allMPs, dimension, options={"KDTreeLevels": 3})

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)
    fieldOutputController.addPerMaterialPointFieldOutput("displacement", allMPs, "displacement")
    fieldOutputController.addPerMaterialPointFieldOutput("nonlocal damage", allMPs, "nonlocal damage")
    fieldOutputController.initializeJob()

    dirichletLeft = Dirichlet(
        "left", mpmModel.nodeSets["rectangular_grid_left"], "displacement", {0: 0.0}, mpmModel, journal
    )
    dirichletFix = Dirichlet(
        "fix", mpmModel.nodeSets["rectangular_grid_leftBottom"], "displacement", {1: 0.0}, mpmModel, journal
    )
    dirichletRight = Dirichlet(
        "right", mpmModel.nodeSets["rectangular_grid_right"], "displacement", {0: -SHORTENING}, mpmModel, journal
    )

    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 1e-1, 1e-1, 1e-3, 100, journal)

    nonlinearSolver = NonlinearQuasistaticSolver(journal)

    iterationOptions = {"max. iterations": 12, "critical iterations": 6, "allowed residual growths": 3}

    linearSolver = getLinSolverByName("pardiso", {})

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            mpmModel,
            fieldOutputController,
            mpmManagers=[mpmManager],
            dirichlets=[dirichletLeft, dirichletFix, dirichletRight],
            userIterationOptions=iterationOptions,
        )
    except StepFailed as e:
        journal.errorMessage(str(e), "StepFailed")
        raise
    finally:
        fieldOutputController.finalizeJob()

    return mpmModel


def results(mpmModel):
    u = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
    n = np.array([mp.getResultArray("nonlocal damage") for mp in mpmModel.materialPoints.values()])
    return u, n


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim(assert_gold):
    mpmModel = run_sim()
    u, n = results(mpmModel)

    assert n.max() > 5 * KAPPA0, "the nonlocal field must drive the material well into damage"
    assert u[:, 0].min() < -0.9 * SHORTENING, "the right end must follow the imposed shortening"

    assert_gold(u, np.loadtxt("gold.csv"))


if __name__ == "__main__":
    mpmModel = run_sim()
    u, n = results(mpmModel)
    print("max nonlocal field", n.max(), " min u_x", u[:, 0].min())

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    if args.create_gold:
        np.savetxt("gold.csv", u)
