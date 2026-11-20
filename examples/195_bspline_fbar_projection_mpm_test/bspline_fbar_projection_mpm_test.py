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

"""B-bar/F-bar projection onto the B-spline space of degree p - 1 (Elguedj, Bazilevs, Calo and Hughes, CMAME 2008) for
B-spline MPM cells (``SplineProjection``; AgMPM ``projectionMode="spline"``): Cook's membrane in plane strain, nearly
incompressible (nu = 0.499999), small load (linear regime), quadratic B-spline grid of h = 4 on [0, 48] x [0, 60],
3 x 3 material points per cell (sub-cell centroids clipped by the panel, so the inclined edges cut cells and the
aggregation is active).

1. The global L2 projection of det dF onto the degree-1 spline space, with aggregation of displacements and projection
   space, is free of volumetric locking: tip deflection 0.998 x the reference 7.770 (Q9/P1disc FE, E 250, F 100; the
   standard B-spline cell gives 0.62 x at this resolution), gold file of the MP displacements.
2. The same without aggregation (standalone ``model.cellProjections``): 0.999 x.
3. The standard B-spline cell locks: < 0.7 x.
"""

import argparse

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.config.mplibrary import getMaterialPointClass
from edelweissmeshfree.constraints.aggregatedmpm import AggregatedMaterialPointMethod
from edelweissmeshfree.constraints.splineprojection import SplineProjection
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators import rectangularbsplinegridgenerator
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

E, NU, FORCE, H, N, P = 2.5e5, 0.499999, 100.0, 4.0, 3, 2
REFERENCE = 7.770  # tip u_y for E = 250, F = 100 (Q9/P1disc FE, Richardson extrapolated)
PANEL = np.array([[0.0, 0.0], [48.0, 44.0], [48.0, 60.0], [0.0, 44.0]])
VARIANTS = ["spline-aggregated", "spline", "standard-aggregated"]


def clip(polygon, a, b):
    """Sutherland-Hodgman: the part of a polygon left of the directed line a -> b."""
    out = []
    n = b - a
    for i in range(len(polygon)):
        p, q = polygon[i], polygon[(i + 1) % len(polygon)]
        sp = n[0] * (p[1] - a[1]) - n[1] * (p[0] - a[0])
        sq = n[0] * (q[1] - a[1]) - n[1] * (q[0] - a[0])
        if sp >= 0:
            out.append(p)
        if sp * sq < 0:
            out.append(p + (q - p) * sp / (sp - sq))
    return out


def panelMaterialPoints():
    """Centroids and areas of the N x N sub-cells of every cell, clipped by the panel."""
    X, V = [], []
    d = H / N
    for i in range(int(round(48 / d))):
        for j in range(int(round(60 / d))):
            x0, y0 = i * d, j * d
            polygon = [np.array(v) for v in ((x0, y0), (x0 + d, y0), (x0 + d, y0 + d), (x0, y0 + d))]
            for k in range(4):
                polygon = clip(polygon, PANEL[k], PANEL[(k + 1) % 4])
                if len(polygon) < 3:
                    break
            if len(polygon) < 3:
                continue
            p = np.array(polygon)
            x, y, x1, y1 = p[:, 0], p[:, 1], np.roll(p[:, 0], -1), np.roll(p[:, 1], -1)
            c = x * y1 - x1 * y
            area = 0.5 * c.sum()
            if area > 1e-10 * H * H:
                X.append([((x + x1) * c).sum() / (6 * area), ((y + y1) * c).sum() / (6 * area)])
                V.append(area)
    return np.array(X), np.array(V)


class EdgeTraction(MaterialPointPointWiseDistributedLoad):
    """Dead uniform shear traction on the edge x = 48, y in [44, 60], 4 Gauss points per cell edge."""

    def __init__(self, name, model, journal, mpSet, traction):
        super().__init__(name, model, journal, mpSet, "Pressure", np.zeros(2))
        self.traction = np.asarray(traction, dtype=float)
        self.cellAt = {}
        for cell in model.cells.values():
            lower, upper = (np.asarray(v, dtype=float) for v in cell.getBoundingBox())
            if abs(upper[0] - 48.0) < 1e-9:
                self.cellAt[int(round(lower[1] / H))] = cell

    def assembleLoad(self, P, K, U_np, dU, timeStep, locations):
        gx, gw = np.polynomial.legendre.leggauss(4)
        for j in range(int(round(44 / H)), int(round(60 / H))):
            cell = self.cellAt[j]
            for g, w in zip(gx, gw):
                x = np.array([48.0, (j + 0.5 * (g + 1)) * H])
                N_ = np.asarray(cell.getInterpolationVector(np.ascontiguousarray(x))).ravel()
                P[cell] += -np.outer(N_, self.traction * 0.5 * w * H * timeStep.stepProgress).ravel()


def run_sim(variant: str):
    journal = Journal()
    model = MPMModel(2)
    cellType = "Displacement" if variant.startswith("standard") else "DisplacementFBar"
    rectangularbsplinegridgenerator.generateModelData(
        model, journal, x0=0.0, l=48.0, y0=0.0, h=60.0, nX=int(48 / H), nY=int(60 / H), order=P,
        cellProvider="BSplineMarmotCell", cellType="{:}/BSpline/{:}".format(cellType, P),
    )  # fmt: skip
    material = {
        "material": "COMPRESSIBLENEOHOOKE",
        "properties": np.array([E / (3 * (1 - 2 * NU)), E / (2 * (1 + NU)), 1.0]),
    }
    factory = getMaterialPointClass("marmot")
    X, V = panelMaterialPoints()
    for k, (x, v) in enumerate(zip(X, V)):
        model.materialPoints[k + 1] = factory("Displacement/PlaneStrain", k + 1, np.array([x]), float(v), material)
    model.materialPointSets["all"] = MaterialPointSet("all", list(model.materialPoints.values()))
    model.prepareYourself(journal)
    model.mpmManagers.append(
        SmartMaterialPointManager(model.cellSets["all"], model.materialPointSets["all"], 2, options={"KDTreeLevels": 3})
    )
    if variant == "spline-aggregated":
        model.multiPointConstraints["agmpm"] = AggregatedMaterialPointMethod(
            "agmpm", model, journal, projectionMode="spline", splineProjection="global"
        )
    elif variant == "spline":
        model.cellProjections["thetaBar"] = SplineProjection("thetaBar", model, journal, mode="global")
    else:
        model.multiPointConstraints["agmpm"] = AggregatedMaterialPointMethod("agmpm", model, journal)

    fieldOutputController = MPMFieldOutputController(model, journal)
    fieldOutputController.addPerMaterialPointFieldOutput("displacement", model.materialPointSets["all"], "displacement")
    fieldOutputController.initializeJob()
    clamp = Dirichlet(
        "clamp", model.nodeSets["rectangular_grid_left"], "displacement", {0: 0.0, 1: 0.0}, model, journal
    )
    load = EdgeTraction("shear", model, journal, model.materialPointSets["all"], [0.0, FORCE / 16.0])
    stepActions = StepActionCollection()
    stepActions["dirichlet"][clamp.name] = clamp
    stepActions["distributedload"][load.name] = load
    jobInfo = jobInfoWithMeshfreeTolerances()
    step = AdaptiveStep(
        0, model, fieldOutputController, journal, jobInfo,
        NIST(jobInfo, journal, linsolver="pardiso", **meshfreeCriterionOptions), [], stepActions,
        startInc=1.0, maxInc=1.0, minInc=1e-2, maxNumInc=10, maxIter=10, criticalIter=5, maxGrowIter=3,
    )  # fmt: skip
    try:
        step.solve()
    finally:
        fieldOutputController.finalizeJob()
    return model, X


def displacements(model):
    return np.array([mp.getResultArray("displacement") for mp in model.materialPoints.values()])


def tipRatio(model, X):
    """u_y at the corner (48, 60), from the nearest MP and its deformation gradient, scaled to E = 250, F = 100."""
    corner = np.array([48.0, 60.0])
    k = int(np.argmin(np.linalg.norm(X - corner, axis=1)))
    mp = list(model.materialPoints.values())[k]
    u = np.asarray(mp.getResultArray("displacement")).ravel()[:2]
    F = np.asarray(mp.getResultArray("deformation gradient")).reshape(3, 3)[:2, :2]
    return (u + (F - np.eye(2)) @ (corner - X[k]))[1] * E / 250.0 * 100.0 / FORCE / REFERENCE


def goldFile(variant):
    return "gold_{:}.csv".format(variant)


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_spline_projection_with_aggregation_is_free_of_locking(assert_gold):
    model, X = run_sim("spline-aggregated")
    assert tipRatio(model, X) == pytest.approx(1.0, abs=0.01)
    assert_gold(displacements(model), np.loadtxt(goldFile("spline-aggregated")), atol=1e-10)


def test_spline_projection_without_aggregation_is_free_of_locking():
    model, X = run_sim("spline")
    assert tipRatio(model, X) == pytest.approx(1.0, abs=0.01)


def test_standard_bspline_cell_locks():
    model, X = run_sim("standard-aggregated")
    assert tipRatio(model, X) < 0.7


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold files.")
    args = parser.parse_args()
    for variant in VARIANTS:
        model, X = run_sim(variant)
        print("{:22}: tip / reference = {:.4f}".format(variant, tipRatio(model, X)))
        if args.create_gold and variant == "spline-aggregated":
            np.savetxt(goldFile(variant), displacements(model))
